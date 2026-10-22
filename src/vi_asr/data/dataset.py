"""Dataset and sample loading for Vietnamese ASR."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import soundfile as sf
import torch
from torch.utils.data import Dataset

from vi_asr.data.augment import SpecAugment
from vi_asr.data.audio_store import SHARD_INDEX_FILENAME, ShardedZipAudioReader
from vi_asr.data.frontend import KaldiFrontend
from vi_asr.data.manifest import ManifestItem
from vi_asr.data.teacher_cache import TeacherCacheReader
from vi_asr.data.tokenizer import Tokenizer
from vi_asr.utils.logging import setup_logger

logger = setup_logger("vi_asr.data.dataset")


class ASRDataset(Dataset):
    """Dataset loading audio waveforms, extracting Kaldi filterbanks, and aligning teacher states."""

    def __init__(
        self,
        manifest_items: List[ManifestItem],
        frontend: KaldiFrontend,
        tokenizer: Tokenizer,
        audio_root: Optional[Union[str, Path]] = None,
        specaugment: Optional[SpecAugment] = None,
        teacher_reader: Optional[TeacherCacheReader] = None,
        is_train: bool = True,
    ):
        super().__init__()
        self.items = manifest_items
        self.frontend = frontend
        self.tokenizer = tokenizer
        self.audio_root = Path(audio_root) if audio_root else None
        self.specaugment = specaugment
        self.teacher_reader = teacher_reader
        self.is_train = is_train
        self.audio_store: Optional[ShardedZipAudioReader] = None
        if self.audio_root and (self.audio_root / SHARD_INDEX_FILENAME).is_file():
            self.audio_store = ShardedZipAudioReader(self.audio_root)

    def __len__(self) -> int:
        return len(self.items)

    def load_waveform(self, item: ManifestItem) -> torch.Tensor:
        """Load audio waveform tensor [1, T] at 16kHz."""
        candidates: List[Path] = []
        if self.audio_root and item.audio_relpath:
            candidates.append(self.audio_root / item.audio_relpath)
            candidates.append(self.audio_root / Path(item.audio_relpath).name)
        if item.audio_relpath:
            candidates.append(Path(item.audio_relpath))
            candidates.append(Path("/kaggle/working/artifacts/prepared") / item.audio_relpath)

        audio_path: Optional[Path] = None
        for c in candidates:
            if c.exists() and c.is_file():
                audio_path = c
                break

        if audio_path is None and self.audio_store is not None and item.audio_relpath:
            wav_np, sample_rate = self.audio_store.read(item.audio_relpath)
            if sample_rate != 16000:
                raise RuntimeError(
                    f"Prepared audio must be 16 kHz, got {sample_rate} for {item.sample_id}"
                )
            return torch.from_numpy(wav_np).unsqueeze(0)

        if audio_path is None and item.audio_relpath:
            fname = Path(item.audio_relpath).name
            kaggle_input = Path("/kaggle/input")
            if kaggle_input.exists():
                for found in kaggle_input.rglob(fname):
                    if found.is_file():
                        audio_path = found
                        break

        if audio_path is None or not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found for sample {item.sample_id}: {item.audio_relpath}")

        wav_np, sr = sf.read(str(audio_path), dtype="float32")
        if int(sr) != 16000:
            raise RuntimeError(f"Prepared audio must be 16 kHz, got {sr} for {item.sample_id}")
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)  # mono

        wav_tensor = torch.from_numpy(wav_np).unsqueeze(0)  # [1, T]
        return wav_tensor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]

        # 1. Fail closed: never replace a missing training item with another sample or silence.
        waveform = self.load_waveform(item)

        # 2. Extract 80-mel filterbank (KaldiFrontend handles scale 32768.0 and CMVN)
        feats, _ = self.frontend(waveform)
        feats = feats.squeeze(0)  # [F, 80]

        # 3. Apply SpecAugment during training if enabled
        if self.is_train and self.specaugment is not None:
            aug_out = self.specaugment(feats)
            feats = aug_out[0] if isinstance(aug_out, tuple) else aug_out
            if feats.ndim == 3:
                feats = feats.squeeze(0)

        # 4. Tokenize target transcript for CTC
        target_ids = self.tokenizer.encode(item.text_asr)
        target_tensor = torch.tensor(target_ids, dtype=torch.int64)

        sample: Dict[str, Any] = {
            "sample_id": item.sample_id,
            "feats": feats,  # [F, 80]
            "feat_length": feats.size(0),
            "targets": target_tensor,  # [U]
            "target_length": len(target_ids),
            "text_asr": item.text_asr,
        }

        # 5. Load teacher states from offline cache if available (P1/P2 variants)
        if self.teacher_reader is not None:
            if item.sample_id in self.teacher_reader:
                states, _ = self.teacher_reader.get(item.sample_id)
                sample["teacher_states"] = states  # [N, 768]
                sample["teacher_length"] = states.size(0)
            else:
                sample["teacher_states"] = torch.zeros((0, 768), dtype=torch.float32)
                sample["teacher_length"] = 0

        # 6. Region label (if available)
        if item.region_id is not None:
            sample["region_id"] = int(item.region_id)

        return sample
