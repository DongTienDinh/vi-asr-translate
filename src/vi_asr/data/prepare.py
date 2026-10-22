"""Data preparation adapter for Hugging Face nguyendv02/ViMD_Dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from vi_asr.data.manifest import REGION_NAME_TO_ID, ManifestItem, save_manifest
from vi_asr.data.normalize import (
    compute_text_hash,
    normalize_text_asr,
    normalize_text_raw,
)
from vi_asr.utils.logging import setup_logger

logger = setup_logger("vi_asr.data.prepare")


def decode_audio_value(audio_value: Any) -> tuple[np.ndarray, int]:
    """Decode legacy HF audio dictionaries or modern torchcodec AudioDecoder values.

    Hugging Face Datasets 4.4+ yields an AudioDecoder whose ``get_all_samples``
    result stores data as ``[channels, samples]``. Older releases yield a
    dictionary with ``array`` and ``sampling_rate``. This adapter normalizes both
    contracts to one mono float32 waveform ``[samples]``.
    """
    audio_array: Any = None
    sample_rate: Any = None

    if isinstance(audio_value, dict):
        audio_array = audio_value.get("array")
        sample_rate = audio_value.get("sampling_rate")
        if audio_array is None and audio_value.get("path"):
            import soundfile as sf

            audio_array, sample_rate = sf.read(str(audio_value["path"]), dtype="float32")
    elif hasattr(audio_value, "get_all_samples"):
        decoded = audio_value.get_all_samples()
        audio_array = decoded.data
        sample_rate = decoded.sample_rate
    elif isinstance(audio_value, (str, Path)):
        import soundfile as sf

        audio_array, sample_rate = sf.read(str(audio_value), dtype="float32")

    if audio_array is None or sample_rate is None:
        raise ValueError(f"Unsupported or empty audio value: {type(audio_value)!r}")

    if hasattr(audio_array, "detach"):
        audio_array = audio_array.detach().cpu().numpy()
    waveform = np.asarray(audio_array, dtype=np.float32)

    if waveform.ndim == 2:
        # torchcodec: [channels, samples]; soundfile: [samples, channels].
        if waveform.shape[0] <= 8 and waveform.shape[1] > waveform.shape[0]:
            waveform = waveform.mean(axis=0)
        elif waveform.shape[1] <= 8:
            waveform = waveform.mean(axis=1)
        else:
            raise ValueError(f"Ambiguous 2-D audio shape: {waveform.shape}")
    elif waveform.ndim != 1:
        waveform = np.squeeze(waveform)
        if waveform.ndim != 1:
            raise ValueError(f"Expected mono/ multichannel audio, got shape {waveform.shape}")

    if waveform.size == 0:
        raise ValueError("Decoded audio is empty")
    return np.ascontiguousarray(waveform, dtype=np.float32), int(sample_rate)


def parse_sample_to_manifest_item(
    sample: Dict[str, Any],
    split: str,
    source_revision: str,
    sample_index: int,
    output_dir: Optional[Union[str, Path]] = None,
) -> Optional[ManifestItem]:
    """Convert raw dataset dictionary item into canonical ManifestItem."""
    sample_id = str(sample.get("id") or sample.get("sample_id") or f"{split}_{sample_index:06d}")

    # Text fields
    raw_text = str(sample.get("transcript") or sample.get("text") or sample.get("sentence") or "")
    text_raw = normalize_text_raw(raw_text)
    text_asr = normalize_text_asr(raw_text)
    if not text_asr:
        return None

    # Regional metadata
    region_raw = str(sample.get("region") or sample.get("dialect") or "")
    region_id = REGION_NAME_TO_ID.get(region_raw)
    province = str(sample.get("province")) if sample.get("province") else None
    gender = str(sample.get("gender")) if sample.get("gender") else None
    speaker_id = str(sample.get("speaker_id") or sample.get("speaker")) if sample.get("speaker_id") or sample.get("speaker") else None

    # Audio metadata
    audio_info = sample.get("audio")
    audio_relpath = f"audio/{split}/{sample_id}.wav"
    try:
        audio_array, sample_rate = decode_audio_value(audio_info)
    except Exception as decode_err:
        logger.warning("Could not decode audio for sample %s: %s", sample_id, decode_err)
        return None

    if output_dir is not None:
        dest_audio_path = Path(output_dir) / audio_relpath
        if not dest_audio_path.exists():
            dest_audio_path.parent.mkdir(parents=True, exist_ok=True)
            import soundfile as sf

            try:
                sf.write(str(dest_audio_path), audio_array, sample_rate, subtype="PCM_16")
            except Exception as write_err:
                logger.warning("Could not write audio file %s: %s", dest_audio_path, write_err)
                return None

    num_samples = int(audio_array.shape[0])

    duration = float(sample.get("duration", num_samples / sample_rate if sample_rate > 0 else 0.0))

    return ManifestItem(
        sample_id=sample_id,
        split=split,
        audio_relpath=audio_relpath,
        source_revision=source_revision,
        sample_rate=sample_rate,
        num_samples=num_samples,
        duration=duration,
        text_raw=text_raw,
        text_asr=text_asr,
        speaker_id=speaker_id,
        region_id=region_id,
        province=province,
        gender=gender,
        text_teacher=None,
        audio_hash=None,
        text_hash=compute_text_hash(text_asr),
    )


def prepare_manifests_from_hf(
    dataset_dict: Any,
    output_dir: Union[str, Path],
    source_revision: str = "3a5b30157034e7eadd5c75fae1a820c6f9383398",
) -> Dict[str, Path]:
    """Prepare canonical train, valid, test JSONL manifests from a HuggingFace DatasetDict."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    manifest_paths: Dict[str, Path] = {}

    for split in ["train", "validation", "test"]:
        actual_split_key = split
        if actual_split_key not in dataset_dict:
            # Map valid to validation if needed
            if split == "validation" and "valid" in dataset_dict:
                actual_split_key = "valid"
            elif split not in dataset_dict:
                logger.warning(f"Split {split} not found in dataset keys: {list(dataset_dict.keys())}")
                continue

        canonical_split = "valid" if actual_split_key in ["validation", "valid", "dev"] else actual_split_key
        ds = dataset_dict[actual_split_key]
        manifests_dir = out_p / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        try:
            count_str = f"({len(ds)} items)"
        except (TypeError, AttributeError):
            count_str = "(streaming mode)"
        logger.info(f"Processing split {actual_split_key} {count_str} -> {canonical_split}.jsonl")

        items: List[ManifestItem] = []
        for idx, sample in enumerate(ds):
            item = parse_sample_to_manifest_item(
                sample, canonical_split, source_revision, idx, output_dir=out_p
            )
            if item is not None:
                items.append(item)
            if (idx + 1) % 500 == 0:
                logger.info(f"[{canonical_split}] Processed {idx + 1} samples (collected {len(items)} items)...")

        dest_file = out_p / f"{canonical_split}.jsonl"
        save_manifest(items, dest_file)
        # Also mirror into manifests/ subdirectory for nested path loaders
        save_manifest(items, manifests_dir / f"{canonical_split}.jsonl")
        manifest_paths[canonical_split] = dest_file
        logger.info(f"Saved {len(items)} manifest items to {dest_file} and {manifests_dir / f'{canonical_split}.jsonl'}")

    return manifest_paths
