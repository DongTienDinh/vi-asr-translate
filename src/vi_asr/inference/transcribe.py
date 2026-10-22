"""Standalone ASR Inference Pipeline (strictly audio-only, Section 2.2)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import soundfile as sf
import torch

from vi_asr.data.frontend import GlobalCMVN, KaldiFrontend
from vi_asr.data.tokenizer import Tokenizer
from vi_asr.models.ctc_asr import ViPViCTCASR
from vi_asr.models.encoder_adapter import ChunkFormerEncoderAdapter

logger = logging.getLogger(__name__)


class ASRInferencePipeline:
    """End-to-end ASR transcription pipeline.

    Audio-only contract:
    - Input: raw audio waveform [1, T] or file path.
    - Output: Vietnamese text transcript.
    - Zero teacher/text dependencies.
    """

    def __init__(
        self,
        model: ViPViCTCASR,
        tokenizer: Tokenizer,
        frontend: Optional[KaldiFrontend] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.frontend = frontend or KaldiFrontend(
            fbank_waveform_multiplier=32768.0,
            eval_dither=0.0,
        )
        self.frontend.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        tokenizer_model_path: Union[str, Path],
        cmvn_path: Optional[Union[str, Path]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> ASRInferencePipeline:
        """Instantiate pipeline from checkpoint and tokenizer artifacts."""
        tokenizer = Tokenizer(tokenizer_model_path)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

        global_cmvn = None
        resolved_cmvn_path = Path(cmvn_path) if cmvn_path else None
        if resolved_cmvn_path is None:
            model_cfg = checkpoint_config.get("model", {})
            encoder_id = model_cfg.get("encoder_id")
            encoder_revision = model_cfg.get("encoder_revision")
            if encoder_id and encoder_revision:
                from huggingface_hub import hf_hub_download

                resolved_cmvn_path = Path(
                    hf_hub_download(
                        repo_id=str(encoder_id),
                        filename="global_cmvn",
                        revision=str(encoder_revision),
                    )
                )
        if resolved_cmvn_path is None or not resolved_cmvn_path.is_file():
            raise FileNotFoundError(
                "Pinned ViP-VL global_cmvn is required for deterministic inference"
            )
        global_cmvn = GlobalCMVN.from_file(resolved_cmvn_path)

        frontend = KaldiFrontend(
            fbank_waveform_multiplier=32768.0,
            eval_dither=0.0,
            global_cmvn=global_cmvn,
        )

        adapter = ChunkFormerEncoderAdapter(
            input_size=80,
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=12,
            dropout_rate=0.0,
            encoder_layerdrop=0.0,
        )
        model = ViPViCTCASR(adapter, vocab_size=tokenizer.vocab_size, blank_id=0)

        # Load only the ASR submodule; alignment is train-only and never enters inference.
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        # Clean state dict if wrapped in TinyASRSystem or DDP
        clean_state = {}
        for k, v in state_dict.items():
            if k.startswith("asr_model."):
                clean_state[k[len("asr_model.") :]] = v
            elif k.startswith("module.asr_model."):
                clean_state[k[len("module.asr_model.") :]] = v
            elif k.startswith("alignment_module.") or k.startswith("module.alignment_module."):
                continue
            elif k.startswith("module."):
                clean_state[k[len("module.") :]] = v
            else:
                clean_state[k] = v

        model.load_state_dict(clean_state, strict=True)
        return cls(model=model, tokenizer=tokenizer, frontend=frontend, device=device)

    @torch.no_grad()
    def transcribe_waveform(self, waveform: torch.Tensor) -> str:
        """Transcribe 1D or 2D [1, T] 16kHz audio waveform tensor to text."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        feats, feat_lens = self.frontend(waveform)  # [1, F, 80], [1]
        feat_lens = feat_lens.to(self.device)
        feats = feats.to(self.device)

        hyp_ids_list = self.model.transcribe(feats, feat_lens, context_policy="full_utterance")
        token_ids = hyp_ids_list[0] if hyp_ids_list else []
        return self.tokenizer.decode(token_ids)

    def transcribe_file(self, audio_path: Union[str, Path]) -> str:
        """Transcribe audio file."""
        wav_np, sr = sf.read(str(audio_path), dtype="float32")
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)

        wav_t = torch.from_numpy(wav_np).float()
        if sr != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, 16000)
            wav_t = resampler(wav_t)

        return self.transcribe_waveform(wav_t)
