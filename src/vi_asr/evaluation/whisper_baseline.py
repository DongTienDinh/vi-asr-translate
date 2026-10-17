"""Whisper-small multilingual zero-shot baseline adapter (Section 20.1)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Union

import soundfile as sf
import torch

from vi_asr.evaluation.metrics import compute_wer_cer

logger = logging.getLogger(__name__)


class WhisperBaselineEvaluator:
    """Zero-shot multilingual Whisper evaluation adapter."""

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name = model_name
        self.device = torch.device(device)

        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        logger.info("Loading Whisper baseline model: %s on %s...", model_name, device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def transcribe_audio_file(self, audio_path: str) -> str:
        """Transcribe single audio file using Whisper with Vietnamese language prompt."""
        wav_np, sr = sf.read(audio_path, dtype="float32")
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)

        input_features = self.processor(
            wav_np,
            sampling_rate=sr,
            return_tensors="pt",
        ).input_features.to(self.device)

        if self.device.type == "cuda":
            input_features = input_features.half()

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="vi", task="transcribe")
        predicted_ids = self.model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()
