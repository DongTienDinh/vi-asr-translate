"""Full Bilingual Speech-to-Text & Translation Pipeline (Audio -> Vietnamese -> English)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from vi_asr.evaluation.cascade import VinAITranslator
from vi_asr.inference.segmenter import AudioSegment, EnergyVADSegmenter
from vi_asr.inference.transcribe import ASRInferencePipeline

logger = logging.getLogger(__name__)


class BilingualCascadePipeline:
    """Full bilingual pipeline: Audio -> VAD Chunks -> ASR (Vi) -> VinAI Translate (En)."""

    def __init__(
        self,
        asr_pipeline: ASRInferencePipeline,
        translator: Optional[VinAITranslator] = None,
        segmenter: Optional[EnergyVADSegmenter] = None,
    ):
        self.asr_pipeline = asr_pipeline
        self.translator = translator
        self.segmenter = segmenter or EnergyVADSegmenter()

    def process_file(
        self,
        audio_path: Union[str, Path],
        translate: bool = True,
    ) -> List[Dict[str, Any]]:
        """Process an audio file into timed bilingual segments."""
        segments = self.segmenter.segment_file(audio_path)
        logger.info("Segmented audio into %d speech chunks", len(segments))

        results: List[Dict[str, Any]] = []
        vi_transcripts: List[str] = []

        # 1. Transcribe each chunk
        for seg in segments:
            vi_text = self.asr_pipeline.transcribe_waveform(seg.waveform)
            vi_transcripts.append(vi_text)
            results.append({
                "segment_id": seg.segment_id,
                "start_sec": round(seg.start_sec, 2),
                "end_sec": round(seg.end_sec, 2),
                "duration_sec": round(seg.duration_sec, 2),
                "vietnamese": vi_text,
                "english": "",
            })

        # 2. Translate in batch if translator provided
        if translate and self.translator is not None and vi_transcripts:
            en_translations = self.translator.translate_batch(vi_transcripts)
            for res, en_text in zip(results, en_translations):
                res["english"] = en_text

        return results
