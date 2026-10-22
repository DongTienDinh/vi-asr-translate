"""Unit tests verifying Audio-Only Inference Contract (Section 21.1, Invariant 11)."""

import tempfile
import unittest
from pathlib import Path

import torch

from vi_asr.data.tokenizer import train_bpe_tokenizer, Tokenizer
from vi_asr.inference.segmenter import EnergyVADSegmenter
from vi_asr.inference.transcribe import ASRInferencePipeline
from vi_asr.models.ctc_asr import ViPViCTCASR
from vi_asr.models.encoder_adapter import ChunkFormerEncoderAdapter


class TestAudioOnlyInference(unittest.TestCase):
    def test_energy_vad_segmentation(self):
        """Verify EnergyVADSegmenter splits audio into bounded duration segments."""
        segmenter = EnergyVADSegmenter(
            sample_rate=16000,
            max_segment_duration_sec=5.0,
            min_speech_duration_sec=0.5,
        )

        # 8 seconds of synthetic audio (silence -> speech -> silence -> speech)
        sr = 16000
        wav = torch.zeros(1, sr * 8)
        # Speech block 1: sec 1 to 3
        wav[0, sr * 1 : sr * 3] = torch.randn(sr * 2) * 0.1
        # Speech block 2: sec 5 to 7
        wav[0, sr * 5 : sr * 7] = torch.randn(sr * 2) * 0.1

        segments = segmenter.segment_waveform(wav)
        self.assertGreater(len(segments), 1)

        for seg in segments:
            self.assertLessEqual(seg.duration_sec, 5.0 + 1e-4)
            self.assertEqual(seg.waveform.size(0), 1)

    def test_audio_only_transcription_contract(self):
        """Verify ASRInferencePipeline operates strictly audio-only with zero teacher dependency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tok_prefix = Path(tmpdir) / "test_tok"
            sentences = ["hà nội mùa thu", "thành phố hồ chí minh", "tiếng việt thân yêu"]
            model_path = train_bpe_tokenizer(sentences, tok_prefix, vocab_size=32)
            tokenizer = Tokenizer(model_path)

            adapter = ChunkFormerEncoderAdapter(
                input_size=80,
                output_size=128,
                attention_heads=4,
                linear_units=256,
                num_blocks=2,
            )
            model = ViPViCTCASR(adapter, vocab_size=tokenizer.vocab_size, blank_id=0)

            pipeline = ASRInferencePipeline(model=model, tokenizer=tokenizer)
            self.assertFalse(pipeline.frontend.training)

            # 1 second of random audio
            wav = torch.randn(1, 16000) * 0.05
            text_1 = pipeline.transcribe_waveform(wav)
            text_2 = pipeline.transcribe_waveform(wav)

            # Strict determinism: Same waveform produces exact same transcript
            self.assertEqual(text_1, text_2)
            self.assertIsInstance(text_1, str)


if __name__ == "__main__":
    unittest.main()
