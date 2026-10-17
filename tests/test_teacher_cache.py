"""Unit tests for PhoBERT teacher extraction, sliding window stitching, and sharded cache (Gate G1)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

from vi_asr.data.manifest import ManifestItem
from vi_asr.data.teacher_cache import (
    PhoBERTTeacherExtractor,
    TeacherCacheReader,
    build_offline_teacher_cache,
)


class TestTeacherCacheStitchingAndReader(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_sliding_window_stitching_mocked_model(self):
        """Verify that utterances exceeding 254 tokens are properly stitched without silent truncation."""
        extractor = MagicMock(spec=PhoBERTTeacherExtractor)
        extractor.model_id = "vinai/phobert-base"
        extractor.revision = "test_revision"
        extractor.max_content_window = 254
        extractor.bos_token_id = 0
        extractor.eos_token_id = 2
        extractor.pad_token_id = 1

        # Simulate 300 content tokens (> 254 limit)
        total_tokens = 300
        simulated_token_ids = list(range(100, 100 + total_tokens))

        extractor.tokenizer = MagicMock()
        extractor.tokenizer.encode.return_value = simulated_token_ids

        # Mock forward pass returning [1, window_len, 768]
        def mock_forward(input_ids):
            batch, seq_len = input_ids.shape
            hidden = torch.full((batch, seq_len, 768), fill_value=1.0)
            mock_out = MagicMock()
            mock_out.last_hidden_state = hidden
            return mock_out

        extractor.model = MagicMock()
        extractor.model.side_effect = mock_forward
        extractor.device = torch.device("cpu")

        # Run real extract_from_text logic with mocked internals
        states, token_ids = PhoBERTTeacherExtractor.extract_from_text(extractor, "test dummy text")

        self.assertEqual(len(token_ids), 300)
        self.assertEqual(states.shape, (300, 768))
        self.assertEqual(states.dtype, torch.float16)

    def test_sharded_cache_write_and_read(self):
        """Verify that build_offline_teacher_cache and TeacherCacheReader correctly write and read memmapped arrays."""
        items = [
            ManifestItem(
                sample_id="utt_1",
                split="train",
                audio_relpath="audio/1.wav",
                source_revision="rev1",
                sample_rate=16000,
                num_samples=16000,
                duration=1.0,
                text_raw="Hà Nội mùa thu",
                text_asr="hà nội mùa thu",
            ),
            ManifestItem(
                sample_id="utt_2",
                split="train",
                audio_relpath="audio/2.wav",
                source_revision="rev1",
                sample_rate=16000,
                num_samples=32000,
                duration=2.0,
                text_raw="Thành phố Hồ Chí Minh",
                text_asr="thành phố hồ chí minh",
            ),
        ]

        # Mock extractor that returns deterministic shapes
        mock_extractor = MagicMock(spec=PhoBERTTeacherExtractor)
        mock_extractor.model_id = "vinai/phobert-base"
        mock_extractor.revision = "test_rev"
        mock_extractor.max_content_window = 254

        def mock_extract(text):
            if "Hà Nội" in text:
                return torch.ones((4, 768), dtype=torch.float16) * 1.5, [10, 11, 12, 13]
            else:
                return torch.ones((5, 768), dtype=torch.float16) * 2.5, [20, 21, 22, 23, 24]

        mock_extractor.extract_from_text.side_effect = mock_extract

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "teacher_cache"
            build_offline_teacher_cache(items, cache_dir, mock_extractor)

            # Check files created
            self.assertTrue((cache_dir / "meta.json").exists())
            self.assertTrue((cache_dir / "index.jsonl").exists())
            self.assertTrue((cache_dir / "tokens-000.npy").exists())
            self.assertTrue((cache_dir / "token_ids-000.npy").exists())
            self.assertTrue((cache_dir / "offsets-000.npy").exists())

            # Read with TeacherCacheReader
            with TeacherCacheReader(cache_dir) as reader:
                self.assertEqual(len(reader), 2)
                self.assertIn("utt_1", reader)
                self.assertIn("utt_2", reader)

                # Retrieve utt_1
                t1, ids1 = reader.get("utt_1")
                self.assertEqual(t1.shape, (4, 768))
                self.assertEqual(t1.dtype, torch.float32)  # reader converts float16 to float32
                self.assertEqual(ids1, [10, 11, 12, 13])
                self.assertAlmostEqual(t1[0, 0].item(), 1.5, places=2)

                # Retrieve utt_2
                t2, ids2 = reader.get("utt_2")
                self.assertEqual(t2.shape, (5, 768))
                self.assertEqual(ids2, [20, 21, 22, 23, 24])
                self.assertAlmostEqual(t2[0, 0].item(), 2.5, places=2)

                # Test missing sample ID raises KeyError (NEVER silent zeroes)
                with self.assertRaises(KeyError):
                    reader.get("non_existent_utt_999")


if __name__ == "__main__":
    unittest.main()
