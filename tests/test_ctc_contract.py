"""Unit tests for text normalization, manifest serialization, tokenizer, CTC collapse, and CTC contract (Gate G1)."""

import tempfile
import unittest
from pathlib import Path

from vi_asr.data.audit import audit_splits
from vi_asr.data.manifest import ManifestItem, load_manifest, save_manifest
from vi_asr.data.normalize import normalize_text_asr, normalize_unicode_nfc
from vi_asr.data.tokenizer import Tokenizer, train_bpe_tokenizer


class TestNormalization(unittest.TestCase):
    def test_unicode_nfc_preservation(self):
        # Decomposed 'e' + combining acute accent vs precomposed 'é'
        decomposed = "e\u0301"
        precomposed = "\u00e9"
        self.assertNotEqual(decomposed, precomposed)
        self.assertEqual(normalize_unicode_nfc(decomposed), normalize_unicode_nfc(precomposed))

    def test_normalize_text_asr_vietnamese_diacritics(self):
        # Punctuation stripped, lowercase, diacritics kept intact
        raw = "Xin chào Việt Nam! Hôm nay thời tiết thế nào?"
        expected = "xin chào việt nam hôm nay thời tiết thế nào"
        self.assertEqual(normalize_text_asr(raw), expected)

    def test_normalize_text_asr_numbers_and_whitespace(self):
        raw = "   Đoạn đường   dài 120km   tại TP.HCM...   "
        expected = "đoạn đường dài 120km tại tp hcm"
        self.assertEqual(normalize_text_asr(raw), expected)


class TestManifestSerialization(unittest.TestCase):
    def test_manifest_roundtrip(self):
        item = ManifestItem(
            sample_id="utt_001",
            split="train",
            audio_relpath="audio/train/utt_001.wav",
            source_revision="test_rev_123",
            sample_rate=16000,
            num_samples=32000,
            duration=2.0,
            text_raw="Hà Nội mùa này vắng những cơn mưa.",
            text_asr="hà nội mùa này vắng những cơn mưa",
            speaker_id="spk_01",
            region_id=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_file = Path(tmpdir) / "test_manifest.jsonl"
            save_manifest([item], manifest_file)
            self.assertTrue(manifest_file.exists())

            loaded = load_manifest(manifest_file)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].sample_id, item.sample_id)
            self.assertEqual(loaded[0].text_asr, item.text_asr)
            self.assertEqual(loaded[0].region_id, 0)
            self.assertEqual(loaded[0].duration, 2.0)


class TestSentencePieceTokenizerAndCTC(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "xin chào các bạn học sinh",
            "hôm nay trời nắng đẹp tại hà nội",
            "việt nam đất nước tươi đẹp",
            "chúng tôi đang nghiên cứu mô hình nhận dạng tiếng nói",
            "âm thanh tiếng việt đa dạng ba miền bắc trung nam",
            "nghiên cứu khoa học có tính tái lập cao",
        ]

    def test_tokenizer_training_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_prefix = Path(tmpdir) / "sp_test"
            model_path = train_bpe_tokenizer(
                transcripts=self.corpus,
                model_prefix=model_prefix,
                vocab_size=120,
                character_coverage=1.0,
                pad_id=0,
                unk_id=1,
            )
            self.assertTrue(model_path.exists())

            tok = Tokenizer(model_path)
            self.assertEqual(tok.pad_id, 0)
            self.assertEqual(tok.blank_id, 0)
            self.assertEqual(tok.unk_id, 1)

            text = "việt nam đất nước tươi đẹp"
            ids = tok.encode(text)
            self.assertIsInstance(ids, list)
            self.assertTrue(len(ids) > 0)
            self.assertNotIn(0, ids)  # Blank/pad must not appear in valid target encoding

            decoded = tok.decode(ids)
            self.assertEqual(decoded, text)

    def test_ctc_collapse_logic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_prefix = Path(tmpdir) / "sp_test"
            model_path = train_bpe_tokenizer(
                transcripts=self.corpus,
                model_prefix=model_prefix,
                vocab_size=120,
            )
            tok = Tokenizer(model_path)

            ids = tok.encode("việt nam")
            # ids[0] is token 1, ids[1] is token 2
            t1, t2 = ids[0], ids[1]

            # 1. Consecutive duplicates without blank collapse into single: [t1, t1, t2] -> [t1, t2]
            collapsed = tok.ctc_collapse([0, t1, t1, 0, t2, t2, 0])
            self.assertEqual(collapsed, "việt nam")

            # 2. Blank-only returns empty string
            self.assertEqual(tok.ctc_collapse([0, 0, 0]), "")

    def test_ctc_validity_inequality(self):
        # target: [10, 10, 20] -> U=3, repeated consecutive pair (10, 10) -> R=1. Min required T_out = 4
        target_ids = [10, 10, 20]
        # output_frames = 3: mathematically impossible for CTC with repeated label
        self.assertFalse(Tokenizer.check_ctc_validity(target_ids, output_frames=3))
        # output_frames = 4: valid (can fit [10, blank, 10, 20])
        self.assertTrue(Tokenizer.check_ctc_validity(target_ids, output_frames=4))

        # target without consecutive duplicates: [10, 20, 30] -> U=3, R=0. Min required T_out = 3
        target_no_repeats = [10, 20, 30]
        self.assertFalse(Tokenizer.check_ctc_validity(target_no_repeats, output_frames=2))
        self.assertTrue(Tokenizer.check_ctc_validity(target_no_repeats, output_frames=3))


class TestAuditSuite(unittest.TestCase):
    def test_audit_split_overlap_and_duplicates(self):
        train_items = [
            ManifestItem(
                sample_id="tr_1",
                split="train",
                audio_relpath="a.wav",
                source_revision="r1",
                sample_rate=16000,
                num_samples=32000,
                duration=2.0,
                text_raw="Câu 1",
                text_asr="câu 1",
                speaker_id="spk_A",
                region_id=0,
                audio_hash="hash_alpha",
            ),
            ManifestItem(
                sample_id="tr_2",
                split="train",
                audio_relpath="b.wav",
                source_revision="r1",
                sample_rate=16000,
                num_samples=48000,
                duration=3.0,
                text_raw="Câu 2",
                text_asr="câu 2",
                speaker_id="spk_B",
                region_id=1,
                audio_hash="hash_beta",
            ),
        ]

        valid_items = [
            # Intentional overlap with spk_A and hash_alpha for leakage detection test
            ManifestItem(
                sample_id="va_1",
                split="valid",
                audio_relpath="dup.wav",
                source_revision="r1",
                sample_rate=16000,
                num_samples=32000,
                duration=2.0,
                text_raw="Câu 1",
                text_asr="câu 1",
                speaker_id="spk_A",
                region_id=0,
                audio_hash="hash_alpha",
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            audit = audit_splits(train_items, valid_items, report_dir=tmpdir)
            self.assertEqual(audit["audio_duplicates"], 1)
            self.assertEqual(audit["speaker_overlap"]["train_valid_count"], 1)
            self.assertEqual(audit["splits"]["train"]["total_samples"], 2)
            self.assertEqual(audit["splits"]["valid"]["total_samples"], 1)
            self.assertTrue((Path(tmpdir) / "audit.json").exists())
            self.assertTrue((Path(tmpdir) / "split_overlap.json").exists())


if __name__ == "__main__":
    unittest.main()
