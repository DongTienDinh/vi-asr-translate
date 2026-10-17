"""Unit tests for audio frontend, Kaldi numerical parity, CMVN, and SpecAugment (Gate G2)."""

import sys
import unittest
from pathlib import Path

import torch
import torchaudio.compliance.kaldi as kaldi

# Ensure src is in sys.path
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vi_asr.data.augment import SpecAugment
from vi_asr.data.frontend import GlobalCMVN, KaldiFrontend


class TestFrontendKaldiParity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.sample_rate = 16000
        # 1.5 seconds of synthetic audio (sine wave + small noise in float range [-1.0, 1.0])
        t = torch.linspace(0, 1.5, int(1.5 * self.sample_rate))
        self.waveform = 0.5 * torch.sin(2 * 3.14159 * 440 * t) + 0.05 * torch.randn_like(t)
        self.waveform = torch.clamp(self.waveform, -1.0, 1.0)

    def test_kaldi_parity_deterministic_eval(self):
        """Verify absolute numerical difference vs upstream Kaldi fbank is < 1e-5."""
        frontend = KaldiFrontend(
            sample_rate=self.sample_rate,
            num_mel_bins=80,
            frame_length_ms=25.0,
            frame_shift_ms=10.0,
            window_type="povey",
            energy_floor=0.0,
            eval_dither=0.0,
            fbank_waveform_multiplier=32768.0,
        )
        frontend.eval()

        # 1. Output from our KaldiFrontend
        feats, lengths = frontend(self.waveform.unsqueeze(0))
        our_feat = feats[0, : lengths[0]]

        # 2. Direct upstream Kaldi calculation with explicit 32768.0 scale
        scaled_wav = self.waveform.unsqueeze(0) * 32768.0
        ref_feat = kaldi.fbank(
            scaled_wav,
            num_mel_bins=80,
            frame_length=25.0,
            frame_shift=10.0,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=16000,
            window_type="povey",
        )

        self.assertEqual(our_feat.shape, ref_feat.shape)
        max_diff = torch.max(torch.abs(our_feat - ref_feat)).item()
        self.assertLess(
            max_diff,
            1e-5,
            f"Kaldi parity difference {max_diff} exceeded tolerance threshold 1e-5",
        )

    def test_waveform_multiplier_importance(self):
        """Verify that omitting the 32768.0 multiplier causes substantial scale discrepancy."""
        scaled_wav = self.waveform.unsqueeze(0) * 32768.0
        unscaled_wav = self.waveform.unsqueeze(0)

        feat_scaled = kaldi.fbank(scaled_wav, num_mel_bins=80, dither=0.0)
        feat_unscaled = kaldi.fbank(unscaled_wav, num_mel_bins=80, dither=0.0)

        # Log energy difference between 32768x and 1x should be approximately 2 * ln(32768) ~= 20.79
        diff_mean = torch.mean(feat_scaled - feat_unscaled).item()
        self.assertGreater(diff_mean, 10.0, "Multiplier 32768.0 had insufficient effect on fbank energy")

    def test_deterministic_eval_vs_dither_train(self):
        """Verify that in eval mode dither=0 is strictly deterministic."""
        frontend = KaldiFrontend(train_dither=1.0, eval_dither=0.0)
        frontend.eval()

        feat1, _ = frontend(self.waveform)
        feat2, _ = frontend(self.waveform)
        self.assertTrue(torch.equal(feat1, feat2), "Eval mode fbank must be bitwise identical")

        frontend.train()
        feat_train1, _ = frontend(self.waveform)
        feat_train2, _ = frontend(self.waveform)
        self.assertFalse(torch.equal(feat_train1, feat_train2), "Train mode with dither=1.0 should vary across runs")

    def test_global_cmvn_contract(self):
        """Verify GlobalCMVN correctly computes (x - mean) * istd."""
        mean = torch.randn(80)
        istd = torch.abs(torch.randn(80)) + 0.1
        cmvn = GlobalCMVN(mean, istd, norm_var=True)

        x = torch.randn(2, 50, 80)
        out = cmvn(x)
        expected = (x - mean) * istd

        max_diff = torch.max(torch.abs(out - expected)).item()
        self.assertLess(max_diff, 1e-6)

    def test_specaugment_behavior(self):
        """Verify SpecAugment masks only during training and respects length boundaries."""
        specaug = SpecAugment(
            freq_masks=2,
            freq_max_bins=8,
            time_masks=2,
            time_max_frames=30,
            max_time_fraction_per_mask=0.1,
        )

        feats = torch.ones(2, 100, 80)
        lengths = torch.tensor([100, 60], dtype=torch.int64)

        # Eval mode: must be identity
        specaug.eval()
        eval_out, _ = specaug(feats, lengths)
        self.assertTrue(torch.equal(feats, eval_out))

        # Train mode: should mask portions to 0.0
        specaug.train()
        torch.manual_seed(100)
        train_out, out_lens = specaug(feats, lengths)
        self.assertTrue(torch.equal(lengths, out_lens))
        self.assertTrue(torch.any(train_out == 0.0), "SpecAugment should have applied zero masks")


if __name__ == "__main__":
    unittest.main()
