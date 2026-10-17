"""Unit tests for configuration loader, hierarchy merging, and CLI router (Gate G0)."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Add src to sys.path so tests can run without pip install -e .
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vi_asr.config import (
    ConfigValidationError,
    compute_config_hash,
    dump_resolved_config,
    resolve_config,
)


class TestConfigHierarchy(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent.parent
        self.base_path = self.root / "configs" / "base.yaml"
        self.p2_path = self.root / "configs" / "variants" / "p2.yaml"
        self.t4x2_path = self.root / "configs" / "profiles" / "t4x2.yaml"
        self.kaggle_path = self.root / "configs" / "paths" / "kaggle.yaml"
        self.local_path = self.root / "configs" / "paths" / "local.yaml"

    def test_base_config_validity(self):
        cfg = resolve_config(self.base_path, paths_path=self.local_path)
        self.assertEqual(cfg["frontend"]["sample_rate"], 16000)
        self.assertEqual(cfg["frontend"]["num_mel_bins"], 80)
        self.assertEqual(cfg["frontend"]["fbank_waveform_multiplier"], 32768.0)
        self.assertFalse(cfg["alignment"]["enabled"])
        self.assertEqual(cfg["loss"]["lambda_token"], 0.0)

    def test_variant_merging(self):
        cfg = resolve_config(
            self.base_path,
            variant_path=self.p2_path,
            paths_path=self.local_path,
        )
        self.assertTrue(cfg["alignment"]["enabled"])
        self.assertTrue(cfg["alignment"]["temporal"]["enabled"])
        self.assertEqual(cfg["loss"]["lambda_token"], 0.1)
        self.assertEqual(cfg["experiment"]["name"], "p2_token_temporal")

    def test_profile_merging(self):
        cfg = resolve_config(
            self.base_path,
            variant_path=self.p2_path,
            profile_path=self.t4x2_path,
            paths_path=self.local_path,
        )
        self.assertEqual(cfg["runtime"]["expected_world_size"], 2)
        self.assertEqual(cfg["batching"]["accumulation_steps"], 8)
        self.assertEqual(cfg["batching"]["max_padded_fbank_frames_per_rank"], 6400)

    def test_paths_merging(self):
        cfg = resolve_config(
            self.base_path,
            paths_path=self.kaggle_path,
        )
        self.assertEqual(cfg["paths"]["data_root"], "/kaggle/input/vimd-dataset")

    def test_cli_overrides(self):
        cfg = resolve_config(
            self.base_path,
            paths_path=self.local_path,
            overrides=["training.epochs=50", "alignment.temperature=0.2"],
        )
        self.assertEqual(cfg["training"]["epochs"], 50)
        self.assertEqual(cfg["alignment"]["temperature"], 0.2)

    def test_validation_error_on_invalid_sample_rate(self):
        with self.assertRaises(ConfigValidationError):
            resolve_config(
                self.base_path,
                paths_path=self.local_path,
                overrides=["frontend.sample_rate=8000"],
            )

    def test_config_hash_determinism(self):
        cfg1 = resolve_config(self.base_path, paths_path=self.local_path)
        cfg2 = resolve_config(self.base_path, paths_path=self.local_path)
        hash1 = compute_config_hash(cfg1)
        hash2 = compute_config_hash(cfg2)
        self.assertEqual(hash1, hash2)

    def test_dump_resolved_config(self):
        cfg = resolve_config(self.base_path, variant_path=self.p2_path, paths_path=self.local_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "resolved.yaml"
            cfg_hash = dump_resolved_config(cfg, out_file)
            self.assertTrue(out_file.exists())
            self.assertTrue(len(cfg_hash) == 64)


class TestCliInvocation(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent.parent

    def test_cli_help(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.root / "src")
        res = subprocess.run(
            [sys.executable, "-m", "vi_asr", "--help"],
            cwd=str(self.root),
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("audit", res.stdout)
        self.assertIn("train", res.stdout)
        self.assertIn("evaluate-asr", res.stdout)

    def test_cli_audit(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.root / "src")
        with tempfile.TemporaryDirectory() as tmpdir:
            res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "vi_asr",
                    "audit",
                    "--config",
                    "configs/base.yaml",
                    "--paths",
                    "configs/paths/local.yaml",
                    "--report-dir",
                    tmpdir,
                ],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(res.returncode, 0)
            self.assertTrue((Path(tmpdir) / "resolved_config.yaml").exists())


if __name__ == "__main__":
    unittest.main()
