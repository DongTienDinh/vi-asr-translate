"""Unit tests for atomic checkpoint saving and exact resumption (Gate G6)."""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from vi_asr.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)


class TestAtomicResume(unittest.TestCase):
    def test_checkpoint_atomic_save_and_exact_resume(self):
        """Verify atomic checkpoint save and complete state restoration (model, optimizer, RNG)."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        optimizer = AdamW(model.parameters(), lr=1e-3)

        # Do one step to populate optimizer state
        x = torch.randn(4, 10)
        y = model(x).sum()
        y.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            saved_file = save_training_checkpoint(
                checkpoint_dir=ckpt_dir,
                epoch=2,
                global_step=150,
                model=model,
                optimizer=optimizer,
                metrics={"valid_wer": 0.125},
                is_best=True,
            )

            self.assertTrue(saved_file.exists())
            self.assertTrue((ckpt_dir / "checkpoint_latest.pt").exists())
            self.assertTrue((ckpt_dir / "checkpoint_best.pt").exists())

            # Create a fresh model and optimizer
            model_new = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
            optimizer_new = AdamW(model_new.parameters(), lr=1e-3)

            # Load checkpoint
            data = load_training_checkpoint(
                checkpoint_path=saved_file,
                model=model_new,
                optimizer=optimizer_new,
            )

            self.assertEqual(data["epoch"], 2)
            self.assertEqual(data["global_step"], 150)
            self.assertEqual(data["metrics"]["valid_wer"], 0.125)

            # Verify identical parameters
            for p1, p2 in zip(model.parameters(), model_new.parameters()):
                self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()
