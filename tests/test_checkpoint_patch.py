"""Unit tests verifying activation checkpointing patch equivalence and gradient correctness."""

import copy
import sys
import unittest
from pathlib import Path

import torch

# Ensure third_party/chunkformer and src are importable
ROOT_PATH = Path(__file__).resolve().parent.parent
CHUNKFORMER_PATH = ROOT_PATH / "third_party" / "chunkformer"
if str(CHUNKFORMER_PATH) not in sys.path:
    sys.path.insert(0, str(CHUNKFORMER_PATH))

from chunkformer.modules.encoder import ChunkFormerEncoder


class TestActivationCheckpointingPatch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # Construct compact 2-block encoder for fast and deterministic unit testing
        self.encoder_params = dict(
            input_size=80,
            output_size=64,
            attention_heads=2,
            linear_units=128,
            num_blocks=2,
            dropout_rate=0.0,
            positional_dropout_rate=0.0,
            attention_dropout_rate=0.0,
            encoder_layerdrop=0.0,
            cnn_module_kernel=15,
            gradient_checkpointing=False,
        )

        self.model_no_cp = ChunkFormerEncoder(**self.encoder_params)
        # Deepcopy weights so parameters are bitwise identical
        self.model_with_cp = copy.deepcopy(self.model_no_cp)
        self.model_with_cp.gradient_checkpointing = True

    def test_forward_output_equivalence_eval_mode(self):
        """In eval mode, forward outputs must be identical."""
        self.model_no_cp.eval()
        self.model_with_cp.eval()

        torch.manual_seed(123)
        xs = torch.randn(2, 64, 80)
        xs_lens = torch.tensor([64, 48], dtype=torch.int32)

        with torch.no_grad():
            out_no_cp, masks_no_cp = self.model_no_cp(xs, xs_lens)
            out_cp, masks_cp = self.model_with_cp(xs, xs_lens)

        self.assertEqual(masks_no_cp.tolist(), masks_cp.tolist())
        max_diff = torch.max(torch.abs(out_no_cp - out_cp)).item()
        self.assertLess(max_diff, 1e-6, f"Forward diff {max_diff} exceeded tolerance 1e-6")

    def test_forward_and_backward_gradient_equivalence_train_mode(self):
        """In train mode with dropout=0, backward gradients must match within floating point precision."""
        self.model_no_cp.train()
        self.model_with_cp.train()

        torch.manual_seed(999)
        xs1 = torch.randn(2, 64, 80, requires_grad=True)
        xs2 = xs1.clone().detach().requires_grad_(True)
        xs_lens = torch.tensor([64, 50], dtype=torch.int32)

        # Forward pass
        out_no_cp, _ = self.model_no_cp(xs1, xs_lens)
        out_cp, _ = self.model_with_cp(xs2, xs_lens)

        # Assert forward output match in training mode
        max_fwd_diff = torch.max(torch.abs(out_no_cp - out_cp)).item()
        self.assertLess(max_fwd_diff, 1e-5, f"Forward train diff {max_fwd_diff} exceeded tolerance")

        # Backward pass
        loss_no_cp = out_no_cp.sum()
        loss_cp = out_cp.sum()

        loss_no_cp.backward()
        loss_cp.backward()

        # Check input gradients
        max_input_grad_diff = torch.max(torch.abs(xs1.grad - xs2.grad)).item()
        self.assertLess(max_input_grad_diff, 1e-5, f"Input grad diff {max_input_grad_diff} exceeded tolerance")

        # Check parameter gradients
        for (name1, p1), (name2, p2) in zip(
            self.model_no_cp.named_parameters(),
            self.model_with_cp.named_parameters(),
        ):
            self.assertEqual(name1, name2)
            if p1.grad is not None and p2.grad is not None:
                max_p_diff = torch.max(torch.abs(p1.grad - p2.grad)).item()
                self.assertLess(
                    max_p_diff,
                    1e-4,
                    f"Param {name1} grad diff {max_p_diff} exceeded tolerance",
                )

    def test_checkpointing_flag_behavior(self):
        """Verify gradient_checkpointing attribute is set properly."""
        self.assertFalse(self.model_no_cp.gradient_checkpointing)
        self.assertTrue(self.model_with_cp.gradient_checkpointing)


if __name__ == "__main__":
    unittest.main()
