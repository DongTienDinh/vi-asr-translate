"""Unit test for mathematical gradient equivalence of gradient accumulation (Gate G5)."""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vi_asr.models.ctc_asr import CTCHead


class TestDDPGradientEquivalence(unittest.TestCase):
    def test_gradient_accumulation_equivalence(self):
        """Verify that microbatch accumulation with per-sample normalization matches full batch gradient within 1e-5."""
        torch.manual_seed(42)

        hidden_dim = 64
        vocab_size = 20
        total_samples = 4
        T = 15

        # Two identical models
        model_full = CTCHead(hidden_size=hidden_dim, vocab_size=vocab_size, dropout=0.0)
        model_accum = CTCHead(hidden_size=hidden_dim, vocab_size=vocab_size, dropout=0.0)
        model_accum.load_state_dict(model_full.state_dict())

        # Inputs
        H = torch.randn(total_samples, T, hidden_dim)
        targets = torch.tensor([[1, 2, 3], [4, 5, 0], [6, 7, 8], [9, 0, 0]], dtype=torch.int64)
        target_lens = torch.tensor([3, 2, 3, 1], dtype=torch.int64)
        input_lens = torch.full((total_samples,), T, dtype=torch.int64)

        # 1. Full batch gradient
        model_full.zero_grad()
        logits_full = model_full(H)
        log_probs_full = logits_full.float().log_softmax(dim=-1).transpose(0, 1)
        raw_losses_full = F.ctc_loss(
            log_probs_full, targets, input_lens, target_lens, blank=0, reduction="none"
        )
        loss_full = (raw_losses_full / target_lens.float()).mean()
        loss_full.backward()

        grad_full_weight = model_full.linear.weight.grad.clone()
        grad_full_bias = model_full.linear.bias.grad.clone()

        # 2. Gradient accumulation: 2 microbatches of 2 samples each
        model_accum.zero_grad()
        micro_size = 2
        num_micro = total_samples // micro_size  # 2

        for m in range(num_micro):
            start = m * micro_size
            end = start + micro_size
            H_m = H[start:end]
            t_m = targets[start:end]
            tl_m = target_lens[start:end]
            il_m = input_lens[start:end]

            logits_m = model_accum(H_m)
            log_probs_m = logits_m.float().log_softmax(dim=-1).transpose(0, 1)
            raw_loss_m = F.ctc_loss(
                log_probs_m, t_m, il_m, tl_m, blank=0, reduction="none"
            )
            # Per-example normalized, scaled by 1 / num_micro
            loss_m = (raw_loss_m / tl_m.float()).sum() / total_samples
            loss_m.backward()

        grad_accum_weight = model_accum.linear.weight.grad
        grad_accum_bias = model_accum.linear.bias.grad

        # Check equivalence
        diff_weight = (grad_full_weight - grad_accum_weight).abs().max().item()
        diff_bias = (grad_full_bias - grad_accum_bias).abs().max().item()

        self.assertLess(diff_weight, 1e-5, f"Weight gradient diff {diff_weight} exceeds tolerance")
        self.assertLess(diff_bias, 1e-5, f"Bias gradient diff {diff_bias} exceeds tolerance")


if __name__ == "__main__":
    unittest.main()
