"""Unit tests for Token-Frame Pooling, Soft Gaussian Temporal Prior, and Multi-task Loss (Gate G3)."""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vi_asr.models.alignment import TokenFramePoolingAlignment
from vi_asr.training.losses import MultiTaskASRLoss, TokenCosineLoss


class TestAlignmentMasksAndLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_attention_rows_sum_one_and_padding_zero(self):
        """Verify Gate G3 Invariant: Attention rows sum to 1.0 for valid tokens, and padding weights are 0.0."""
        alignment = TokenFramePoolingAlignment(
            audio_dim=64,
            teacher_dim=128,
            score_dim=32,
            temperature=0.1,
            temporal_prior_enabled=True,
            beta=0.1,
            sigma=0.2,
        )

        B = 2
        max_T = 12
        max_N = 6

        H = torch.randn(B, max_T, 64)
        E = torch.randn(B, max_N, 128)

        # Sample 0 has 10 frames and 4 tokens; Sample 1 has 7 frames and 3 tokens
        audio_lengths = torch.tensor([10, 7], dtype=torch.int64)
        token_lengths = torch.tensor([4, 3], dtype=torch.int64)

        out = alignment(H, E, audio_lengths, token_lengths)
        A = out["attention"]  # [B, N, T]

        self.assertEqual(A.shape, (B, max_N, max_T))

        for b in range(B):
            valid_T = int(audio_lengths[b].item())
            valid_N = int(token_lengths[b].item())

            # Valid content token rows must sum to 1.0
            for n in range(valid_N):
                row_sum = A[b, n, :valid_T].sum().item()
                self.assertAlmostEqual(row_sum, 1.0, places=5)
                # Padding frames must have 0.0 weight
                if valid_T < max_T:
                    pad_frame_sum = A[b, n, valid_T:].abs().sum().item()
                    self.assertEqual(pad_frame_sum, 0.0)

            # Padding token rows must be all 0.0
            for n in range(valid_N, max_N):
                self.assertEqual(A[b, n, :].abs().sum().item(), 0.0)

    def test_constant_values_invariant(self):
        """Verify Gate G3 Invariant: If all audio frames H_t are identical vector h0, pooled C_n = h0."""
        alignment = TokenFramePoolingAlignment(
            audio_dim=32,
            teacher_dim=64,
            score_dim=16,
            temperature=0.1,
            temporal_prior_enabled=False,
        )

        B = 2
        max_T = 8
        max_N = 4

        # Constant audio vectors
        h0 = torch.randn(32)
        H = h0.unsqueeze(0).unsqueeze(0).expand(B, max_T, 32).clone()
        E = torch.randn(B, max_N, 64)

        audio_lengths = torch.tensor([8, 5], dtype=torch.int64)
        token_lengths = torch.tensor([4, 2], dtype=torch.int64)

        out = alignment(H, E, audio_lengths, token_lengths)
        C = out["pooled_audio"]  # [B, N, 32]

        for b in range(B):
            valid_N = int(token_lengths[b].item())
            for n in range(valid_N):
                # Pooled vector must strictly equal h0
                diff = (C[b, n] - h0).abs().max().item()
                self.assertLess(diff, 1e-5)

    def test_padding_invariance(self):
        """Verify Gate G3 Invariant: Adding padding frames/tokens does not alter predictions or loss on valid items."""
        alignment = TokenFramePoolingAlignment(
            audio_dim=32,
            teacher_dim=64,
            score_dim=16,
            temperature=0.1,
            temporal_prior_enabled=True,
        )
        alignment.eval()

        # Unpadded single item
        T1 = 6
        N1 = 3
        H1 = torch.randn(1, T1, 32)
        E1 = torch.randn(1, N1, 64)
        out1 = alignment(H1, E1, torch.tensor([T1]), torch.tensor([N1]))
        E_hat1 = out1["predicted_teacher"]

        # Padded version with extra 5 frames and 4 tokens
        T_pad = 11
        N_pad = 7
        H2 = torch.zeros(1, T_pad, 32)
        E2 = torch.zeros(1, N_pad, 64)
        H2[0, :T1] = H1[0]
        E2[0, :N1] = E1[0]
        # Put random garbage in the padding region
        H2[0, T1:] = torch.randn(T_pad - T1, 32)
        E2[0, N1:] = torch.randn(N_pad - N1, 64)

        out2 = alignment(H2, E2, torch.tensor([T1]), torch.tensor([N1]))
        E_hat2 = out2["predicted_teacher"]

        # Assert valid portion predictions match closely
        diff = (E_hat1[0, :N1] - E_hat2[0, :N1]).abs().max().item()
        self.assertLess(diff, 1e-5)

        # Assert loss on valid tokens matches closely
        loss_fn = TokenCosineLoss()
        loss1, _ = loss_fn(E_hat1, E1, torch.tensor([N1]))
        loss2, _ = loss_fn(E_hat2, E2, torch.tensor([N1]))
        self.assertAlmostEqual(loss1.item(), loss2.item(), places=5)

    def test_teacher_zero_grad_and_encoder_grad(self):
        """Verify Gate G3 Invariant: Frozen teacher E receives NO gradients, while encoder H and projections do."""
        alignment = TokenFramePoolingAlignment(
            audio_dim=32,
            teacher_dim=64,
            score_dim=16,
            temperature=0.1,
            temporal_prior_enabled=True,
        )

        H = torch.randn(2, 8, 32, requires_grad=True)
        E = torch.randn(2, 4, 64, requires_grad=True)

        audio_lengths = torch.tensor([8, 6], dtype=torch.int64)
        token_lengths = torch.tensor([4, 3], dtype=torch.int64)

        out = alignment(H, E, audio_lengths, token_lengths)
        loss_fn = TokenCosineLoss()
        loss, _ = loss_fn(out["predicted_teacher"], E, token_lengths)

        loss.backward()

        # Gate G3 assertion: Teacher MUST NOT receive gradients
        self.assertIsNone(E.grad)

        # Audio encoder H MUST receive gradients
        self.assertIsNotNone(H.grad)
        self.assertTrue(torch.any(H.grad != 0))

        # Projections MUST receive gradients
        self.assertIsNotNone(alignment.proj_k.weight.grad)
        self.assertIsNotNone(alignment.proj_q.weight.grad)
        self.assertIsNotNone(alignment.proj_v.weight.grad)

    def test_multi_task_loss_components(self):
        """Verify MultiTaskASRLoss computes CTC loss and token loss according to formula."""
        multi_loss = MultiTaskASRLoss(lambda_token=0.1, lambda_region=0.0, ctc_blank_id=0)

        B = 2
        T = 10
        V = 20
        logits = torch.randn(B, T, V, requires_grad=True)
        out_lens = torch.tensor([10, 8], dtype=torch.int64)
        targets = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.int64)
        target_lens = torch.tensor([3, 2], dtype=torch.int64)

        pred_teach = torch.randn(B, 4, 64)
        target_teach = torch.randn(B, 4, 64)
        token_lens = torch.tensor([4, 2], dtype=torch.int64)

        res = multi_loss(
            ctc_logits=logits,
            ctc_output_lengths=out_lens,
            ctc_targets=targets,
            ctc_target_lengths=target_lens,
            predicted_teacher=pred_teach,
            target_teacher=target_teach,
            token_lengths=token_lens,
        )

        total_loss = res["loss"]
        ctc_loss = res["ctc_loss"]
        token_loss = res["token_loss"]

        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(torch.isfinite(ctc_loss))
        self.assertTrue(torch.isfinite(token_loss))

        # Check formula: total_loss = ctc_loss + 0.1 * token_loss
        expected_total = (res["per_example_loss"]).mean()
        self.assertAlmostEqual(total_loss.item(), expected_total.item(), places=5)
        self.assertAlmostEqual(total_loss.item(), (ctc_loss + 0.1 * token_loss).item(), places=5)


if __name__ == "__main__":
    unittest.main()
