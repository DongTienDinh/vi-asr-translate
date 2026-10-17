"""Loss functions for Vietnamese CTC ASR and Multi-task Alignment Learning."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenCosineLoss(nn.Module):
    """Cosine distance loss between predicted audio context and frozen teacher representations.

    Invariant (Section 13.1):
    l_token_i = mean_content_tokens(1 - cosine(E_hat_i,n, E_i,n))
    Computed as mean over actual content tokens in each utterance, then averaged across batch.
    Padded tokens are strictly masked out and have zero contribution.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        predicted_teacher: torch.Tensor,
        target_teacher: torch.Tensor,
        token_lengths: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:

        predicted_teacher: Tensor [B, N, 768] (E_hat)
        target_teacher: Tensor [B, N, 768] (E, frozen)
        token_lengths: Tensor [B] actual content token count N_i per sample
        valid_token_mask: Optional bool Tensor [B, N] (True=content, False=pad)

        Returns:
            mean_loss: Scalar float32 loss averaged across batch
            per_example_loss: Tensor [B] float32 loss per sample
        """
        B, N, _ = predicted_teacher.shape
        device = predicted_teacher.device

        if valid_token_mask is None:
            n_range = torch.arange(N, device=device).unsqueeze(0)  # [1, N]
            valid_token_mask = n_range < token_lengths.unsqueeze(1)  # [B, N]

        # Cosine similarity in FP32 for numerical stability.
        # Invariant (Gate G3): Teacher targets must never receive gradients.
        pred_f = predicted_teacher.float()
        target_f = target_teacher.detach().float()

        # Cosine similarity per token: [B, N]
        cos_sim = F.cosine_similarity(pred_f, target_f, dim=-1, eps=self.eps)
        per_token_loss = 1.0 - cos_sim  # [B, N] in [0, 2]

        # Strictly mask out padded tokens
        mask_f = valid_token_mask.float()
        per_token_loss_masked = per_token_loss * mask_f  # [B, N]

        # Per-example normalization: mean over true content tokens N_i
        token_sums = per_token_loss_masked.sum(dim=-1)  # [B]
        valid_counts = token_lengths.float().clamp_min(1.0)  # [B]
        per_example_loss = token_sums / valid_counts  # [B]

        # For samples with 0 content tokens (e.g. silence), loss is strictly 0.0
        zero_token_mask = (token_lengths == 0)
        if zero_token_mask.any():
            per_example_loss = per_example_loss.masked_fill(zero_token_mask, 0.0)

        mean_loss = per_example_loss.mean()
        return mean_loss, per_example_loss


class MultiTaskASRLoss(nn.Module):
    """Combined Multi-Task Loss for Vietnamese ASR:

    l_i = l_ctc_i + lambda_token * l_token_i + lambda_region * l_region_i
    loss = mean_i(l_i)

    Guarantees:
    - Pure FP32 computations for CTCLoss and CosineLoss.
    - CTC loss is normalized per example by target length U_i.
    - Token loss is normalized per example by content token count N_i.
    - Zero-token and zero-target edge cases handled cleanly.
    """

    def __init__(
        self,
        lambda_token: float = 0.0,
        lambda_region: float = 0.0,
        ctc_blank_id: int = 0,
        ctc_zero_infinity: bool = False,
    ):
        super().__init__()
        self.lambda_token = float(lambda_token)
        self.lambda_region = float(lambda_region)
        self.ctc_blank_id = int(ctc_blank_id)
        self.ctc_zero_infinity = bool(ctc_zero_infinity)

        self.token_loss_fn = TokenCosineLoss()
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        ctc_logits: torch.Tensor,
        ctc_output_lengths: torch.Tensor,
        ctc_targets: torch.Tensor,
        ctc_target_lengths: torch.Tensor,
        predicted_teacher: Optional[torch.Tensor] = None,
        target_teacher: Optional[torch.Tensor] = None,
        token_lengths: Optional[torch.Tensor] = None,
        region_logits: Optional[torch.Tensor] = None,
        region_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss components and total per-example normalized loss.

        Returns:
            Dict containing:
                "loss": Scalar total loss for backward
                "per_example_loss": Tensor [B] per-sample total loss
                "ctc_loss": Scalar mean CTC loss
                "token_loss": Scalar mean token loss (0.0 if disabled)
                "region_loss": Scalar mean region loss (0.0 if disabled)
        """
        B = ctc_logits.size(0)
        device = ctc_logits.device

        # 1. CTC Loss in FP32
        log_probs = ctc_logits.float().log_softmax(dim=-1).transpose(0, 1)  # [T, B, V]
        raw_ctc_losses = F.ctc_loss(
            log_probs=log_probs,
            targets=ctc_targets,
            input_lengths=ctc_output_lengths,
            target_lengths=ctc_target_lengths,
            blank=self.ctc_blank_id,
            reduction="none",
            zero_infinity=self.ctc_zero_infinity,
        )  # [B]

        # Target length normalization per utterance: l_ctc_i = raw_ctc_i / U_i
        target_lens_clamped = ctc_target_lengths.float().clamp_min(1.0)
        per_example_ctc = raw_ctc_losses / target_lens_clamped  # [B]
        ctc_loss_mean = per_example_ctc.mean()

        total_per_example = per_example_ctc.clone()

        # 2. Token Alignment Cosine Loss
        if self.lambda_token > 0.0 and predicted_teacher is not None and target_teacher is not None and token_lengths is not None:
            token_loss_mean, per_example_token = self.token_loss_fn(
                predicted_teacher=predicted_teacher,
                target_teacher=target_teacher,
                token_lengths=token_lengths,
            )
            total_per_example = total_per_example + self.lambda_token * per_example_token
        else:
            token_loss_mean = torch.tensor(0.0, device=device, dtype=torch.float32)

        # 3. Optional Region Classification Loss
        if self.lambda_region > 0.0 and region_logits is not None and region_targets is not None:
            per_example_region = self.ce_loss_fn(region_logits.float(), region_targets)  # [B]
            region_loss_mean = per_example_region.mean()
            total_per_example = total_per_example + self.lambda_region * per_example_region
        else:
            region_loss_mean = torch.tensor(0.0, device=device, dtype=torch.float32)

        total_loss = total_per_example.mean()

        return {
            "loss": total_loss,
            "per_example_loss": total_per_example,
            "ctc_loss": ctc_loss_mean,
            "token_loss": token_loss_mean,
            "region_loss": region_loss_mean,
        }
