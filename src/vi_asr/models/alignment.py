"""Token-Frame Attention Pooling with Soft Gaussian Temporal Prior (Variants P1, P2).

Adaptation from MFGCN text-conditioned acoustic pooling:
- Audio representations H [B, T, 512]
- Frozen PhoBERT content-token representations E [B, N, 768] (cached offline, zero gradient)
- Cosine similarity attention matrix S [B, N, T] with temperature tau = 0.1
- Soft Gaussian temporal prior P(n, t) = exp( - (t/T_i - n/N_i)^2 / (2*sigma^2) )
- Attention weights A [B, N, T] with sum_t A_{n, t} = 1.0 for content tokens, 0 for padding
- Audio-only weighted context C = A @ H [B, N, 512]
- Predicted teacher representation E_hat = P_v(C) [B, N, 768]
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenFramePoolingAlignment(nn.Module):
    """Token-Frame Attention Pooling module with optional soft Gaussian temporal prior.

    Engineering Invariants (Gate G3):
    1. Zero Teacher Gradient: E is detached; teacher weights remain completely frozen.
    2. Attention Row Normalization: Rows corresponding to valid content tokens sum to 1.0.
    3. Padding Frame Exclusion: Frames beyond T_i have strictly 0.0 attention weight.
    4. Padding Token Exclusion: Tokens beyond N_i produce strictly 0.0 attention and 0 loss.
    5. Audio-Only Pooling: C is purely a linear combination of audio frames H (no text leak into audio).
    6. Constant-Values Invariant: If all frames H_t are identical vector h0, pooled C_n = h0.
    """

    def __init__(
        self,
        audio_dim: int = 512,
        teacher_dim: int = 768,
        score_dim: int = 256,
        temperature: float = 0.1,
        temporal_prior_enabled: bool = False,
        beta: float = 0.1,
        sigma: float = 0.2,
        use_temporal_prior: Optional[bool] = None,
        temporal_strength: Optional[float] = None,
        temporal_sigma: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.teacher_dim = teacher_dim
        self.score_dim = score_dim
        self.temperature = float(temperature)

        # Allow alias keywords from CLI or config
        if use_temporal_prior is not None:
            temporal_prior_enabled = use_temporal_prior
        if temporal_strength is not None:
            beta = temporal_strength
        if temporal_sigma is not None:
            sigma = temporal_sigma

        self.temporal_prior_enabled = bool(temporal_prior_enabled)
        self.beta = float(beta)
        self.sigma = float(sigma)

        # Projections
        # P_k: audio hidden -> score space
        self.proj_k = nn.Linear(audio_dim, score_dim)
        # P_q: teacher token hidden -> score space
        self.proj_q = nn.Linear(teacher_dim, score_dim)
        # P_v: pooled audio context -> teacher space
        self.proj_v = nn.Linear(audio_dim, teacher_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize projection layers with Xavier uniform and zero bias."""
        for layer in (self.proj_k, self.proj_q, self.proj_v):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def compute_gaussian_prior(
        self,
        audio_lengths: torch.Tensor,
        token_lengths: torch.Tensor,
        max_T: int,
        max_N: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Compute soft Gaussian temporal prior matrix per sample:

        prior[b, n, t] = - beta * ( (t + 0.5)/T_b - (n + 0.5)/N_b )^2 / (2 * sigma^2)
        Computed using true sequence lengths T_b and N_b (not padded lengths).
        """
        B = audio_lengths.size(0)
        prior = torch.zeros((B, max_N, max_T), device=device, dtype=dtype)

        # Grid of indices [max_N, 1] and [1, max_T]
        n_idx = torch.arange(max_N, device=device, dtype=dtype).unsqueeze(1) + 0.5  # [max_N, 1]
        t_idx = torch.arange(max_T, device=device, dtype=dtype).unsqueeze(0) + 0.5  # [1, max_T]

        two_sigma_sq = 2.0 * (self.sigma**2)

        for b in range(B):
            T_b = float(audio_lengths[b].item())
            N_b = float(token_lengths[b].item())
            if T_b <= 0.0 or N_b <= 0.0:
                continue

            valid_T = int(T_b)
            valid_N = int(N_b)

            # Normalized positions in [0, 1]
            mu_n = n_idx[:valid_N] / N_b  # [valid_N, 1]
            x_t = t_idx[:, :valid_T] / T_b  # [1, valid_T]

            # Gaussian distance penalty
            diff = x_t - mu_n  # [valid_N, valid_T]
            sample_prior = -self.beta * (diff**2) / two_sigma_sq
            prior[b, :valid_N, :valid_T] = sample_prior

        return prior

    def forward(
        self,
        H: torch.Tensor,
        E: torch.Tensor,
        audio_lengths: torch.Tensor,
        token_lengths: torch.Tensor,
        valid_frame_mask: Optional[torch.Tensor] = None,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for token-frame attention pooling.

        Args:
            H: Tensor [B, T, 512] audio encoder representations (requires grad)
            E: Tensor [B, N, 768] frozen PhoBERT token representations (detached)
            audio_lengths: Tensor [B] actual frame count T_i per sample
            token_lengths: Tensor [B] actual token count N_i per sample
            valid_frame_mask: Optional bool Tensor [B, T] (True=speech, False=pad)
            valid_token_mask: Optional bool Tensor [B, N] (True=content, False=pad)

        Returns:
            Dict with:
                "predicted_teacher": Tensor [B, N, 768] predicted teacher representations E_hat
                "attention": Tensor [B, N, T] attention weights A
                "pooled_audio": Tensor [B, N, 512] pooled audio context C
                "cosine_sim": Tensor [B, N, T] raw cosine similarity S
        """
        B, T, _ = H.shape
        _, N, _ = E.shape
        device = H.device

        # Invariant 1: Ensure teacher representations strictly have NO gradients
        E_frozen = E.detach()

        # Construct masks if not provided
        if valid_frame_mask is None:
            t_range = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
            valid_frame_mask = t_range < audio_lengths.unsqueeze(1)  # [B, T]

        if valid_token_mask is None:
            n_range = torch.arange(N, device=device).unsqueeze(0)  # [1, N]
            valid_token_mask = n_range < token_lengths.unsqueeze(1)  # [B, N]

        # 1. Project and L2 normalize keys (audio) and queries (tokens)
        K = F.normalize(self.proj_k(H), p=2, dim=-1, eps=1e-8)  # [B, T, score_dim]
        Q = F.normalize(self.proj_q(E_frozen), p=2, dim=-1, eps=1e-8)  # [B, N, score_dim]

        # 2. Compute Cosine similarity S = Q @ K^T
        S = torch.bmm(Q, K.transpose(1, 2))  # [B, N, T]

        # 3. Scale by temperature
        logits = S / max(self.temperature, 1e-4)

        # 4. Optional soft Gaussian temporal prior
        if self.temporal_prior_enabled:
            prior = self.compute_gaussian_prior(
                audio_lengths=audio_lengths,
                token_lengths=token_lengths,
                max_T=T,
                max_N=N,
                device=device,
                dtype=logits.dtype,
            )
            logits = logits + prior

        # 5. Mask out invalid audio frames before softmax (-1e4 for stability in float32/fp16)
        frame_mask = valid_frame_mask.unsqueeze(1)  # [B, 1, T]
        logits_masked = logits.masked_fill(~frame_mask, -10000.0)

        # 6. Attention Softmax along time dimension (computed in float32 for numerical stability)
        A = F.softmax(logits_masked.float(), dim=-1).to(H.dtype)  # [B, N, T]

        # Invariant 3 & 4: Zero out padding frames and padding token rows
        token_mask = valid_token_mask.unsqueeze(-1).to(A.dtype)  # [B, N, 1]
        frame_mask_f = frame_mask.to(A.dtype)  # [B, 1, T]
        A = A * frame_mask_f * token_mask

        # Re-normalize valid token rows to strictly sum to 1.0 (guard against fp16 precision drift)
        row_sums = A.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # [B, N, 1]
        A = torch.where(token_mask.bool(), A / row_sums, torch.zeros_like(A))

        # 7. Audio-only weighted context pooling: C = A @ H [B, N, audio_dim]
        C = torch.bmm(A, H)

        # 8. Project pooled audio context to teacher representation space: E_hat = P_v(C)
        E_hat = self.proj_v(C)

        return {
            "predicted_teacher": E_hat,
            "attention": A,
            "pooled_audio": C,
            "cosine_sim": S,
        }
