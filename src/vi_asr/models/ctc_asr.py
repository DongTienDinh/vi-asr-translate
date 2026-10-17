"""Pure Linear CTC ASR Model on ChunkFormer Encoder Backbone."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from vi_asr.models.encoder_adapter import ChunkFormerEncoderAdapter


class CTCHead(nn.Module):
    """Linear CTC classification head with optional dropout and zero-bias initialization."""

    def __init__(self, hidden_size: int = 512, vocab_size: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.linear = nn.Linear(hidden_size, vocab_size)

        # Initialize weights with Xavier uniform and bias to zero for stable CTC start
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

        x: Tensor [B, T, hidden_size]
        Returns:
            logits: Tensor [B, T, vocab_size]
        """
        x = self.dropout(x)
        return self.linear(x)


class ViPViCTCASR(nn.Module):
    """Vietnamese ASR Model combining ChunkFormer encoder with Pure Linear CTC head.

    Engineering Invariants:
    1. Pure CTC: zero upstream attention decoder parameters to avoid DDP overhead/OOM.
    2. FP32 CTC Loss: computation strictly in float32 for numerical stability.
    3. Per-example target length normalization: loss divided by target length U_i per sample,
       then averaged across batch.
    4. Inference path is audio-only: no text tokens or teacher representations injected.
    """

    def __init__(
        self,
        encoder_adapter: ChunkFormerEncoderAdapter,
        vocab_size: int = 1024,
        blank_id: int = 0,
        ctc_head_dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder_adapter = encoder_adapter
        self.vocab_size = vocab_size
        self.blank_id = blank_id

        self.ctc_head = CTCHead(
            hidden_size=encoder_adapter.output_size,
            vocab_size=vocab_size,
            dropout=ctc_head_dropout,
        )

    def encode(
        self,
        feats: torch.Tensor,
        lengths: torch.Tensor,
        context_policy: str = "full_utterance",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Delegate encoding to ChunkFormer encoder adapter."""
        return self.encoder_adapter.encode(feats, lengths, context_policy=context_policy)

    def forward(
        self,
        feats: torch.Tensor,
        lengths: torch.Tensor,
        context_policy: str = "full_utterance",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder and CTC head.

        Args:
            feats: Tensor [B, F, 80] input filterbank features
            lengths: Tensor [B] number of frames per sample
            context_policy: "full_utterance"

        Returns:
            logits: Tensor [B, T, vocab_size] unnormalized CTC logits
            output_lengths: Tensor [B] int64 sequence length after subsampling
            valid_mask: Tensor [B, T] bool mask (True=speech, False=padding)
        """
        H, output_lengths, valid_mask = self.encode(feats, lengths, context_policy=context_policy)
        logits = self.ctc_head(H)
        return logits, output_lengths, valid_mask

    def compute_ctc_loss(
        self,
        logits: torch.Tensor,
        output_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-example target-normalized CTC loss in FP32.

        Args:
            logits: Tensor [B, T, vocab_size] unnormalized logits
            output_lengths: Tensor [B] int64 output lengths T_out
            targets: Tensor [B, U_max] or 1D concatenated [sum U_i]
            target_lengths: Tensor [B] int64 target lengths U_i

        Returns:
            mean_loss: Scalar float32 loss averaged across batch
            per_example_loss: Tensor [B] float32 loss per sample
        """
        # Ensure log_probs are computed in FP32
        log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)  # [T, B, V]

        # Compute raw CTC loss per sample without reduction
        raw_losses = F.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=output_lengths,
            target_lengths=target_lengths,
            blank=self.blank_id,
            reduction="none",
            zero_infinity=False,
        )  # [B]

        # Engineering Invariant 2: Per-example normalization by target length U_i
        target_lens_float = target_lengths.float().clamp_min(1.0)
        per_example_loss = raw_losses / target_lens_float  # [B]
        mean_loss = per_example_loss.mean()

        return mean_loss, per_example_loss

    @torch.no_grad()
    def transcribe(
        self,
        feats: torch.Tensor,
        lengths: torch.Tensor,
        context_policy: str = "full_utterance",
    ) -> List[List[int]]:
        """Greedy CTC transcription with CTC collapse.

        Args:
            feats: Tensor [B, F, 80]
            lengths: Tensor [B]
            context_policy: "full_utterance"

        Returns:
            hypotheses: List of predicted token ID sequences
        """
        self.eval()
        logits, output_lengths, _ = self.forward(feats, lengths, context_policy=context_policy)
        best_paths = torch.argmax(logits, dim=-1)  # [B, T]

        hypotheses: List[List[int]] = []
        for i in range(feats.size(0)):
            valid_T = int(output_lengths[i].item())
            path = best_paths[i, :valid_T].tolist()

            hyp: List[int] = []
            prev_token = -1
            for tok in path:
                if tok != prev_token:
                    if tok != self.blank_id:
                        hyp.append(tok)
                    prev_token = tok
            hypotheses.append(hyp)

        return hypotheses
