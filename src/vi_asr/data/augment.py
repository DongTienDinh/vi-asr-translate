"""SpecAugment data augmentation module for acoustic filterbanks."""

from __future__ import annotations

import random
from typing import Optional, Tuple
import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """SpecAugment on filterbank features.

    Applies frequency masking and adaptive time masking on speech spectrograms.
    Only active when module is in training mode (`self.training == True`).

    Engineering Invariants:
    1. Applied after CMVN and before encoder subsampling.
    2. Time mask width is capped by max_time_fraction_per_mask to protect short utterances.
    """

    def __init__(
        self,
        freq_masks: int = 2,
        freq_max_bins: int = 8,
        time_masks: int = 2,
        time_max_frames: int = 30,
        max_time_fraction_per_mask: float = 0.1,
        replace_with_zero: bool = True,
    ):
        super().__init__()
        self.freq_masks = freq_masks
        self.freq_max_bins = freq_max_bins
        self.time_masks = time_masks
        self.time_max_frames = time_max_frames
        self.max_time_fraction_per_mask = max_time_fraction_per_mask
        self.replace_with_zero = replace_with_zero

    def forward(
        self,
        feats: torch.Tensor,
        feat_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Args:

            feats: Tensor of shape [B, T, F] (mel filterbanks)
            feat_lengths: Tensor of shape [B] (actual valid frames)

        Returns:
            augmented_feats: Tensor of shape [B, T, F]
            feat_lengths: Unchanged valid lengths
        """
        if not self.training:
            return feats, feat_lengths

        # Clone so input is not modified in-place
        augmented = feats.clone()
        B, T, F = augmented.shape

        for b in range(B):
            valid_T = int(feat_lengths[b].item()) if feat_lengths is not None else T
            fill_val = 0.0 if self.replace_with_zero else augmented[b, :valid_T, :].mean().item()

            # Frequency masking
            for _ in range(self.freq_masks):
                f_width = random.randint(0, min(self.freq_max_bins, F))
                if f_width > 0:
                    f_start = random.randint(0, F - f_width)
                    augmented[b, :valid_T, f_start : f_start + f_width] = fill_val

            # Time masking capped by utterance duration
            max_t_allowed = min(self.time_max_frames, max(1, int(valid_T * self.max_time_fraction_per_mask)))
            for _ in range(self.time_masks):
                t_width = random.randint(0, max_t_allowed)
                if t_width > 0 and valid_T > t_width:
                    t_start = random.randint(0, valid_T - t_width)
                    augmented[b, t_start : t_start + t_width, :] = fill_val

        return augmented, feat_lengths
