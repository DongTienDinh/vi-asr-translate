"""Audio frontend for Kaldi-compliant filterbank feature extraction."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as kaldi


class GlobalCMVN(nn.Module):
    """Global Cepstral Mean and Variance Normalization.

    Normalizes features: (x - mean) * istd.
    Contract: Must be applied exactly once per feature sequence.
    """

    def __init__(self, mean: torch.Tensor, istd: torch.Tensor, norm_var: bool = True):
        super().__init__()
        assert mean.shape == istd.shape, f"Shape mismatch: {mean.shape} vs {istd.shape}"
        self.norm_var = norm_var
        self.register_buffer("mean", mean.float())
        self.register_buffer("istd", istd.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

        x: (batch, time, feat_dim) or (time, feat_dim)
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x

    @classmethod
    def from_kaldi_file(cls, cmvn_file: Union[str, Path], norm_var: bool = True) -> GlobalCMVN:
        """Load CMVN statistics from Kaldi text file."""
        p = Path(cmvn_file)
        if not p.exists():
            raise FileNotFoundError(f"CMVN file not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            tokens = f.read().split()

        assert tokens[0] == "[" and tokens[-1] == "]", "Invalid Kaldi CMVN format"
        feat_dim = int((len(tokens) - 4) / 2)
        means = [float(tokens[i]) for i in range(1, feat_dim + 1)]
        count = float(tokens[feat_dim + 1])
        variance = [float(tokens[i]) for i in range(feat_dim + 2, 2 * feat_dim + 2)]

        means_arr = np.zeros(feat_dim, dtype=np.float32)
        istd_arr = np.zeros(feat_dim, dtype=np.float32)
        for i in range(feat_dim):
            m = means[i] / count
            v = variance[i] / count - m * m
            if v < 1.0e-20:
                v = 1.0e-20
            means_arr[i] = m
            istd_arr[i] = 1.0 / math.sqrt(v)

        return cls(torch.from_numpy(means_arr), torch.from_numpy(istd_arr), norm_var=norm_var)


class KaldiFrontend(nn.Module):
    """Kaldi-compliant 80-channel filterbank extraction module.

    Engineering Invariants:
    1. Waveform input in float [-1.0, 1.0] MUST be multiplied by 32768.0 before Kaldi fbank.
    2. Output feature dimension: 80 mel channels.
    3. povey window, 25ms frame length, 10ms frame shift.
    4. CMVN applied exactly once if configured.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        window_type: str = "povey",
        energy_floor: float = 0.0,
        train_dither: float = 1.0,
        eval_dither: float = 0.0,
        fbank_waveform_multiplier: float = 32768.0,
        global_cmvn: Optional[GlobalCMVN] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms
        self.window_type = window_type
        self.energy_floor = energy_floor
        self.train_dither = train_dither
        self.eval_dither = eval_dither
        self.fbank_waveform_multiplier = fbank_waveform_multiplier
        self.global_cmvn = global_cmvn

    def extract_fbank_single(
        self,
        waveform: torch.Tensor,
        dither: Optional[float] = None,
    ) -> torch.Tensor:
        """Extract fbank for a single 1D waveform tensor [Time]."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        assert waveform.dim() == 2, f"Expected (1, Time), got {waveform.shape}"

        if dither is None:
            dither = self.train_dither if self.training else self.eval_dither

        # CRITICAL Kaldi scale: multiply float [-1.0, 1.0] by 32768.0
        scaled_waveform = waveform * self.fbank_waveform_multiplier

        feat = kaldi.fbank(
            scaled_waveform,
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length_ms,
            frame_shift=self.frame_shift_ms,
            dither=dither,
            energy_floor=self.energy_floor,
            sample_frequency=self.sample_rate,
            window_type=self.window_type,
        )
        return feat  # [T, 80]

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract fbank for a batch of waveforms.

        Args:
            waveforms: Tensor of shape [B, Time]
            waveform_lengths: Optional Tensor of shape [B] specifying valid lengths in samples

        Returns:
            feats: Padded feature tensor of shape [B, T_max, 80]
            feat_lengths: Tensor of shape [B] specifying number of valid frames per item
        """
        device = waveforms.device
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)
        B, max_len = waveforms.shape

        feats_list = []
        feat_lengths_list = []

        for b in range(B):
            cur_len = int(waveform_lengths[b].item()) if waveform_lengths is not None else max_len
            wav_single = waveforms[b, :cur_len]
            feat_single = self.extract_fbank_single(wav_single)
            feats_list.append(feat_single)
            feat_lengths_list.append(feat_single.size(0))

        max_feat_len = max(feat_lengths_list)
        feats_padded = torch.zeros(B, max_feat_len, self.num_mel_bins, device=device, dtype=torch.float32)

        for b, feat in enumerate(feats_list):
            feats_padded[b, : feat.size(0), :] = feat.to(device)

        feat_lengths = torch.tensor(feat_lengths_list, device=device, dtype=torch.int64)

        if self.global_cmvn is not None:
            feats_padded = self.global_cmvn(feats_padded)

        return feats_padded, feat_lengths
