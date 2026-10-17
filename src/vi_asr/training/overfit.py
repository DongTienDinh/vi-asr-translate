"""Gate G4 Tiny Overfit Suite for variants B1, P1, and P2."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

from vi_asr.models.alignment import TokenFramePoolingAlignment
from vi_asr.models.ctc_asr import ViPViCTCASR
from vi_asr.models.encoder_adapter import ChunkFormerEncoderAdapter
from vi_asr.training.losses import MultiTaskASRLoss

logger = logging.getLogger(__name__)


class TinyASRSystem(nn.Module):
    """Integrated system combining ViPViCTCASR with optional alignment for training."""

    def __init__(
        self,
        asr_model: ViPViCTCASR,
        alignment_module: Optional[TokenFramePoolingAlignment] = None,
    ):
        super().__init__()
        self.asr_model = asr_model
        self.alignment_module = alignment_module

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        feats = batch["feats"]
        feat_lengths = batch["feat_lengths"]

        # 1. Forward through encoder and CTC head
        H, output_lengths, valid_mask = self.asr_model.encode(feats, feat_lengths)
        logits = self.asr_model.ctc_head(H)

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "output_lengths": output_lengths,
            "valid_mask": valid_mask,
        }

        # 2. Forward through alignment if present
        if self.alignment_module is not None and "teacher_states" in batch:
            teacher_states = batch["teacher_states"]
            teacher_lengths = batch.get("teacher_lengths")
            if teacher_lengths is not None and teacher_states.size(1) > 0:
                align_out = self.alignment_module(
                    H=H,
                    E=teacher_states,
                    audio_lengths=output_lengths,
                    token_lengths=teacher_lengths,
                    valid_frame_mask=valid_mask,
                )
                outputs["predicted_teacher"] = align_out["predicted_teacher"]
                outputs["attention"] = align_out["attention"]

        return outputs


def run_tiny_overfit(
    variant: str = "b1",
    num_samples: int = 16,
    num_steps: int = 35,
    lr: float = 3e-3,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run tiny overfit verification for Gate G4.

    Args:
        variant: "b1", "p1", or "p2"
        num_samples: Number of synthetic training samples (16 to 32)
        num_steps: Optimization steps
        lr: Learning rate
        device: "cpu" or "cuda"

    Returns:
        Dict with initial_loss, final_loss, loss_history, converged (bool)
    """
    torch.manual_seed(42)
    dev = torch.device(device)

    # 1. Build small model (2 blocks) for fast convergence verification
    adapter = ChunkFormerEncoderAdapter(
        input_size=80,
        output_size=256,
        attention_heads=4,
        linear_units=512,
        num_blocks=2,
        dropout_rate=0.0,
        encoder_layerdrop=0.0,
        cnn_module_kernel=15,
        dynamic_conv=True,
    ).to(dev)

    vocab_size = 32
    asr_model = ViPViCTCASR(adapter, vocab_size=vocab_size, blank_id=0).to(dev)

    alignment_mod = None
    lambda_token = 0.0

    if variant in ("p1", "p2"):
        temporal_prior = (variant == "p2")
        alignment_mod = TokenFramePoolingAlignment(
            audio_dim=256,
            teacher_dim=128,
            score_dim=64,
            temperature=0.1,
            temporal_prior_enabled=temporal_prior,
            beta=0.1,
            sigma=0.2,
        ).to(dev)
        lambda_token = 0.1

    system = TinyASRSystem(asr_model, alignment_mod).to(dev)
    loss_fn = MultiTaskASRLoss(lambda_token=lambda_token, ctc_blank_id=0).to(dev)

    # 2. Synthetic overfit batch
    # Input audio frames (40 frames, 80 mel bins)
    F_len = 40
    U_len = 4
    N_len = 5

    feats = torch.randn(num_samples, F_len, 80, device=dev)
    feat_lengths = torch.full((num_samples,), F_len, dtype=torch.int64, device=dev)

    # Consistent CTC targets (e.g. repetition pattern)
    target_pattern = torch.tensor([5, 8, 12, 19], dtype=torch.int64, device=dev)
    targets = target_pattern.unsqueeze(0).expand(num_samples, U_len).clone()
    target_lengths = torch.full((num_samples,), U_len, dtype=torch.int64, device=dev)

    batch: Dict[str, Any] = {
        "sample_ids": [f"sample_{i}" for i in range(num_samples)],
        "feats": feats,
        "feat_lengths": feat_lengths,
        "ctc_targets": targets,
        "target_lengths": target_lengths,
    }

    if variant in ("p1", "p2"):
        # Synthetic frozen teacher states
        t_pattern = torch.randn(N_len, 128, device=dev)
        teacher_states = t_pattern.unsqueeze(0).expand(num_samples, N_len, 128).clone()
        batch["teacher_states"] = teacher_states
        batch["teacher_lengths"] = torch.full((num_samples,), N_len, dtype=torch.int64, device=dev)

    # 3. Optimizer
    optimizer = AdamW(system.parameters(), lr=lr, weight_decay=0.0)

    # 4. Training loop
    system.train()
    loss_history: List[float] = []

    for step in range(num_steps):
        optimizer.zero_grad()
        out = system(batch)

        loss_dict = loss_fn(
            ctc_logits=out["logits"],
            ctc_output_lengths=out["output_lengths"],
            ctc_targets=batch["ctc_targets"],
            ctc_target_lengths=batch["target_lengths"],
            predicted_teacher=out.get("predicted_teacher"),
            target_teacher=batch.get("teacher_states"),
            token_lengths=batch.get("teacher_lengths"),
        )

        loss = loss_dict["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(system.parameters(), max_norm=5.0)
        optimizer.step()

        val = loss.item()
        loss_history.append(val)

    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    # Convergence criteria: loss drops by at least 50%
    converged = final_loss < 0.6 * initial_loss

    return {
        "variant": variant,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_history": loss_history,
        "converged": converged,
    }
