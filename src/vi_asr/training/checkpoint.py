"""Atomic checkpoint saving and resumption for DDP training (Section 18.2, Gate G6)."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def capture_rng_states(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Capture complete RNG states across Python, NumPy, PyTorch CPU and CUDA."""
    states: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: Dict[str, Any]) -> None:
    """Restore all RNG states deterministically."""
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch" in states:
        torch.set_rng_state(states["torch"])
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


def save_training_checkpoint(
    checkpoint_dir: Union[str, Path],
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
) -> Path:
    """Save training checkpoint atomically using temporary file and atomic rename.

    Invariant:
    Unwraps DDP `model.module` so checkpoints can be loaded on single GPU or CPU.
    Saves complete RNG states so resumption exactly mirrors uninterrupted trajectory.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP model
    raw_model = model.module if hasattr(model, "module") else model

    checkpoint_data: Dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_states": capture_rng_states(),
        "metrics": metrics or {},
        "config": config or {},
    }

    if scaler is not None and hasattr(scaler, "state_dict"):
        checkpoint_data["scaler_state_dict"] = scaler.state_dict()

    if scheduler is not None and hasattr(scheduler, "state_dict"):
        checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

    final_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    tmp_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt.tmp"

    torch.save(checkpoint_data, tmp_path)
    os.replace(tmp_path, final_path)  # Atomic on POSIX and Windows

    # Update latest symlink/copy
    latest_path = ckpt_dir / "checkpoint_latest.pt"
    tmp_latest = ckpt_dir / "checkpoint_latest.pt.tmp"
    torch.save(checkpoint_data, tmp_latest)
    os.replace(tmp_latest, latest_path)

    if is_best:
        best_path = ckpt_dir / "checkpoint_best.pt"
        tmp_best = ckpt_dir / "checkpoint_best.pt.tmp"
        torch.save(checkpoint_data, tmp_best)
        os.replace(tmp_best, best_path)

    logger.info("Saved atomic checkpoint to %s (epoch %d, step %d)", final_path, epoch, global_step)
    return final_path


def load_training_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    restore_rng: bool = True,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load training checkpoint and restore all model, optimizer, scaler, and RNG states."""
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_p}")

    data = torch.load(ckpt_p, map_location=device, weights_only=False)

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(data["model_state_dict"], strict=True)

    if optimizer is not None and "optimizer_state_dict" in data:
        optimizer.load_state_dict(data["optimizer_state_dict"])

    if scaler is not None and "scaler_state_dict" in data and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(data["scaler_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in data and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(data["scheduler_state_dict"])

    if restore_rng and "rng_states" in data:
        restore_rng_states(data["rng_states"])

    logger.info("Resumed from checkpoint %s at epoch %d, step %d", ckpt_p, data.get("epoch", 0), data.get("global_step", 0))
    return data
