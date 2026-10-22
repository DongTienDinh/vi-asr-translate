"""Atomic checkpoint saving and resumption for DDP training (Section 18.2, Gate G6)."""

from __future__ import annotations

import logging
import os
import random
import shutil
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
    rng_states_by_rank: Optional[list[Dict[str, Any]]] = None,
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
        "rng_states_by_rank": rng_states_by_rank,
        "metrics": metrics or {},
        "config": config or {},
    }

    if scaler is not None and hasattr(scaler, "state_dict"):
        checkpoint_data["scaler_state_dict"] = scaler.state_dict()

    if scheduler is not None and hasattr(scheduler, "state_dict"):
        checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

    # Serialize exactly once.  The old latest remains valid if torch.save fails.
    latest_path = ckpt_dir / "checkpoint_latest.pt"
    tmp_latest = ckpt_dir / "checkpoint_latest.pt.tmp"
    try:
        torch.save(checkpoint_data, tmp_latest)
        os.replace(tmp_latest, latest_path)
    finally:
        if tmp_latest.exists():
            tmp_latest.unlink()

    def replace_link_or_copy(source: Path, destination: Path) -> None:
        """Atomically point to one checkpoint inode; copy only if links are unavailable."""
        temporary = destination.with_suffix(destination.suffix + ".link.tmp")
        if temporary.exists():
            temporary.unlink()
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)

    # Keep one human-readable epoch name, but prune older epoch snapshots.  On
    # Kaggle/NTFS this is normally a hard link and consumes no duplicate space.
    final_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    replace_link_or_copy(latest_path, final_path)
    for old_epoch in ckpt_dir.glob("checkpoint_epoch_*.pt"):
        if old_epoch != final_path:
            old_epoch.unlink()

    if is_best:
        best_path = ckpt_dir / "checkpoint_best.pt"
        replace_link_or_copy(latest_path, best_path)

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

    if restore_rng:
        rank_states = data.get("rng_states_by_rank")
        if rank_states:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank >= len(rank_states):
                raise RuntimeError(
                    f"Checkpoint has RNG state for {len(rank_states)} ranks, cannot restore rank {rank}"
                )
            restore_rng_states(rank_states[rank])
        elif "rng_states" in data:
            restore_rng_states(data["rng_states"])

    logger.info("Resumed from checkpoint %s at epoch %d, step %d", ckpt_p, data.get("epoch", 0), data.get("global_step", 0))
    return data
