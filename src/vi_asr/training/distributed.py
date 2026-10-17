"""Distributed Data Parallel (DDP) utilities and gradient normalization (Gate G5)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def init_distributed(
    backend: Optional[str] = None,
) -> Tuple[int, int, int]:
    """Initialize torch.distributed process group if running in distributed environment.

    Returns:
        (rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if not dist.is_initialized():
            if backend is None:
                backend = "nccl" if torch.cuda.is_available() else "gloo"

            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)

            dist.init_process_group(
                backend=backend,
                init_method="env://",
            )
            logger.info("Initialized DDP process group: rank %d/%d (local_rank %d, backend %s)", rank, world_size, local_rank, backend)
        return rank, world_size, local_rank

    return 0, 1, 0


def cleanup_distributed() -> None:
    """Clean up torch.distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce sum across all distributed ranks."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_reduce_sample_count(local_count: int, device: torch.device) -> int:
    """All-reduce total number of samples across ranks in an accumulation step."""
    count_t = torch.tensor([local_count], dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
    return int(count_t.item())


def rescale_gradients_by_sample_count(
    parameters: Any,
    effective_sample_count: int,
    nominal_sample_count: int = 32,
) -> None:
    """Rescale accumulated gradients by (nominal_sample_count / effective_sample_count).

    Invariant 4 (Section 16.2):
    Microbatch backward scaled by 1/D. Final step all-reduces total sample count W
    and rescales gradients by (R * D / W) where R=2, D=16 (or R * A * D = 32).
    """
    if effective_sample_count <= 0 or nominal_sample_count <= 0:
        return

    scale_factor = float(nominal_sample_count) / float(effective_sample_count)
    if abs(scale_factor - 1.0) > 1e-6:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale_factor)
