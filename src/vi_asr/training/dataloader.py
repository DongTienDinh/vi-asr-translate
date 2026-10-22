"""Deterministic DataLoader runtime configuration helpers."""

from __future__ import annotations

from typing import Any, Dict

import torch


def dataloader_kwargs(
    runtime_config: Dict[str, Any],
    *,
    cuda_available: bool,
    generator_seed: int,
) -> Dict[str, Any]:
    """Translate a runtime profile into valid ``DataLoader`` kwargs."""
    num_workers = max(0, int(runtime_config.get("num_workers_per_rank", 0)))
    kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(runtime_config.get("pin_memory", cuda_available)) and cuda_available,
    }

    generator = torch.Generator()
    generator.manual_seed(int(generator_seed))
    kwargs["generator"] = generator

    if num_workers > 0:
        kwargs["persistent_workers"] = bool(runtime_config.get("persistent_workers", False))
        prefetch_factor = runtime_config.get("prefetch_factor", 2)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        start_method = runtime_config.get("dataloader_start_method")
        if start_method:
            kwargs["multiprocessing_context"] = str(start_method)

    return kwargs


def reseed_loader_for_epoch(loader: Any, *, base_seed: int, epoch: int, rank: int) -> int:
    """Set an epoch/rank-specific worker seed and return the chosen seed."""
    epoch_seed = int(base_seed) + 100_003 * int(epoch) + 1_009 * int(rank)
    generator = getattr(loader, "generator", None)
    if generator is not None:
        generator.manual_seed(epoch_seed)
    return epoch_seed
