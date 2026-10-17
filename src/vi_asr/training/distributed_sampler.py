"""Global duration-aware distributed sampler and validation shard sampler (Gates G5, G7)."""

from __future__ import annotations

import logging
import math
import random
from typing import Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class GlobalDurationSampler(Sampler[List[int]]):
    """Global Duration-aware Batch Sampler for multi-GPU training.

    Guarantees (Section 16.1):
    1. Uniform microstep count across ranks: ranks always step synchronously.
    2. Equal gradient accumulation boundaries.
    3. Duration bucketing avoids VRAM spikes by grouping similar length utterances.
    4. Deterministic shuffle: rng seeded by `seed + epoch`.
    5. Yields batches of indices for local rank: List[int].
    """

    def __init__(
        self,
        lengths: Sequence[int],
        world_size: int = 1,
        rank: int = 0,
        max_samples_per_rank: int = 2,
        max_frames_per_rank: int = 6400,
        seed: int = 42,
        shuffle: bool = True,
        drop_incomplete_group: bool = True,
    ):
        self.lengths = list(lengths)
        self.world_size = world_size
        self.rank = rank
        self.max_samples_per_rank = max_samples_per_rank
        self.max_frames_per_rank = max_frames_per_rank
        self.seed = seed
        self.shuffle = shuffle
        self.drop_incomplete_group = drop_incomplete_group
        self.epoch = 0

        self.total_samples = len(self.lengths)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic permutation across epochs."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)

        # 1. Bucket and sort by length with randomized intra-bucket perturbation
        indices = list(range(self.total_samples))
        if self.shuffle:
            # Sort by approximate length with jitter to ensure variety
            indices.sort(key=lambda idx: (self.lengths[idx] // 50, rng.random()))
        else:
            indices.sort(key=lambda idx: self.lengths[idx])

        # 2. Form local microbatches respecting sample count and frame limits
        local_batches: List[List[int]] = []
        current_batch: List[int] = []
        current_frames = 0

        for idx in indices:
            L = self.lengths[idx]
            new_frames = max(current_frames, L) * (len(current_batch) + 1)

            if len(current_batch) >= self.max_samples_per_rank or new_frames > self.max_frames_per_rank:
                if current_batch:
                    local_batches.append(current_batch)
                current_batch = [idx]
                current_frames = L
            else:
                current_batch.append(idx)
                current_frames = max(current_frames, L)

        if current_batch:
            local_batches.append(current_batch)

        # 3. Shuffle microbatch order across the epoch if shuffle is enabled
        if self.shuffle:
            rng.shuffle(local_batches)

        # 4. Group microbatches into groups of size world_size
        num_complete_groups = len(local_batches) // self.world_size
        dropped = len(local_batches) % self.world_size

        if dropped > 0 and self.drop_incomplete_group:
            logger.debug(
                "GlobalDurationSampler: Dropped %d incomplete rank microbatches (%d samples) in epoch %d",
                dropped,
                sum(len(b) for b in local_batches[num_complete_groups * self.world_size :]),
                self.epoch,
            )
            local_batches = local_batches[: num_complete_groups * self.world_size]

        # 5. Yield batches for this specific rank
        # Group g has world_size batches: local_batches[g * R : (g + 1) * R]
        # Rank r takes batch at index: g * R + r
        for g in range(num_complete_groups):
            batch_idx = g * self.world_size + self.rank
            if batch_idx < len(local_batches):
                yield local_batches[batch_idx]

    def __len__(self) -> int:
        num_groups = len(self.lengths) // (self.max_samples_per_rank * self.world_size)
        return max(1, num_groups)


class ValidationShardSampler(Sampler[int]):
    """Validation/Evaluation Shard Sampler (Section 18.1).

    Guarantees:
    1. Zero padding samples, zero dropped samples, zero duplicates.
    2. Exactly partitions dataset across ranks: `indices[rank::world_size]`.
    3. Global validation metrics computed by gathering all ranks without bias.
    """

    def __init__(self, num_samples: int, world_size: int = 1, rank: int = 0):
        self.num_samples = num_samples
        self.world_size = world_size
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        # Simple deterministic stride: rank, rank + world_size, ...
        return iter(range(self.rank, self.num_samples, self.world_size))

    def __len__(self) -> int:
        return math.ceil(max(0, self.num_samples - self.rank) / self.world_size)
