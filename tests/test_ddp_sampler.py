"""Unit tests for GlobalDurationSampler and ValidationShardSampler (Gate G5, G7)."""

import unittest
from vi_asr.training.distributed_sampler import GlobalDurationSampler, ValidationShardSampler


class TestDistributedSamplers(unittest.TestCase):
    def test_global_duration_sampler_sync_and_sharding(self):
        """Verify GlobalDurationSampler yields exactly equal microstep counts across ranks without sample collision."""
        num_samples = 45
        lengths = [100 + (i * 17) % 500 for i in range(num_samples)]
        world_size = 2

        sampler_r0 = GlobalDurationSampler(
            lengths=lengths,
            world_size=world_size,
            rank=0,
            max_samples_per_rank=2,
            max_frames_per_rank=6400,
            seed=42,
            shuffle=True,
            drop_incomplete_group=True,
        )

        sampler_r1 = GlobalDurationSampler(
            lengths=lengths,
            world_size=world_size,
            rank=1,
            max_samples_per_rank=2,
            max_frames_per_rank=6400,
            seed=42,
            shuffle=True,
            drop_incomplete_group=True,
        )

        batches_r0 = list(sampler_r0)
        batches_r1 = list(sampler_r1)

        # Gate G5 assertion 1: Exact equal microstep count across ranks
        self.assertEqual(len(batches_r0), len(batches_r1))
        self.assertGreater(len(batches_r0), 0)

        # Gate G5 assertion 2: No collision between ranks in the same microstep
        for b0, b1 in zip(batches_r0, batches_r1):
            set0 = set(b0)
            set1 = set(b1)
            self.assertEqual(len(set0.intersection(set1)), 0, "Ranks must not process identical samples in the same step")

    def test_validation_shard_sampler_exact_coverage(self):
        """Verify ValidationShardSampler partitions dataset with zero drops and zero duplicates (Gate G7)."""
        num_samples = 77  # odd number
        world_size = 2

        s0 = ValidationShardSampler(num_samples=num_samples, world_size=world_size, rank=0)
        s1 = ValidationShardSampler(num_samples=num_samples, world_size=world_size, rank=1)

        idx0 = list(s0)
        idx1 = list(s1)

        # Disjoint
        self.assertEqual(len(set(idx0).intersection(set(idx1))), 0)

        # Complete coverage
        all_gathered = sorted(idx0 + idx1)
        self.assertEqual(all_gathered, list(range(num_samples)))


if __name__ == "__main__":
    unittest.main()
