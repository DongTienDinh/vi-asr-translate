"""Tests for profile-driven DataLoader execution."""

import unittest

from vi_asr.training.dataloader import dataloader_kwargs, reseed_loader_for_epoch


class _Loader:
    def __init__(self, generator):
        self.generator = generator


class TestDataLoaderRuntime(unittest.TestCase):
    def test_worker_options_are_applied(self):
        options = dataloader_kwargs(
            {
                "num_workers_per_rank": 1,
                "pin_memory": True,
                "persistent_workers": False,
                "prefetch_factor": 2,
                "dataloader_start_method": "spawn",
            },
            cuda_available=True,
            generator_seed=42,
        )
        self.assertEqual(options["num_workers"], 1)
        self.assertTrue(options["pin_memory"])
        self.assertEqual(options["prefetch_factor"], 2)
        self.assertEqual(options["multiprocessing_context"], "spawn")

    def test_worker_only_options_are_omitted_for_single_process(self):
        options = dataloader_kwargs(
            {"num_workers_per_rank": 0, "prefetch_factor": 2},
            cuda_available=False,
            generator_seed=42,
        )
        self.assertEqual(options["num_workers"], 0)
        self.assertNotIn("prefetch_factor", options)
        self.assertNotIn("multiprocessing_context", options)

    def test_epoch_reseed_is_deterministic_and_rank_specific(self):
        options = dataloader_kwargs({}, cuda_available=False, generator_seed=0)
        loader = _Loader(options["generator"])
        chosen = reseed_loader_for_epoch(loader, base_seed=42, epoch=3, rank=1)
        self.assertEqual(chosen, 301_060)
        self.assertEqual(loader.generator.initial_seed(), chosen)


if __name__ == "__main__":
    unittest.main()
