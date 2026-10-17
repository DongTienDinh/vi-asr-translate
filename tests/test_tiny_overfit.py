"""Unit tests for Gate G4 Tiny Overfit Suite (B1, P1, P2 convergence)."""

import unittest
from vi_asr.training.overfit import run_tiny_overfit


class TestTinyOverfitGateG4(unittest.TestCase):
    def test_overfit_variant_b1(self):
        """Verify B1 (pure CTC) loss decreases substantially on toy samples."""
        res = run_tiny_overfit(variant="b1", num_samples=8, num_steps=30, lr=4e-3)
        self.assertTrue(res["converged"], f"B1 failed to converge: {res['initial_loss']} -> {res['final_loss']}")
        self.assertLess(res["final_loss"], res["initial_loss"])

    def test_overfit_variant_p1(self):
        """Verify P1 (CTC + Token Pooling) loss decreases substantially on toy samples."""
        res = run_tiny_overfit(variant="p1", num_samples=8, num_steps=30, lr=4e-3)
        self.assertTrue(res["converged"], f"P1 failed to converge: {res['initial_loss']} -> {res['final_loss']}")
        self.assertLess(res["final_loss"], res["initial_loss"])

    def test_overfit_variant_p2(self):
        """Verify P2 (CTC + Token Pooling + Temporal Prior) loss decreases substantially on toy samples."""
        res = run_tiny_overfit(variant="p2", num_samples=8, num_steps=30, lr=4e-3)
        self.assertTrue(res["converged"], f"P2 failed to converge: {res['initial_loss']} -> {res['final_loss']}")
        self.assertLess(res["final_loss"], res["initial_loss"])


if __name__ == "__main__":
    unittest.main()
