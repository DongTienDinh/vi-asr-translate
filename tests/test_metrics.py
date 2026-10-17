"""Unit tests for evaluation metrics (WER, CER, BLEU, chrF++, paired bootstrap)."""

import unittest
from vi_asr.evaluation.bootstrap import paired_bootstrap_test
from vi_asr.evaluation.metrics import compute_wer_cer
from vi_asr.evaluation.translation_metrics import compute_bleu_chrf


class TestEvaluationMetrics(unittest.TestCase):
    def test_wer_cer_computation(self):
        """Verify WER and CER calculations match exact edit distance breakdowns."""
        references = ["hà nội mùa thu đẹp", "thành phố hồ chí minh"]
        hypotheses = ["hà nội mùa thu", "thành phố hồ chí"]

        res = compute_wer_cer(hypotheses, references)

        # Reference 1 has 5 words, hyp has 4 words (1 deletion)
        # Reference 2 has 5 words, hyp has 4 words (1 deletion)
        # Total words = 10, total deletions = 2 -> WER = 2/10 = 0.20
        self.assertAlmostEqual(res["wer"], 0.20, places=3)
        self.assertEqual(res["wer_stats"]["deletions"], 2)
        self.assertEqual(res["wer_stats"]["total_units"], 10)
        self.assertGreater(res["cer"], 0.0)

    def test_bleu_chrf_computation(self):
        """Verify sacreBLEU and chrF++ computation."""
        references = ["Hanoi autumn is beautiful and peaceful today.", "Good morning Vietnam."]
        hypotheses = ["Hanoi autumn is beautiful and peaceful today.", "Good morning Vietnam."]

        res = compute_bleu_chrf(hypotheses, references)

        # Perfect match -> BLEU = 100.0
        self.assertAlmostEqual(res["bleu"], 100.0, places=1)
        self.assertGreater(res["chrf"], 90.0)

    def test_paired_bootstrap(self):
        """Verify paired bootstrap hypothesis test runs and returns valid confidence intervals and p-value."""
        references = [f"sentence number {i}" for i in range(20)]
        hyps_a = [f"sentence number {i}" for i in range(20)]  # perfect
        hyps_b = [f"sentence wrong {i}" for i in range(20)]   # worse

        def wer_metric(hyps, refs):
            return compute_wer_cer(hyps, refs)["wer"]

        boot_res = paired_bootstrap_test(
            metric_fn=wer_metric,
            hyps_a=hyps_a,
            hyps_b=hyps_b,
            references=references,
            num_resamples=50,
            seed=42,
        )

        self.assertEqual(boot_res["score_a"], 0.0)
        self.assertGreater(boot_res["score_b"], 0.0)
        self.assertGreater(boot_res["delta"], 0.0)
        self.assertIsInstance(boot_res["p_value"], float)
        self.assertLessEqual(boot_res["p_value"], 1.0)


if __name__ == "__main__":
    unittest.main()
