"""Paired bootstrap resampling for statistical significance testing (Section 24)."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np


def paired_bootstrap_test(
    metric_fn: Callable[[Sequence[str], Sequence[str]], float],
    hyps_a: Sequence[str],
    hyps_b: Sequence[str],
    references: Sequence[str],
    num_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Perform paired bootstrap hypothesis test comparing system A vs system B.

    Returns:
        Dict with:
            "score_a": float,
            "score_b": float,
            "delta": float (score_b - score_a),
            "ci_95": Tuple[float, float] (2.5% to 97.5% interval for delta),
            "p_value": float (two-sided empirical p-value)
    """
    assert len(hyps_a) == len(hyps_b) == len(references), "Length mismatch"
    N = len(references)
    if N == 0:
        return {"score_a": 0.0, "score_b": 0.0, "delta": 0.0, "ci_95": (0.0, 0.0), "p_value": 1.0}

    orig_a = metric_fn(hyps_a, references)
    orig_b = metric_fn(hyps_b, references)
    orig_delta = orig_b - orig_a

    rng = random.Random(seed)
    deltas = []

    for _ in range(num_resamples):
        sample_indices = [rng.randint(0, N - 1) for _ in range(N)]
        sampled_refs = [references[i] for i in sample_indices]
        sampled_a = [hyps_a[i] for i in sample_indices]
        sampled_b = [hyps_b[i] for i in sample_indices]

        sc_a = metric_fn(sampled_a, sampled_refs)
        sc_b = metric_fn(sampled_b, sampled_refs)
        deltas.append(sc_b - sc_a)

    deltas_arr = np.array(deltas)
    ci_lower = float(np.percentile(deltas_arr, 2.5))
    ci_upper = float(np.percentile(deltas_arr, 97.5))

    # Two-sided empirical p-value
    p_less = np.mean(deltas_arr <= 0)
    p_greater = np.mean(deltas_arr >= 0)
    p_value = float(2.0 * min(p_less, p_greater))
    p_value = min(1.0, p_value)

    return {
        "score_a": float(orig_a),
        "score_b": float(orig_b),
        "delta": float(orig_delta),
        "ci_95": (ci_lower, ci_upper),
        "p_value": p_value,
    }
