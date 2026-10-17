"""ASR evaluation metrics (WER, CER, and detailed error breakdown)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import jiwer


@dataclass
class ErrorStats:
    """Detailed error counts for ASR evaluation."""
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0
    total_reference_units: int = 0

    @property
    def error_rate(self) -> float:
        if self.total_reference_units == 0:
            return 0.0
        return float(self.substitutions + self.deletions + self.insertions) / float(self.total_reference_units)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_rate": self.error_rate,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "hits": self.hits,
            "total_units": self.total_reference_units,
        }


def compute_wer_cer(
    hypotheses: Sequence[str],
    references: Sequence[str],
) -> Dict[str, Any]:
    """Compute corpus-level WER and CER with exact error breakdowns.

    Invariant 5:
    Corpus WER is computed strictly as (sum S + sum D + sum I) / (sum N),
    NEVER the mean of sentence-level error rates.
    """
    assert len(hypotheses) == len(references), "Count mismatch between hypotheses and references"
    if len(hypotheses) == 0:
        return {
            "wer": 0.0,
            "cer": 0.0,
            "wer_stats": ErrorStats().to_dict(),
            "cer_stats": ErrorStats().to_dict(),
        }

    # 1. Word Error Rate (WER)
    wer_out = jiwer.process_words(list(references), list(hypotheses))
    wer_stats = ErrorStats(
        substitutions=wer_out.substitutions,
        deletions=wer_out.deletions,
        insertions=wer_out.insertions,
        hits=wer_out.hits,
        total_reference_units=wer_out.substitutions + wer_out.deletions + wer_out.hits,
    )

    # 2. Character Error Rate (CER)
    cer_out = jiwer.process_characters(list(references), list(hypotheses))
    cer_stats = ErrorStats(
        substitutions=cer_out.substitutions,
        deletions=cer_out.deletions,
        insertions=cer_out.insertions,
        hits=cer_out.hits,
        total_reference_units=cer_out.substitutions + cer_out.deletions + cer_out.hits,
    )

    return {
        "wer": wer_stats.error_rate,
        "cer": cer_stats.error_rate,
        "wer_stats": wer_stats.to_dict(),
        "cer_stats": cer_stats.to_dict(),
    }
