"""Machine translation evaluation metrics (sacreBLEU and chrF++)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import sacrebleu


def compute_bleu_chrf(
    hypotheses: Sequence[str],
    references: Sequence[str],
) -> Dict[str, Any]:
    """Compute corpus-level BLEU and chrF++ scores.

    Args:
        hypotheses: List of English candidate translations
        references: List of English reference translations

    Returns:
        Dict with "bleu", "chrf", and full signature strings.
    """
    assert len(hypotheses) == len(references), "Count mismatch between hypotheses and references"
    if len(hypotheses) == 0:
        return {"bleu": 0.0, "chrf": 0.0, "bleu_signature": "", "chrf_signature": ""}

    # sacrebleu expects list of reference streams (e.g. [[ref1, ref2, ...]])
    ref_stream = [list(references)]
    hyps = list(hypotheses)

    bleu = sacrebleu.corpus_bleu(hyps, ref_stream)
    chrf = sacrebleu.corpus_chrf(hyps, ref_stream, word_order=2)  # chrF++ with word order=2

    return {
        "bleu": float(bleu.score),
        "chrf": float(chrf.score),
        "bleu_signature": str(bleu),
        "chrf_signature": str(chrf),
    }
