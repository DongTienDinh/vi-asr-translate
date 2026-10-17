"""VinAI Translate vi2en cascade evaluator (Section 25)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from vi_asr.evaluation.translation_metrics import compute_bleu_chrf

logger = logging.getLogger(__name__)


class VinAITranslator:
    """Wrapper for frozen vinai/vinai-translate-vi2en-v2 machine translation."""

    def __init__(
        self,
        model_id: str = "vinai/vinai-translate-vi2en-v2",
        revision: str = "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_beams: int = 5,
        max_new_tokens: int = 256,
    ):
        self.model_id = model_id
        self.revision = revision
        self.device = torch.device(device)
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

        logger.info("Loading VinAI Translate model: %s (revision: %s)...", model_id, revision)
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            src_lang="vi_VN",
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            revision=revision,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def translate_batch(self, texts: Sequence[str]) -> List[str]:
        """Translate a batch of Vietnamese sentences to English."""
        if len(texts) == 0:
            return []

        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        translated_tokens = self.model.generate(
            **inputs,
            decoder_start_token_id=self.tokenizer.lang_code_to_id["en_XX"],
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        results = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return results

    def evaluate_cascade(
        self,
        asr_transcriptions: Sequence[str],
        english_references: Sequence[str],
    ) -> Dict[str, Any]:
        """Run cascade MT evaluation and compute BLEU and chrF++."""
        translations = self.translate_batch(asr_transcriptions)
        metrics = compute_bleu_chrf(translations, english_references)
        metrics["translations"] = translations
        return metrics
