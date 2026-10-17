"""SentencePiece BPE Tokenizer and CTC collapse decoder."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Union

import sentencepiece as spm

from vi_asr.data.normalize import normalize_text_asr


def train_bpe_tokenizer(
    transcripts: Iterable[str],
    model_prefix: Union[str, Path],
    vocab_size: int = 1024,
    character_coverage: float = 1.0,
    pad_id: int = 0,
    unk_id: int = 1,
    bos_id: int = -1,
    eos_id: int = -1,
) -> Path:
    """Train a SentencePiece BPE tokenizer strictly from train transcripts.

    Engineering Invariants:
    1. character_coverage=1.0 ensures all Vietnamese diacritics are covered without UNKs.
    2. pad_id=0 doubles as the CTC blank token.
    3. bos and eos are disabled (-1) for pure CTC modeling.
    """
    prefix_path = Path(model_prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        tmp_corpus = f.name
        for line in transcripts:
            norm_line = normalize_text_asr(line)
            if norm_line:
                f.write(norm_line + "\n")

    cmd = (
        f"--input={tmp_corpus} "
        f"--model_prefix={prefix_path} "
        f"--vocab_size={vocab_size} "
        f"--character_coverage={character_coverage} "
        f"--model_type=bpe "
        f"--pad_id={pad_id} "
        f"--unk_id={unk_id} "
        f"--bos_id={bos_id} "
        f"--eos_id={eos_id} "
        f"--pad_piece=<blank> "
        f"--unk_piece=<unk>"
    )

    spm.SentencePieceTrainer.train(cmd)
    Path(tmp_corpus).unlink(missing_ok=True)
    return prefix_path.with_suffix(".model")


class Tokenizer:
    """Wrapper around SentencePieceProcessor with CTC decoding support."""

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Tokenizer model not found: {self.model_path}")

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(self.model_path))

        self.pad_id = self.sp.pad_id()
        self.blank_id = self.pad_id
        self.unk_id = self.sp.unk_id()
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text: str) -> List[int]:
        """Normalize text and return list of token IDs."""
        norm_text = normalize_text_asr(text)
        return self.sp.encode(norm_text, out_type=int)

    def decode(self, token_ids: List[int]) -> str:
        """Decode list of token IDs back into text."""
        # Filter out pad/blank tokens
        filtered = [tid for tid in token_ids if tid != self.blank_id and tid != self.pad_id]
        return self.sp.decode(filtered)

    def ctc_collapse(self, token_ids: List[int]) -> str:
        """Apply CTC collapse rule:

        1. Collapse consecutive repeated tokens: [a, a] -> [a]
        2. Remove blank/padding tokens.
        3. Retain tokens separated by blank: [a, blank, a] -> [a, a]
        """
        collapsed_ids: List[int] = []
        prev_id: Optional[int] = None

        for tid in token_ids:
            if tid != prev_id:
                if tid != self.blank_id:
                    collapsed_ids.append(tid)
                prev_id = tid

        return self.decode(collapsed_ids)

    @staticmethod
    def check_ctc_validity(target_ids: List[int], output_frames: int) -> bool:
        """Verify CTC validity inequality: T_out >= U + R.

        Args:
            target_ids: Target token ID sequence of length U.
            output_frames: Number of acoustic output frames T_out from encoder subsampling.

        Returns:
            True if sample is valid for CTC training, False if mathematically impossible.
        """
        U = len(target_ids)
        if U == 0:
            return output_frames >= 0

        # Count consecutive repeated token pairs
        R = sum(1 for i in range(1, U) if target_ids[i] == target_ids[i - 1])
        min_required_frames = U + R
        return output_frames >= min_required_frames
