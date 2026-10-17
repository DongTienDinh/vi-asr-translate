"""Text normalization policies and hashing utilities for Vietnamese ASR."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Union


def normalize_unicode_nfc(text: str) -> str:
    """Normalize text into canonical Unicode NFC form."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)


def normalize_text_asr(text: str) -> str:
    """Canonical normalization policy for ASR transcripts and evaluation.

    Rules:
    1. Unicode NFC normalization.
    2. Lowercase.
    3. Retain Vietnamese alphabetic characters, digits, and spaces.
    4. Remove punctuation marks.
    5. Collapse redundant whitespace and trim.
    """
    text = normalize_unicode_nfc(text)
    text = text.lower()

    # Retain Vietnamese letters, latin letters, numbers, and whitespace.
    # Exclude punctuation such as commas, periods, quotes, exclamation marks, etc.
    # Vietnamese diacritic characters are fully preserved in Unicode NFC.
    text = re.sub(r"[^\w\s\d]", " ", text, flags=re.UNICODE)
    # Replace underscores with spaces (in case input was pre-segmented)
    text = text.replace("_", " ")
    # Collapse multiple whitespace characters into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text_raw(text: str) -> str:
    """Normalize raw text preserving punctuation and casing, with NFC normalization."""
    text = normalize_unicode_nfc(text)
    return re.sub(r"\s+", " ", text).strip()


def compute_text_hash(text: str) -> str:
    """Compute deterministic SHA256 of normalized text."""
    nfc_text = normalize_unicode_nfc(text)
    return hashlib.sha256(nfc_text.encode("utf-8")).hexdigest()


def compute_audio_hash(waveform_bytes: Union[bytes, bytearray]) -> str:
    """Compute deterministic SHA256 of raw audio bytes."""
    return hashlib.sha256(waveform_bytes).hexdigest()
