"""Manifest data models, region mappings, and JSONL serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Standard 3-region classification for ViMD
REGION_NAME_TO_ID: Dict[str, int] = {
    "North": 0,
    "Bac": 0,
    "Miền Bắc": 0,
    "Central": 1,
    "Trung": 1,
    "Miền Trung": 1,
    "South": 2,
    "Nam": 2,
    "Miền Nam": 2,
}

REGION_ID_TO_NAME: Dict[int, str] = {
    0: "North",
    1: "Central",
    2: "South",
}


@dataclass
class ManifestItem:
    """Canonical metadata schema for each speech utterance."""

    sample_id: str
    split: str  # "train", "valid", "test"
    audio_relpath: str  # relative to data_root
    source_revision: str
    sample_rate: int
    num_samples: int
    duration: float  # seconds
    text_raw: str
    text_asr: str  # NFC normalized, punctuation-free, lowercased
    speaker_id: Optional[str] = None
    region_id: Optional[int] = None  # 0: North, 1: Central, 2: South
    province: Optional[str] = None
    gender: Optional[str] = None
    text_teacher: Optional[str] = None  # word segmented for PhoBERT
    audio_hash: Optional[str] = None
    text_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ManifestItem:
        return cls(
            sample_id=str(d["sample_id"]),
            split=str(d["split"]),
            audio_relpath=str(d["audio_relpath"]),
            source_revision=str(d.get("source_revision", "unknown")),
            sample_rate=int(d.get("sample_rate", 16000)),
            num_samples=int(d["num_samples"]),
            duration=float(d["duration"]),
            text_raw=str(d["text_raw"]),
            text_asr=str(d["text_asr"]),
            speaker_id=str(d["speaker_id"]) if d.get("speaker_id") is not None else None,
            region_id=int(d["region_id"]) if d.get("region_id") is not None else None,
            province=str(d["province"]) if d.get("province") is not None else None,
            gender=str(d["gender"]) if d.get("gender") is not None else None,
            text_teacher=str(d["text_teacher"]) if d.get("text_teacher") is not None else None,
            audio_hash=str(d["audio_hash"]) if d.get("audio_hash") is not None else None,
            text_hash=str(d["text_hash"]) if d.get("text_hash") is not None else None,
        )


def save_manifest(items: List[ManifestItem], output_path: Union[str, Path]) -> None:
    """Write list of manifest items to a JSONL file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


def load_manifest(manifest_path: Union[str, Path]) -> List[ManifestItem]:
    """Read a JSONL manifest file into a list of ManifestItem objects."""
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest file not found: {p}")
    items: List[ManifestItem] = []
    with open(p, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                items.append(ManifestItem.from_dict(data))
            except Exception as e:
                raise ValueError(f"Error parsing line {line_idx + 1} in {p}: {e}") from e
    return items
