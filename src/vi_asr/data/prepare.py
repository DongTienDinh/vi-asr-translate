"""Data preparation adapter for Hugging Face nguyendv02/ViMD_Dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from vi_asr.data.manifest import REGION_NAME_TO_ID, ManifestItem, save_manifest
from vi_asr.data.normalize import (
    compute_text_hash,
    normalize_text_asr,
    normalize_text_raw,
)
from vi_asr.utils.logging import setup_logger

logger = setup_logger("vi_asr.data.prepare")


def parse_sample_to_manifest_item(
    sample: Dict[str, Any],
    split: str,
    source_revision: str,
    sample_index: int,
) -> Optional[ManifestItem]:
    """Convert raw dataset dictionary item into canonical ManifestItem."""
    sample_id = str(sample.get("id") or sample.get("sample_id") or f"{split}_{sample_index:06d}")

    # Text fields
    raw_text = str(sample.get("transcript") or sample.get("text") or sample.get("sentence") or "")
    text_raw = normalize_text_raw(raw_text)
    text_asr = normalize_text_asr(raw_text)
    if not text_asr:
        return None

    # Regional metadata
    region_raw = str(sample.get("region") or sample.get("dialect") or "")
    region_id = REGION_NAME_TO_ID.get(region_raw)
    province = str(sample.get("province")) if sample.get("province") else None
    gender = str(sample.get("gender")) if sample.get("gender") else None
    speaker_id = str(sample.get("speaker_id") or sample.get("speaker")) if sample.get("speaker_id") or sample.get("speaker") else None

    # Audio metadata
    audio_info = sample.get("audio", {})
    if isinstance(audio_info, dict):
        audio_path = audio_info.get("path", f"audio/{split}/{sample_id}.wav")
        sample_rate = int(audio_info.get("sampling_rate", 16000))
        audio_array = audio_info.get("array")
        num_samples = len(audio_array) if audio_array is not None else int(sample.get("num_samples", 0))
    else:
        audio_path = f"audio/{split}/{sample_id}.wav"
        sample_rate = 16000
        num_samples = int(sample.get("num_samples", 0))

    duration = float(sample.get("duration", num_samples / sample_rate if sample_rate > 0 else 0.0))

    return ManifestItem(
        sample_id=sample_id,
        split=split,
        audio_relpath=audio_path,
        source_revision=source_revision,
        sample_rate=sample_rate,
        num_samples=num_samples,
        duration=duration,
        text_raw=text_raw,
        text_asr=text_asr,
        speaker_id=speaker_id,
        region_id=region_id,
        province=province,
        gender=gender,
        text_teacher=None,
        audio_hash=None,
        text_hash=compute_text_hash(text_asr),
    )


def prepare_manifests_from_hf(
    dataset_dict: Any,
    output_dir: Union[str, Path],
    source_revision: str = "3a5b30157034e7eadd5c75fae1a820c6f9383398",
) -> Dict[str, Path]:
    """Prepare canonical train, valid, test JSONL manifests from a HuggingFace DatasetDict."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    manifest_paths: Dict[str, Path] = {}

    for split in ["train", "validation", "test"]:
        actual_split_key = split
        if actual_split_key not in dataset_dict:
            # Map valid to validation if needed
            if split == "validation" and "valid" in dataset_dict:
                actual_split_key = "valid"
            elif split not in dataset_dict:
                logger.warning(f"Split {split} not found in dataset keys: {list(dataset_dict.keys())}")
                continue

        canonical_split = "valid" if actual_split_key in ["validation", "valid", "dev"] else actual_split_key
        ds = dataset_dict[actual_split_key]
        logger.info(f"Processing split {actual_split_key} ({len(ds)} items) -> {canonical_split}.jsonl")

        items: List[ManifestItem] = []
        for idx, sample in enumerate(ds):
            item = parse_sample_to_manifest_item(sample, canonical_split, source_revision, idx)
            if item is not None:
                items.append(item)

        dest_file = out_p / f"{canonical_split}.jsonl"
        save_manifest(items, dest_file)
        manifest_paths[canonical_split] = dest_file
        logger.info(f"Saved {len(items)} manifest items to {dest_file}")

    return manifest_paths
