"""Dataset audit and leakage detection suite (Gate G1)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from vi_asr.data.manifest import ManifestItem, load_manifest
from vi_asr.data.tokenizer import Tokenizer


def compute_split_statistics(items: List[ManifestItem]) -> Dict[str, Any]:
    """Compute aggregate statistics for a single split."""
    total_samples = len(items)
    if total_samples == 0:
        return {
            "total_samples": 0,
            "total_duration_hours": 0.0,
            "min_duration_s": 0.0,
            "max_duration_s": 0.0,
            "mean_duration_s": 0.0,
            "regions": {},
        }

    durations = [item.duration for item in items]
    total_hours = sum(durations) / 3600.0

    # Regional breakdown
    region_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"samples": 0, "hours": 0.0})
    for item in items:
        reg_key = f"region_{item.region_id}" if item.region_id is not None else "unknown"
        region_stats[reg_key]["samples"] += 1
        region_stats[reg_key]["hours"] += item.duration / 3600.0

    return {
        "total_samples": total_samples,
        "total_duration_hours": round(total_hours, 3),
        "min_duration_s": round(min(durations), 3),
        "max_duration_s": round(max(durations), 3),
        "mean_duration_s": round(sum(durations) / total_samples, 3),
        "regions": dict(region_stats),
    }


def audit_splits(
    train_manifest: Union[str, Path, List[ManifestItem]],
    valid_manifest: Optional[Union[str, Path, List[ManifestItem]]] = None,
    test_manifest: Optional[Union[str, Path, List[ManifestItem]]] = None,
    tokenizer: Optional[Tokenizer] = None,
    report_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Perform comprehensive data audit and leakage detection.

    Checks:
    1. Total samples, hours, and regional distribution.
    2. Exact audio duplicate detection via audio_hash.
    3. Speaker overlap between train/valid/test splits.
    4. CTC validity inequality (T_out >= U + R).
    5. Exports audit.json, split_overlap.json, exclusions.jsonl if report_dir provided.
    """
    train_items = load_manifest(train_manifest) if isinstance(train_manifest, (str, Path)) else train_manifest
    valid_items = (
        load_manifest(valid_manifest)
        if isinstance(valid_manifest, (str, Path))
        else (valid_manifest or [])
    )
    test_items = (
        load_manifest(test_manifest)
        if isinstance(test_manifest, (str, Path))
        else (test_manifest or [])
    )

    all_splits = {
        "train": train_items,
        "valid": valid_items,
        "test": test_items,
    }

    audit_summary: Dict[str, Any] = {
        "splits": {},
        "audio_duplicates": 0,
        "speaker_overlap": {},
        "ctc_invalid_count": 0,
    }

    # 1. Split statistics
    for split_name, items in all_splits.items():
        audit_summary["splits"][split_name] = compute_split_statistics(items)

    # 2. Audio duplicate check across splits
    hash_to_samples: Dict[str, List[str]] = defaultdict(list)
    for split_name, items in all_splits.items():
        for item in items:
            if item.audio_hash:
                hash_to_samples[item.audio_hash].append(f"{split_name}:{item.sample_id}")

    duplicates = {h: locs for h, locs in hash_to_samples.items() if len(locs) > 1}
    audit_summary["audio_duplicates"] = len(duplicates)

    # 3. Speaker overlap check
    train_speakers: Set[str] = {item.speaker_id for item in train_items if item.speaker_id}
    valid_speakers: Set[str] = {item.speaker_id for item in valid_items if item.speaker_id}
    test_speakers: Set[str] = {item.speaker_id for item in test_items if item.speaker_id}

    speaker_overlap = {
        "train_valid_overlap": sorted(list(train_speakers.intersection(valid_speakers))),
        "train_test_overlap": sorted(list(train_speakers.intersection(test_speakers))),
        "valid_test_overlap": sorted(list(valid_speakers.intersection(test_speakers))),
    }
    audit_summary["speaker_overlap"] = {
        "train_valid_count": len(speaker_overlap["train_valid_overlap"]),
        "train_test_count": len(speaker_overlap["train_test_overlap"]),
        "valid_test_count": len(speaker_overlap["valid_test_overlap"]),
    }

    # 4. CTC contract check (T_out >= U + R)
    exclusions: List[Dict[str, Any]] = []
    if tokenizer is not None:
        for split_name, items in all_splits.items():
            for item in items:
                # 80 mel bins, 25ms frame, 10ms shift -> F = 1 + int((num_samples - 400) / 160)
                # Depthwise striding 8x subsampling: T_out ~= F / 8
                num_frames = max(1, 1 + int((item.num_samples - 400) / 160))
                # Upstream chunkformer subsampling 8x
                t_out = max(1, int(num_frames / 8))

                target_ids = tokenizer.encode(item.text_asr)
                is_valid = Tokenizer.check_ctc_validity(target_ids, t_out)
                if not is_valid:
                    audit_summary["ctc_invalid_count"] += 1
                    exclusions.append({
                        "sample_id": item.sample_id,
                        "split": split_name,
                        "duration": item.duration,
                        "t_out": t_out,
                        "u_target_len": len(target_ids),
                        "reason": "ctc_inequality_violation",
                    })

    # 5. Export reports
    if report_dir is not None:
        r_dir = Path(report_dir)
        r_dir.mkdir(parents=True, exist_ok=True)

        with open(r_dir / "audit.json", "w", encoding="utf-8") as f:
            json.dump(audit_summary, f, indent=2, ensure_ascii=False)

        with open(r_dir / "split_overlap.json", "w", encoding="utf-8") as f:
            json.dump(speaker_overlap, f, indent=2, ensure_ascii=False)

        with open(r_dir / "exclusions.jsonl", "w", encoding="utf-8") as f:
            for exc in exclusions:
                f.write(json.dumps(exc, ensure_ascii=False) + "\n")

    return audit_summary
