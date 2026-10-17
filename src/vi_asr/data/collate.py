"""Collate function and batch preparation conforming to Master Section 13.3."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


def asr_collate_fn(batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate individual sample dicts into a padded batch dictionary.

    Batch Contract (Master Section 13.3):
        sample_ids: List[str]
        feats: float32 [B, Fmax, 80]
        feat_lengths: int64 [B]
        ctc_targets: int64 [B, Umax]
        target_lengths: int64 [B]
        teacher_states: float32 [B, Nmax, 768] (if present)
        teacher_lengths: int64 [B] (if present)
        region_ids: int64 [B] (if present)
    """
    B = len(batch_items)
    sample_ids = [item["sample_id"] for item in batch_items]

    # Feature padding
    feat_lengths = torch.tensor([item["feat_length"] for item in batch_items], dtype=torch.int64)
    max_F = int(feat_lengths.max().item())
    feat_dim = batch_items[0]["feats"].size(-1)

    padded_feats = torch.zeros(B, max_F, feat_dim, dtype=torch.float32)
    for i, item in enumerate(batch_items):
        f = item["feats"]
        padded_feats[i, : f.size(0)] = f

    # Target padding (pad with 0)
    target_lengths = torch.tensor([item["target_length"] for item in batch_items], dtype=torch.int64)
    max_U = int(target_lengths.max().item()) if B > 0 else 0
    padded_targets = torch.zeros(B, max(max_U, 1), dtype=torch.int64)
    for i, item in enumerate(batch_items):
        u = item["targets"]
        if u.numel() > 0:
            padded_targets[i, : u.size(0)] = u

    batch: Dict[str, Any] = {
        "sample_ids": sample_ids,
        "feats": padded_feats,
        "feat_lengths": feat_lengths,
        "ctc_targets": padded_targets,
        "target_lengths": target_lengths,
    }

    # Optional teacher states
    if "teacher_states" in batch_items[0]:
        t_lengths = torch.tensor(
            [item.get("teacher_length", 0) for item in batch_items],
            dtype=torch.int64,
        )
        max_N = int(t_lengths.max().item()) if B > 0 else 0
        if max_N > 0:
            teacher_dim = batch_items[0]["teacher_states"].size(-1)
            padded_teacher = torch.zeros(B, max_N, teacher_dim, dtype=torch.float32)
            for i, item in enumerate(batch_items):
                st = item.get("teacher_states")
                if st is not None and st.size(0) > 0:
                    padded_teacher[i, : st.size(0)] = st
            batch["teacher_states"] = padded_teacher
            batch["teacher_lengths"] = t_lengths
        else:
            batch["teacher_states"] = torch.zeros(B, 0, 768, dtype=torch.float32)
            batch["teacher_lengths"] = t_lengths

    # Optional region IDs
    if "region_id" in batch_items[0]:
        region_ids = torch.tensor(
            [item.get("region_id", 0) for item in batch_items],
            dtype=torch.int64,
        )
        batch["region_ids"] = region_ids

    return batch
