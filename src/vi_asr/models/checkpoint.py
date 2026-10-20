"""Checkpoint loader and weight verification for ViP-VL ChunkFormer backbone (Gate G2)."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Compute deterministic SHA256 checksum of a file."""
    p = Path(filepath)
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_vip_vl_pretrained(
    model_or_encoder: nn.Module,
    checkpoint_path_or_repo: Union[str, Path] = "khanhld/vip-vl-base-vie",
    revision: Optional[str] = "4b259337d96450d57403acdee82572c587151e22",
    cache_dir: Optional[Union[str, Path]] = None,
    report_path: Optional[Union[str, Path]] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load pretrained ViP-VL ChunkFormer weights with strict backbone verification (Gate G2).

    Engineering Invariants (Gate G2):
    1. Zero missing encoder keys: Every parameter in ChunkFormerEncoder must have a valid key in checkpoint.
    2. Zero unexpected encoder keys: No leftover or mismatched encoder keys.
    3. Upstream SSL pretraining heads (quantizer, projection, MLM loss) are cleanly ignored.
    4. Exact shape verification: Checks each tensor shape before loading; fails fast on mismatch.
    5. Produces verifiable load_report.json.

    Args:
        model_or_encoder: ViPViCTCASR, ChunkFormerEncoderAdapter, or ChunkFormerEncoder instance
        checkpoint_path_or_repo: Path to .pt file or HF repo ID
        revision: Pinned git commit revision on Hugging Face
        cache_dir: Directory for cached weights
        report_path: Optional path to save load_report.json
        map_location: Device to load checkpoint weights onto

    Returns:
        report: Dict containing verification metadata and parameter statistics.
    """
    # 1. Resolve target ChunkFormerEncoder
    if hasattr(model_or_encoder, "encoder_adapter") and hasattr(model_or_encoder.encoder_adapter, "encoder"):
        target_encoder = model_or_encoder.encoder_adapter.encoder
    elif hasattr(model_or_encoder, "encoder"):
        target_encoder = model_or_encoder.encoder
    else:
        target_encoder = model_or_encoder

    # 2. Resolve checkpoint file
    p = Path(str(checkpoint_path_or_repo))
    if p.exists() and p.is_file():
        ckpt_file = p
    else:
        logger.info(
            "Downloading checkpoint from Hugging Face: %s (revision: %s)...",
            checkpoint_path_or_repo,
            revision,
        )
        from huggingface_hub import hf_hub_download

        ckpt_file = Path(
            hf_hub_download(
                repo_id=str(checkpoint_path_or_repo),
                filename="pytorch_model.pt",
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        )

    # 3. Compute checksum
    file_sha256 = compute_file_sha256(ckpt_file)
    logger.info("Loaded checkpoint %s (SHA256: %s)", ckpt_file, file_sha256[:16] + "...")

    # 4. Load state dict
    raw_state = torch.load(ckpt_file, map_location=map_location, weights_only=True)
    if isinstance(raw_state, dict):
        if "model" in raw_state and isinstance(raw_state["model"], dict):
            state_dict = raw_state["model"]
        elif "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
            state_dict = raw_state["state_dict"]
        else:
            state_dict = raw_state
    else:
        raise ValueError(f"Unrecognized checkpoint format in {ckpt_file}")

    # 5. Filter and strip keys
    encoder_keys: Dict[str, torch.Tensor] = {}
    ignored_keys: list[str] = []

    for k, v in state_dict.items():
        if k.startswith("encoder."):
            clean_k = k[len("encoder.") :]
            if "global_cmvn" in clean_k or "cmvn" in clean_k:
                ignored_keys.append(k)
            else:
                encoder_keys[clean_k] = v
        elif k.startswith("speech_encoder."):
            clean_k = k[len("speech_encoder.") :]
            if "global_cmvn" in clean_k or "cmvn" in clean_k:
                ignored_keys.append(k)
            else:
                encoder_keys[clean_k] = v
        elif k.startswith("quantizer.") or k.startswith("project_to_latent.") or k.startswith("final_proj."):
            ignored_keys.append(k)
        elif "mlm" in k or "loss" in k or "cmvn" in k:
            ignored_keys.append(k)
        else:
            # Check if k matches directly without prefix
            if k in target_encoder.state_dict():
                encoder_keys[k] = v
            else:
                ignored_keys.append(k)

    target_state = target_encoder.state_dict()
    target_keys_set = set(target_state.keys())
    clean_keys_set = set(encoder_keys.keys())

    missing_keys = sorted(list(target_keys_set - clean_keys_set))
    unexpected_keys = sorted(list(clean_keys_set - target_keys_set))

    # 6. Check tensor shapes
    shape_mismatches: list[str] = []
    matched_keys = sorted(list(target_keys_set.intersection(clean_keys_set)))
    for k in matched_keys:
        expected_shape = tuple(target_state[k].shape)
        actual_shape = tuple(encoder_keys[k].shape)
        if expected_shape != actual_shape:
            shape_mismatches.append(f"{k}: expected {expected_shape}, got {actual_shape}")

    if shape_mismatches:
        raise RuntimeError(
            f"Gate G2 Checkpoint Load Failure: Shape mismatch in {len(shape_mismatches)} keys:\n"
            + "\n".join(shape_mismatches[:5])
        )

    if missing_keys:
        raise RuntimeError(
            f"Gate G2 Checkpoint Load Failure: Missing {len(missing_keys)} encoder keys in checkpoint:\n"
            + "\n".join(missing_keys[:10])
        )

    if unexpected_keys:
        raise RuntimeError(
            f"Gate G2 Checkpoint Load Failure: Unexpected {len(unexpected_keys)} encoder keys in checkpoint:\n"
            + "\n".join(unexpected_keys[:10])
        )

    # 7. Strictly load weights
    target_encoder.load_state_dict(encoder_keys, strict=True)
    total_encoder_params = sum(p.numel() for p in target_encoder.parameters())
    logger.info(
        "Successfully loaded %d encoder keys (%d parameters) with strict=True.",
        len(matched_keys),
        total_encoder_params,
    )

    # 8. Produce Gate G2 load report
    report: Dict[str, Any] = {
        "gate": "G2",
        "checkpoint_file": str(ckpt_file),
        "checkpoint_sha256": file_sha256,
        "total_encoder_parameters": total_encoder_params,
        "matched_keys_count": len(matched_keys),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "ignored_upstream_keys_count": len(ignored_keys),
        "strict_backbone_passed": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if report_path:
        out_p = Path(report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved load report to %s", out_p)

    return report
