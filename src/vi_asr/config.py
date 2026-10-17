"""Hierarchical YAML configuration loader and validator."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""
    pass


def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge source dictionary into target dictionary."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p.resolve()}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA256 hash of a configuration dict."""
    canonical = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate resolved configuration against core structural invariants."""
    required_sections = ["experiment", "data", "frontend", "model", "training", "paths"]
    for sec in required_sections:
        if sec not in cfg:
            raise ConfigValidationError(f"Missing required configuration section: '{sec}'")

    # Invariants for frontend
    fe = cfg.get("frontend", {})
    if fe.get("sample_rate") != 16000:
        raise ConfigValidationError(f"Frontend sample_rate must be 16000, got {fe.get('sample_rate')}")
    if fe.get("num_mel_bins") != 80:
        raise ConfigValidationError(f"Frontend num_mel_bins must be 80, got {fe.get('num_mel_bins')}")
    if fe.get("fbank_waveform_multiplier") != 32768.0:
        raise ConfigValidationError(f"fbank_waveform_multiplier must be 32768.0, got {fe.get('fbank_waveform_multiplier')}")

    # Invariants for model
    mdl = cfg.get("model", {})
    if mdl.get("encoder_layerdrop_override", 0.0) != 0.0:
        raise ConfigValidationError("encoder_layerdrop_override must be 0.0 for stable DDP training")

    # Invariants for training
    tr = cfg.get("training", {})
    if tr.get("loss_backward_divisor", 32.0) <= 0:
        raise ConfigValidationError("loss_backward_divisor must be positive")


def resolve_config(
    base_path: Union[str, Path],
    variant_path: Optional[Union[str, Path]] = None,
    profile_path: Optional[Union[str, Path]] = None,
    paths_path: Optional[Union[str, Path]] = None,
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load, merge hierarchy, validate, and return the resolved configuration dictionary.

    Order of precedence (later overrides earlier):
    1. base.yaml
    2. variant.yaml (e.g. b1, p1, p2)
    3. profile.yaml (e.g. t4x2, t4_single, cpu_smoke)
    4. paths.yaml (e.g. local, kaggle)
    5. CLI overrides (key=value)
    """
    config = load_yaml(base_path)

    if variant_path:
        variant_cfg = load_yaml(variant_path)
        _deep_merge(config, variant_cfg)

    if profile_path:
        profile_cfg = load_yaml(profile_path)
        _deep_merge(config, profile_cfg)

    if paths_path:
        paths_cfg = load_yaml(paths_path)
        _deep_merge(config, paths_cfg)

    if overrides:
        for override in overrides:
            if "=" not in override:
                raise ConfigValidationError(f"Invalid override format '{override}', expected 'key.subkey=value'")
            key_str, val_str = override.split("=", 1)
            # Parse value
            try:
                parsed_val = yaml.safe_load(val_str)
            except Exception:
                parsed_val = val_str

            # Traverse and set
            keys = key_str.strip().split(".")
            curr = config
            for k in keys[:-1]:
                if k not in curr or not isinstance(curr[k], dict):
                    curr[k] = {}
                curr = curr[k]
            curr[keys[-1]] = parsed_val

    validate_config(config)
    return config


def dump_resolved_config(config: Dict[str, Any], output_path: Union[str, Path]) -> str:
    """Save resolved config to a YAML file and return its SHA256 hash."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return compute_config_hash(config)
