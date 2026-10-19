"""Unified Command Line Interface for vi_asr."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from vi_asr import __version__
from vi_asr.config import dump_resolved_config, resolve_config
from vi_asr.utils.logging import setup_logger

logger = setup_logger("vi_asr.cli")


def add_common_config_args(parser: argparse.ArgumentParser) -> None:
    """Add standard configuration arguments to subcommands."""
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base YAML configuration file",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Path to variant configuration file (e.g. configs/variants/b1.yaml, p1.yaml, p2.yaml)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Path to hardware profile configuration file (e.g. configs/profiles/t4x2.yaml)",
    )
    parser.add_argument(
        "--paths",
        type=str,
        default="configs/paths/local.yaml",
        help="Path to environment paths configuration file (e.g. configs/paths/kaggle.yaml)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override configuration setting in dot notation (e.g. --override training.epochs=10)",
    )


def _get_writable_path(cfg_path: str | Path, default_fallback: str) -> Path:
    p = Path(cfg_path)
    if str(p).startswith("/kaggle/input"):
        logger.warning(
            "Config path '%s' is in read-only /kaggle/input. Redirecting to writable path: '%s'.",
            p,
            default_fallback,
        )
        p = Path(default_fallback)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _find_readable_path(cfg_path: str | Path, fallback: str, check_file: Optional[str] = None) -> Path:
    p = Path(cfg_path)
    if check_file:
        if (p / check_file).exists() or (p / "manifests" / check_file).exists():
            return p
    elif p.exists():
        return p

    fallback_p = Path(fallback)
    if check_file:
        if (fallback_p / check_file).exists() or (fallback_p / "manifests" / check_file).exists():
            logger.info("Path '%s' not found. Falling back to: '%s'.", p, fallback_p)
            return fallback_p
    elif fallback_p.exists():
        logger.info("Path '%s' not found. Falling back to: '%s'.", p, fallback_p)
        return fallback_p

    return p


def cmd_audit(args: argparse.Namespace) -> int:
    """Audit environment, configuration, and data integrity."""
    logger.info("Running preflight configuration and environment audit...")
    cfg = resolve_config(
        base_path=args.config,
        variant_path=args.variant,
        profile_path=args.profile,
        paths_path=args.paths,
        overrides=args.override,
    )
    report_dir = Path(args.report_dir) if args.report_dir else Path(cfg["paths"]["output_root"]) / "audit"
    report_dir.mkdir(parents=True, exist_ok=True)

    config_hash = dump_resolved_config(cfg, report_dir / "resolved_config.yaml")
    logger.info("Resolved config SHA256: %s", config_hash)
    logger.info("Saved resolved configuration to %s", report_dir / "resolved_config.yaml")

    # Run data manifest audit if data manifests exist
    data_root = _find_readable_path(cfg["paths"]["data_root"], "/kaggle/working/artifacts/prepared", check_file="train.jsonl")
    train_manifest = data_root / "train.jsonl"
    if not train_manifest.exists():
        train_manifest = data_root / "manifests" / "train.jsonl"

    if train_manifest.exists():
        from vi_asr.data.audit import audit_splits
        valid_manifest = data_root / "valid.jsonl"
        if not valid_manifest.exists():
            valid_manifest = data_root / "manifests" / "valid.jsonl"
        test_manifest = data_root / "test.jsonl"
        if not test_manifest.exists():
            test_manifest = data_root / "manifests" / "test.jsonl"

        logger.info("Found manifests in %s. Running full audit and leakage checks...", data_root)
        audit_summary = audit_splits(
            train_manifest=train_manifest,
            valid_manifest=valid_manifest if valid_manifest.exists() else None,
            test_manifest=test_manifest if test_manifest.exists() else None,
            report_dir=report_dir,
        )
        logger.info("Audit completed. Total train samples: %d", audit_summary["splits"]["train"]["total_samples"])
    else:
        logger.info("Audit gate G0 passed. Data manifests not yet materialized in %s.", data_root)

    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    """Download/Stream ViMD and prepare canonical manifests and audio."""
    from vi_asr.data.prepare import prepare_manifests_from_hf

    logger.info("Running prepare-data...")
    cfg = resolve_config(
        base_path=args.config,
        variant_path=args.variant,
        profile_path=args.profile,
        paths_path=args.paths,
        overrides=args.override,
    )
    data_root = _get_writable_path(cfg["paths"]["data_root"], "/kaggle/working/artifacts/prepared")
    logger.info("Target data root: %s", data_root)

    # Check if manifests already exist in data_root or manifests/ subdirectory
    train_manifest = data_root / "train.jsonl"
    if not train_manifest.exists():
        train_manifest = data_root / "manifests" / "train.jsonl"

    if train_manifest.exists():
        logger.info("Manifests already exist at %s. Skipping download/generation.", train_manifest)
        return 0

    hf_id = cfg.get("data", {}).get("hf_id", "nguyendv02/ViMD_Dataset")
    revision = cfg.get("data", {}).get("revision", "3a5b30157034e7eadd5c75fae1a820c6f9383398")
    logger.info("Loading dataset %s (revision: %s)...", hf_id, revision)
    try:
        from datasets import load_dataset

        cache_dir = os.environ.get("HF_DATASETS_CACHE") or ("/tmp/huggingface/datasets" if Path("/kaggle").exists() else None)
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        ds = load_dataset(hf_id, revision=revision, cache_dir=cache_dir)
        manifest_paths = prepare_manifests_from_hf(ds, output_dir=data_root, source_revision=revision)
        logger.info("Successfully prepared manifests: %s", manifest_paths)
        return 0
    except Exception as e:
        logger.error("Failed to prepare dataset from Hugging Face: %s", e)
        logger.info(
            "If running offline or with pre-downloaded files, place train.jsonl, valid.jsonl, test.jsonl in %s",
            data_root,
        )
        return 1


def cmd_build_tokenizer(args: argparse.Namespace) -> int:
    """Build SentencePiece BPE tokenizer exclusively from train split."""
    from vi_asr.data.manifest import load_manifest
    from vi_asr.data.tokenizer import Tokenizer, train_bpe_tokenizer

    logger.info("Building SentencePiece BPE tokenizer exclusively from train split...")
    cfg = resolve_config(
        base_path=args.config,
        variant_path=args.variant,
        profile_path=args.profile,
        paths_path=args.paths,
        overrides=args.override,
    )
    data_root = _find_readable_path(cfg["paths"]["data_root"], "/kaggle/working/artifacts/prepared", check_file="train.jsonl")
    train_manifest = data_root / "train.jsonl"
    if not train_manifest.exists():
        train_manifest = data_root / "manifests" / "train.jsonl"

    if not train_manifest.exists():
        logger.error("Train manifest not found at %s. Please run prepare-data first.", train_manifest)
        return 1

    items = load_manifest(train_manifest)
    transcripts = [it.text_asr for it in items]

    tok_cfg = cfg.get("tokenizer", {})
    vocab_size = int(tok_cfg.get("vocab_size_target", 1024))
    char_cov = float(tok_cfg.get("character_coverage", 1.0))
    pad_id = int(tok_cfg.get("pad_id", 0))
    unk_id = int(tok_cfg.get("unk_id", 1))

    proj_root = _get_writable_path(cfg["paths"]["project_root"], "/kaggle/working/vi_asr_project")
    out_dir = proj_root / "artifacts" / "tokenizer"
    model_prefix = out_dir / f"bpe_{vocab_size}"

    model_path = train_bpe_tokenizer(
        transcripts=transcripts,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=char_cov,
        pad_id=pad_id,
        unk_id=unk_id,
    )
    logger.info("Trained tokenizer model saved to %s", model_path)

    tok = Tokenizer(model_path)
    logger.info("Verified tokenizer: vocab_size=%d, blank_id=%d, unk_id=%d", tok.vocab_size, tok.blank_id, tok.unk_id)
    return 0


def cmd_cache_teacher(args: argparse.Namespace) -> int:
    """Extract and cache frozen PhoBERT token representations."""
    from vi_asr.data.manifest import load_manifest
    from vi_asr.data.teacher_cache import PhoBERTTeacherExtractor, build_offline_teacher_cache

    logger.info("Extracting and caching frozen PhoBERT token representations...")
    cfg = resolve_config(
        base_path=args.config,
        variant_path=args.variant,
        profile_path=args.profile,
        paths_path=args.paths,
        overrides=args.override,
    )
    split_name = getattr(args, "split", "train")
    data_root = _find_readable_path(cfg["paths"]["data_root"], "/kaggle/working/artifacts/prepared", check_file=f"{split_name}.jsonl")
    manifest_file = data_root / f"{split_name}.jsonl"
    if not manifest_file.exists():
        manifest_file = data_root / "manifests" / f"{split_name}.jsonl"

    if not manifest_file.exists():
        logger.error("Manifest %s not found. Please run prepare-data first.", manifest_file)
        return 1

    items = load_manifest(manifest_file)
    logger.info("Loaded %d manifest items for split '%s'", len(items), split_name)

    t_cfg = cfg.get("teacher", {})
    model_id = t_cfg.get("id", "vinai/phobert-base")
    revision = t_cfg.get("revision", "01daacda68afe13d83023d16ec647239e344a1e6")
    device = getattr(args, "device", "cpu")

    extractor = PhoBERTTeacherExtractor(model_id=model_id, revision=revision, device=device)
    cache_dir = _get_writable_path(cfg["paths"]["teacher_root"], "/kaggle/working/artifacts/teacher_cache")
    build_offline_teacher_cache(items, cache_dir, extractor)
    logger.info("Teacher cache generation complete: saved to %s", cache_dir)
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Run smoke test (Gate G4) verifying learning convergence across variants."""
    from vi_asr.training.overfit import run_tiny_overfit

    logger.info("Running Gate G4 Smoke Test on variants B1, P1, P2...")
    for var in ("b1", "p1", "p2"):
        logger.info("Testing tiny overfit convergence on variant '%s'...", var)
        res = run_tiny_overfit(variant=var, num_samples=8, num_steps=25)
        logger.info(
            "Variant '%s': loss %.4f -> %.4f (converged: %s)",
            var,
            res["initial_loss"],
            res["final_loss"],
            res["converged"],
        )
        if not res["converged"]:
            logger.error("Smoke test failed for variant %s", var)
            return 1

    logger.info("Gate G4 Smoke Test PASSED across all variants.")
    return 0



def cmd_pilot(args: argparse.Namespace) -> int:
    """Run dual-GPU pilot test on Kaggle T4x2 (e.g. 100 updates)."""
    logger.info("Subcommand 'pilot' invoked (belongs to Step 9/Gate G6).")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Execute main DDP distributed training run."""
    logger.info("Subcommand 'train' invoked (belongs to Step 9).")
    return 0


def cmd_evaluate_asr(args: argparse.Namespace) -> int:
    """Evaluate ASR model checkpoint on test or validation split."""
    logger.info("Subcommand 'evaluate-asr' invoked (belongs to Step 10).")
    return 0


def cmd_baseline_whisper(args: argparse.Namespace) -> int:
    """Run Whisper-small zero-shot multilingual baseline."""
    logger.info("Subcommand 'baseline-whisper' invoked (belongs to Step 10).")
    return 0


def cmd_evaluate_cascade(args: argparse.Namespace) -> int:
    """Evaluate full cascade ASR -> VinAI Translate."""
    logger.info("Subcommand 'evaluate-cascade' invoked (belongs to Step 10).")
    return 0


def cmd_export_asr(args: argparse.Namespace) -> int:
    """Export standalone ASR inference bundle."""
    logger.info("Subcommand 'export-asr' invoked (belongs to Step 11).")
    return 0


def cmd_collect_results(args: argparse.Namespace) -> int:
    """Collect and summarize experiment results across runs."""
    logger.info("Subcommand 'collect-results' invoked.")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Launch Gradio web demonstration interface."""
    logger.info("Subcommand 'demo' invoked (belongs to Step 11).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="vi-asr",
        description=f"Vietnamese ASR to English Translation CLI (v{__version__})",
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # audit
    p_audit = subparsers.add_parser("audit", help="Audit environment and data configuration")
    add_common_config_args(p_audit)
    p_audit.add_argument("--report-dir", type=str, default=None, help="Directory to save audit report")
    p_audit.set_defaults(func=cmd_audit)

    # prepare-data
    p_prep = subparsers.add_parser("prepare-data", help="Prepare ViMD manifests and audio")
    add_common_config_args(p_prep)
    p_prep.set_defaults(func=cmd_prepare_data)

    # build-tokenizer
    p_tok = subparsers.add_parser("build-tokenizer", help="Train SentencePiece BPE tokenizer")
    add_common_config_args(p_tok)
    p_tok.set_defaults(func=cmd_build_tokenizer)

    # cache-teacher
    p_teach = subparsers.add_parser("cache-teacher", help="Precompute PhoBERT token cache")
    add_common_config_args(p_teach)
    p_teach.add_argument("--split", type=str, default="train", help="Dataset split to cache")
    p_teach.add_argument("--device", type=str, default="cuda:0", help="Device for PhoBERT inference")
    p_teach.set_defaults(func=cmd_cache_teacher)

    # smoke
    p_smoke = subparsers.add_parser("smoke", help="Run single-GPU smoke test")
    add_common_config_args(p_smoke)
    p_smoke.add_argument("--run-dir", type=str, default="runs/smoke", help="Output directory")
    p_smoke.set_defaults(func=cmd_smoke)

    # pilot
    p_pilot = subparsers.add_parser("pilot", help="Run 2-GPU pilot test")
    add_common_config_args(p_pilot)
    p_pilot.add_argument("--optimizer-updates", type=int, default=100, help="Number of optimizer updates")
    p_pilot.add_argument("--run-dir", type=str, default="runs/pilot", help="Output directory")
    p_pilot.set_defaults(func=cmd_pilot)

    # train
    p_train = subparsers.add_parser("train", help="Run main distributed training")
    add_common_config_args(p_train)
    p_train.add_argument("--seed", type=int, default=42, help="Random seed")
    p_train.add_argument("--run-dir", type=str, default="runs/main", help="Output directory")
    p_train.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p_train.set_defaults(func=cmd_train)

    # evaluate-asr
    p_eval_asr = subparsers.add_parser("evaluate-asr", help="Evaluate ASR checkpoint")
    add_common_config_args(p_eval_asr)
    p_eval_asr.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    p_eval_asr.add_argument("--split", type=str, default="valid", help="Split to evaluate (valid/test)")
    p_eval_asr.add_argument("--run-dir", type=str, default="eval/asr", help="Output evaluation directory")
    p_eval_asr.set_defaults(func=cmd_evaluate_asr)

    # baseline-whisper
    p_whisper = subparsers.add_parser("baseline-whisper", help="Run Whisper zero-shot baseline")
    add_common_config_args(p_whisper)
    p_whisper.add_argument("--model-size", type=str, default="small", help="Whisper model size")
    p_whisper.add_argument("--split", type=str, default="test", help="Split to evaluate")
    p_whisper.set_defaults(func=cmd_baseline_whisper)

    # evaluate-cascade
    p_casc = subparsers.add_parser("evaluate-cascade", help="Evaluate full ASR -> VinAI MT cascade")
    add_common_config_args(p_casc)
    p_casc.add_argument("--checkpoint", type=str, required=True, help="Path to ASR checkpoint")
    p_casc.add_argument("--run-dir", type=str, default="eval/cascade", help="Output evaluation directory")
    p_casc.set_defaults(func=cmd_evaluate_cascade)

    # export-asr
    p_export = subparsers.add_parser("export-asr", help="Export standalone ASR inference bundle")
    p_export.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    p_export.add_argument("--output", type=str, required=True, help="Output bundle directory")
    p_export.set_defaults(func=cmd_export_asr)

    # collect-results
    p_collect = subparsers.add_parser("collect-results", help="Aggregate results across runs")
    p_collect.add_argument("--runs-root", type=str, required=True, help="Root directory containing runs")
    p_collect.add_argument("--output", type=str, required=True, help="Output archive path")
    p_collect.set_defaults(func=cmd_collect_results)

    # demo
    p_demo = subparsers.add_parser("demo", help="Launch Gradio demo")
    p_demo.add_argument("--bundle-dir", type=str, default="artifacts/models/asr_bundle", help="Model bundle dir")
    p_demo.add_argument("--port", type=int, default=7860, help="Web port")
    p_demo.add_argument("--share", action="store_true", help="Create public Gradio link")
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 1

    if hasattr(args, "func"):
        return args.func(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
