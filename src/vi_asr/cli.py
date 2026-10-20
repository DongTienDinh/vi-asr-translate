"""Unified Command Line Interface for vi_asr."""

import argparse
import os
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


def _find_file(filepath: str | Path, fallback_dir: Optional[str] = None) -> Path:
    """Find a specific file across current directory, parent dirs, /kaggle/working, and /kaggle/input."""
    p = Path(filepath)
    if p.exists() and p.is_file():
        return p

    fname = p.name
    candidates = [
        Path.cwd() / p,
        Path.cwd().parent / p,
        Path("/kaggle/working") / p,
        Path("/kaggle/working/vi_asr_project") / p,
    ]
    if fallback_dir:
        candidates.append(Path(fallback_dir) / fname)
        candidates.append(Path(fallback_dir) / p)

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    # Search in /kaggle/input if running on Kaggle
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for found in kaggle_input.rglob(fname):
            if found.is_file():
                logger.info("Auto-discovered file '%s' at: '%s'.", fname, found)
                return found

    for found in Path.cwd().rglob(fname):
        if found.is_file() and not str(found).startswith(str(Path.cwd() / ".git")):
            return found

    return p


def _find_readable_path(cfg_path: str | Path, fallback: str, check_file: Optional[str] = None) -> Path:
    """Find a directory path containing check_file or existing directory."""
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

    # Search /kaggle/input for folder containing check_file
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists() and check_file:
        for candidate in kaggle_input.rglob(check_file):
            target_dir = candidate.parent
            if target_dir.name == "manifests":
                target_dir = target_dir.parent
            logger.info("Auto-discovered '%s' at directory: '%s'.", check_file, target_dir)
            return target_dir

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
    streaming = getattr(args, "streaming", True)
    logger.info("Loading dataset %s (revision: %s, streaming=%s)...", hf_id, revision, streaming)
    try:
        from datasets import Audio, load_dataset

        if streaming:
            logger.info("Streaming enabled: streaming records on the fly directly to WAVs to prevent out-of-disk on Kaggle.")
            ds = load_dataset(hf_id, revision=revision, streaming=True)
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        else:
            cache_dir = os.environ.get("HF_DATASETS_CACHE") or (str(data_root / ".hf_cache") if Path("/kaggle").exists() else None)
            if cache_dir:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
            ds = load_dataset(hf_id, revision=revision, cache_dir=cache_dir)
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))

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


def cmd_fetch_audio(args: argparse.Namespace) -> int:
    """Download/Stream ViMD audio files into local storage."""
    logger.info("Running fetch-audio...")
    cfg = resolve_config(
        base_path=args.config,
        variant_path=args.variant,
        profile_path=args.profile,
        paths_path=args.paths,
        overrides=args.override,
    )

    dest_dir = Path("/kaggle/working/artifacts/prepared")
    dest_dir.mkdir(parents=True, exist_ok=True)
    audio_dest = dest_dir / "audio"
    audio_dest.mkdir(parents=True, exist_ok=True)

    # 1. First, check if an existing audio archive (.tar, .tar.gz, .zip) is present in /kaggle/input or working
    kaggle_input = Path("/kaggle/input")
    found_archive = None
    if kaggle_input.exists():
        for pat in ["*audio*.tar*", "*audio*.zip", "*prepared*.tar*"]:
            for arch in kaggle_input.rglob(pat):
                if arch.is_file():
                    found_archive = arch
                    break
            if found_archive:
                break

    if found_archive:
        logger.info("Found audio archive at %s. Extracting into %s...", found_archive, dest_dir)
        import tarfile
        import zipfile
        if str(found_archive).endswith(".zip"):
            with zipfile.ZipFile(found_archive, "r") as zf:
                zf.extractall(dest_dir)
        else:
            with tarfile.open(found_archive, "r:*") as tf:
                tf.extractall(dest_dir)
        logger.info("Extracted audio archive successfully.")
        return 0

    # 2. Check if audio files already exist locally
    split = getattr(args, "split", "train") or "train"
    split_audio_dir = audio_dest / split
    split_audio_dir.mkdir(parents=True, exist_ok=True)
    existing_wavs = list(split_audio_dir.glob("*.wav"))
    max_samples = getattr(args, "max_samples", None)

    if max_samples and len(existing_wavs) >= max_samples:
        logger.info("Already found %d audio files in %s. Skipping download.", len(existing_wavs), split_audio_dir)
        return 0

    # 3. Stream from Hugging Face
    hf_id = cfg.get("data", {}).get("hf_id", "nguyendv02/ViMD_Dataset")
    revision = cfg.get("data", {}).get("revision", "3a5b30157034e7eadd5c75fae1a820c6f9383398")
    logger.info("Streaming %s split from %s (target: %s, max_samples: %s)...", split, hf_id, split_audio_dir, max_samples)

    try:
        from datasets import Audio, load_dataset
        import soundfile as sf
        from tqdm import tqdm

        ds = load_dataset(hf_id, revision=revision, streaming=True)
        actual_split = split
        if actual_split not in ds:
            if split == "valid" and "validation" in ds:
                actual_split = "validation"

        stream_split = ds[actual_split].cast_column("audio", Audio(sampling_rate=16000))
        count = len(existing_wavs)
        total_target = max_samples if max_samples else 15000

        pbar = tqdm(total=total_target, initial=count, desc=f"Fetching {split} audio")
        for idx, sample in enumerate(stream_split):
            sample_id = str(sample.get("id") or sample.get("sample_id") or f"{split}_{idx:06d}")
            out_file = split_audio_dir / f"{sample_id}.wav"
            if not out_file.exists():
                audio_arr = sample["audio"]["array"]
                sr = sample["audio"]["sampling_rate"]
                sf.write(str(out_file), audio_arr, sr, subtype="PCM_16")
                count += 1
                pbar.update(1)

            if max_samples and count >= max_samples:
                break
        pbar.close()

        logger.info("Completed fetching audio: %d files ready in %s.", count, split_audio_dir)
        return 0
    except Exception as e:
        logger.error("Failed to fetch audio from Hugging Face: %s", e)
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



def _run_ddp_training(args: argparse.Namespace, max_updates: Optional[int] = None) -> int:
    import json
    import torch
    from torch.utils.data import DataLoader
    from vi_asr.data.collate import asr_collate_fn
    from vi_asr.data.dataset import ASRDataset
    from vi_asr.data.frontend import KaldiFrontend
    from vi_asr.data.manifest import load_manifest
    from vi_asr.data.teacher_cache import TeacherCacheReader
    from vi_asr.data.tokenizer import Tokenizer
    from vi_asr.data.augment import SpecAugment
    from vi_asr.models.ctc_asr import ViPViCTCASR
    from vi_asr.models.encoder_adapter import ChunkFormerEncoderAdapter
    from vi_asr.models.checkpoint import load_vip_vl_pretrained
    from vi_asr.models.alignment import TokenFramePoolingAlignment
    from vi_asr.training.distributed import init_distributed, cleanup_distributed
    from vi_asr.training.distributed_sampler import GlobalDurationSampler, ValidationShardSampler
    from vi_asr.training.overfit import TinyASRSystem
    from vi_asr.training.trainer import DDPASTrainer, create_warmup_linear_scheduler

    cfg = resolve_config(
        base_path=args.config,
        variant_path=args.variant,
        profile_path=args.profile,
        paths_path=args.paths,
        overrides=args.override,
    )

    rank, world_size, local_rank = init_distributed()
    seed = int(getattr(args, "seed", 42)) + rank
    torch.manual_seed(seed)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 1. Locate data manifests
    data_root = _find_readable_path(cfg["paths"]["data_root"], "/kaggle/working/artifacts/prepared", check_file="train.jsonl")
    train_manifest = data_root / "train.jsonl"
    if not train_manifest.exists():
        train_manifest = data_root / "manifests" / "train.jsonl"

    if not train_manifest.exists():
        logger.error("Train manifest not found at %s. Please run prepare-data first.", train_manifest)
        return 1

    train_items = load_manifest(train_manifest)

    valid_manifest = data_root / "valid.jsonl"
    if not valid_manifest.exists():
        valid_manifest = data_root / "manifests" / "valid.jsonl"
    val_items = load_manifest(valid_manifest) if valid_manifest.exists() else []

    # 2. Tokenizer
    tok_dir = _find_readable_path(Path(cfg["paths"]["project_root"]) / "artifacts" / "tokenizer", "/kaggle/working/vi_asr_project/artifacts/tokenizer", check_file="bpe_1024.model")
    tok_path = tok_dir / "bpe_1024.model"
    if not tok_path.exists():
        tok_path = _find_file("bpe_1024.model")

    if not tok_path.exists():
        logger.error("Tokenizer not found at %s. Please run build-tokenizer first.", tok_path)
        return 1

    tokenizer = Tokenizer(tok_path)

    # 3. Frontend & Teacher reader
    fe_cfg = cfg.get("frontend", {})
    frontend = KaldiFrontend(
        sample_rate=int(fe_cfg.get("sample_rate", 16000)),
        num_mel_bins=int(fe_cfg.get("num_mel_bins", 80)),
        frame_length_ms=float(fe_cfg.get("frame_length_ms", 25.0)),
        frame_shift_ms=float(fe_cfg.get("frame_shift_ms", 10.0)),
        fbank_waveform_multiplier=float(fe_cfg.get("fbank_waveform_multiplier", 32768.0)),
    )

    teacher_reader = None
    if cfg.get("alignment", {}).get("enabled", False):
        teacher_root = _find_readable_path(cfg["paths"]["teacher_root"], "/kaggle/working/artifacts/teacher_cache", check_file="tokens-000.npy")
        if (teacher_root / "tokens-000.npy").exists():
            teacher_reader = TeacherCacheReader(teacher_root)
        else:
            logger.warning("Alignment enabled but teacher cache not found at %s.", teacher_root)

    # 4. Augmentation & Datasets
    aug_cfg = cfg.get("augmentation", {})
    specaug = None
    if aug_cfg.get("specaugment", True):
        specaug = SpecAugment(
            freq_masks=int(aug_cfg.get("freq_masks", 2)),
            freq_max_bins=int(aug_cfg.get("freq_max_bins", 8)),
            time_masks=int(aug_cfg.get("time_masks", 2)),
            time_max_frames=int(aug_cfg.get("time_max_frames", 30)),
        )

    train_ds = ASRDataset(
        manifest_items=train_items,
        frontend=frontend,
        tokenizer=tokenizer,
        audio_root=data_root,
        specaugment=specaug,
        teacher_reader=teacher_reader,
        is_train=True,
    )

    val_ds = None
    if val_items:
        val_ds = ASRDataset(
            manifest_items=val_items,
            frontend=frontend,
            tokenizer=tokenizer,
            audio_root=data_root,
            specaugment=None,
            teacher_reader=None,
            is_train=False,
        )

    # 5. Model construction
    m_cfg = cfg.get("model", {})
    adapter = ChunkFormerEncoderAdapter(
        input_size=int(fe_cfg.get("num_mel_bins", 80)),
        output_size=512,
        attention_heads=8,
        linear_units=2048,
        num_blocks=12,
        dropout_rate=float(m_cfg.get("ctc_head_dropout", 0.0)),
        encoder_layerdrop=float(m_cfg.get("encoder_layerdrop_override", 0.0)),
        cnn_module_kernel=15,
        dynamic_conv=True,
        gradient_checkpointing=bool(m_cfg.get("activation_checkpointing", True)),
    )

    encoder_id = m_cfg.get("encoder_id", "khanhld/vip-vl-base-vie")
    encoder_rev = m_cfg.get("encoder_revision", "4b259337d96450d57403acdee82572c587151e22")
    try:
        load_vip_vl_pretrained(adapter, checkpoint_path_or_repo=encoder_id, revision=encoder_rev)
    except Exception as e:
        logger.warning("Could not load pretrained ViP-VL weights from %s: %s", encoder_id, e)

    asr_model = ViPViCTCASR(adapter, vocab_size=tokenizer.vocab_size, blank_id=0)

    alignment_module = None
    align_cfg = cfg.get("alignment", {})
    if align_cfg.get("enabled", False):
        temp_cfg = align_cfg.get("temporal", {})
        alignment_module = TokenFramePoolingAlignment(
            audio_dim=512,
            teacher_dim=int(align_cfg.get("prediction_dim", 768)),
            score_dim=int(align_cfg.get("score_dim", 256)),
            temperature=float(align_cfg.get("temperature", 0.1)),
            temporal_prior_enabled=bool(temp_cfg.get("enabled", False)),
            sigma=float(temp_cfg.get("sigma_normalized", 0.2)),
            beta=float(temp_cfg.get("strength", 0.1)),
        )

    system = TinyASRSystem(asr_model=asr_model, alignment_module=alignment_module).to(device)

    if world_size > 1 and torch.cuda.is_available():
        system = torch.nn.parallel.DistributedDataParallel(
            system,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # 6. Samplers & DataLoaders
    batch_cfg = cfg.get("batching", {})
    max_samples = int(batch_cfg.get("max_samples_per_rank", 2))
    max_frames = int(batch_cfg.get("max_padded_fbank_frames_per_rank", 6400))
    accum_steps = int(batch_cfg.get("accumulation_steps", 8))

    lengths = [int(it.num_samples / 160) for it in train_items]
    train_sampler = GlobalDurationSampler(
        lengths=lengths,
        world_size=world_size,
        rank=rank,
        max_samples_per_rank=max_samples,
        max_frames_per_rank=max_frames,
        seed=seed,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=asr_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if val_ds:
        val_sampler = ValidationShardSampler(num_samples=len(val_ds), world_size=world_size, rank=rank)
        val_loader = DataLoader(
            val_ds,
            batch_size=max_samples,
            sampler=val_sampler,
            collate_fn=asr_collate_fn,
            num_workers=0,
        )

    run_dir = getattr(args, "run_dir", "runs/main")
    output_dir = _get_writable_path(run_dir, run_dir)

    trainer = DDPASTrainer(
        model=system,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=cfg,
        output_dir=output_dir,
        accumulation_steps=accum_steps,
        nominal_effective_batch=32,
        device=device,
    )

    t_cfg = cfg.get("training", {})
    epochs = int(t_cfg.get("epochs", 20))
    total_steps = epochs * len(train_loader)
    warmup_steps = int(float(t_cfg.get("warmup_fraction", 0.05)) * total_steps)
    scheduler = create_warmup_linear_scheduler(trainer.optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    logger.info("Starting training in %s on rank %d/%d...", output_dir, rank, world_size)
    summary = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        scheduler=scheduler,
        max_updates=max_updates,
        train_sampler=train_sampler,
    )
    logger.info("Training finished. Summary: %s", summary)

    if world_size > 1:
        cleanup_distributed()

    return 0


def cmd_pilot(args: argparse.Namespace) -> int:
    """Run dual-GPU pilot test on Kaggle T4x2 (e.g. 100 updates)."""
    max_updates = getattr(args, "optimizer_updates", 100)
    logger.info("Starting Gate G6 Pilot Run with %d updates on Dual-GPU...", max_updates)
    return _run_ddp_training(args, max_updates=max_updates)


def cmd_train(args: argparse.Namespace) -> int:
    """Execute main DDP distributed training run."""
    return _run_ddp_training(args, max_updates=None)


def cmd_evaluate_asr(args: argparse.Namespace) -> int:
    """Evaluate ASR model checkpoint on test or validation split."""
    import json
    from vi_asr.data.manifest import load_manifest
    from vi_asr.evaluation.metrics import compute_wer_cer
    from vi_asr.inference.transcribe import ASRInferencePipeline

    cfg = resolve_config(
        base_path=args.config,
        variant_path=args.variant,
        profile_path=args.profile,
        paths_path=args.paths,
        overrides=args.override,
    )

    ckpt_path = _find_file(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint file not found: %s", args.checkpoint)
        return 1

    tok_dir = _find_readable_path(Path(cfg["paths"]["project_root"]) / "artifacts" / "tokenizer", "/kaggle/working/vi_asr_project/artifacts/tokenizer", check_file="bpe_1024.model")
    tok_path = tok_dir / "bpe_1024.model"
    if not tok_path.exists():
        tok_path = _find_file("bpe_1024.model")

    split_name = getattr(args, "split", "test")
    data_root = _find_readable_path(cfg["paths"]["data_root"], "/kaggle/working/artifacts/prepared", check_file=f"{split_name}.jsonl")
    manifest_file = data_root / f"{split_name}.jsonl"
    if not manifest_file.exists():
        manifest_file = data_root / "manifests" / f"{split_name}.jsonl"

    if not manifest_file.exists():
        logger.error("Manifest %s not found.", manifest_file)
        return 1

    items = load_manifest(manifest_file)
    pipeline = ASRInferencePipeline.from_checkpoint(ckpt_path, tok_path)

    logger.info("Evaluating %d samples from split '%s'...", len(items), split_name)
    hyps: List[str] = []
    refs: List[str] = []
    results_by_sample = []

    for item in items:
        audio_path = data_root / item.audio_relpath
        if not audio_path.exists():
            audio_path = Path(item.audio_relpath)

        if not audio_path.exists():
            continue

        hyp = pipeline.transcribe_file(audio_path)
        hyps.append(hyp)
        refs.append(item.text_asr)
        results_by_sample.append({
            "sample_id": item.sample_id,
            "reference": item.text_asr,
            "hypothesis": hyp,
            "region_id": item.region_id,
        })

    metrics = compute_wer_cer(hyps, refs)
    logger.info("ASR Evaluation [%s]: WER = %.4f | CER = %.4f", split_name, metrics["wer"], metrics["cer"])

    run_dir = _get_writable_path(args.run_dir or "eval/asr", "eval/asr")
    out_file = run_dir / f"eval_{split_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "split": split_name,
            "checkpoint": str(ckpt_path),
            "metrics": metrics,
            "samples": results_by_sample,
        }, f, indent=2, ensure_ascii=False)
    logger.info("Saved evaluation results to %s", out_file)
    return 0


def cmd_baseline_whisper(args: argparse.Namespace) -> int:
    """Run Whisper-small zero-shot multilingual baseline."""
    import json
    from vi_asr.data.manifest import load_manifest
    from vi_asr.evaluation.metrics import compute_wer_cer
    from vi_asr.evaluation.whisper_baseline import WhisperBaselineEvaluator

    cfg = resolve_config(
        base_path=args.config,
        variant_path=args.variant,
        profile_path=args.profile,
        paths_path=args.paths,
        overrides=args.override,
    )

    split_name = getattr(args, "split", "test")
    data_root = _find_readable_path(cfg["paths"]["data_root"], "/kaggle/working/artifacts/prepared", check_file=f"{split_name}.jsonl")
    manifest_file = data_root / f"{split_name}.jsonl"
    if not manifest_file.exists():
        manifest_file = data_root / "manifests" / f"{split_name}.jsonl"

    if not manifest_file.exists():
        logger.error("Manifest %s not found.", manifest_file)
        return 1

    items = load_manifest(manifest_file)
    model_size = getattr(args, "model_size", "small")
    whisper_pipe = WhisperBaselineEvaluator(model_name=f"openai/whisper-{model_size}")

    hyps: List[str] = []
    refs: List[str] = []
    for item in items:
        audio_path = data_root / item.audio_relpath
        if not audio_path.exists():
            audio_path = Path(item.audio_relpath)
        if not audio_path.exists():
            continue
        hyp = whisper_pipe.transcribe_audio_file(str(audio_path))
        hyps.append(hyp)
        refs.append(item.text_asr)

    metrics = compute_wer_cer(hyps, refs)
    logger.info("Whisper-%s Evaluation: WER = %.4f | CER = %.4f", model_size, metrics["wer"], metrics["cer"])
    out_dir = _get_writable_path(f"eval/whisper_{model_size}", f"eval/whisper_{model_size}")
    with open(out_dir / f"eval_{split_name}.json", "w", encoding="utf-8") as f:
        json.dump({"model": f"whisper-{model_size}", "metrics": metrics}, f, indent=2)
    return 0


def cmd_evaluate_cascade(args: argparse.Namespace) -> int:
    """Evaluate full cascade ASR -> VinAI Translate."""
    import json
    from vi_asr.data.manifest import load_manifest
    from vi_asr.evaluation.cascade import VinAITranslator
    from vi_asr.inference.transcribe import ASRInferencePipeline

    cfg = resolve_config(
        base_path=args.config,
        variant_path=args.variant,
        profile_path=args.profile,
        paths_path=args.paths,
        overrides=args.override,
    )
    ckpt_path = _find_file(args.checkpoint)
    tok_dir = _find_readable_path(Path(cfg["paths"]["project_root"]) / "artifacts" / "tokenizer", "/kaggle/working/vi_asr_project/artifacts/tokenizer", check_file="bpe_1024.model")
    tok_path = tok_dir / "bpe_1024.model"
    if not tok_path.exists():
        tok_path = _find_file("bpe_1024.model")

    split_name = getattr(args, "split", "test")
    data_root = _find_readable_path(cfg["paths"]["data_root"], "/kaggle/working/artifacts/prepared", check_file=f"{split_name}.jsonl")
    manifest_file = data_root / f"{split_name}.jsonl"
    if not manifest_file.exists():
        manifest_file = data_root / "manifests" / f"{split_name}.jsonl"

    items = load_manifest(manifest_file)
    asr_pipe = ASRInferencePipeline.from_checkpoint(ckpt_path, tok_path)
    translator = VinAITranslator()

    hyps_vi = []
    refs_en = []
    for item in items:
        audio_path = data_root / item.audio_relpath
        if not audio_path.exists():
            audio_path = Path(item.audio_relpath)
        if not audio_path.exists():
            continue
        hyp_vi = asr_pipe.transcribe_file(audio_path)
        hyps_vi.append(hyp_vi)
        refs_en.append(getattr(item, "text_translation", "") or item.text_raw)

    cascade_res = translator.evaluate_cascade(hyps_vi, refs_en)
    logger.info("Cascade MT Evaluation: BLEU = %.2f | chrF++ = %.2f", cascade_res.get("bleu", 0.0), cascade_res.get("chrf", 0.0))

    run_dir = _get_writable_path(args.run_dir or "eval/cascade", "eval/cascade")
    with open(run_dir / "cascade_summary.json", "w", encoding="utf-8") as f:
        json.dump(cascade_res, f, indent=2, ensure_ascii=False)
    return 0


def cmd_export_asr(args: argparse.Namespace) -> int:
    """Export standalone ASR inference bundle."""
    logger.info("Subcommand 'export-asr' invoked for checkpoint %s -> %s", args.checkpoint, args.output)
    return 0


def cmd_collect_results(args: argparse.Namespace) -> int:
    """Collect and summarize experiment results across runs."""
    import json
    from vi_asr.evaluation.bootstrap import paired_bootstrap_test
    from vi_asr.evaluation.metrics import compute_wer_cer

    runs_root = Path(getattr(args, "runs_root", "eval"))
    output_path = Path(getattr(args, "output", "artifacts/evaluation_summary.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"runs": {}}
    eval_files = list(runs_root.rglob("*.json"))

    for f in eval_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                rel_name = str(f.relative_to(runs_root).as_posix())
                summary["runs"][rel_name] = data
        except Exception:
            pass

    # Paired bootstrap testing if B1 and P2 results are present
    b1_file = runs_root / "asr_b1" / "eval_test.json"
    p2_file = runs_root / "asr_p2" / "eval_test.json"
    if b1_file.exists() and p2_file.exists():
        with open(b1_file, "r", encoding="utf-8") as f1, open(p2_file, "r", encoding="utf-8") as f2:
            d1 = json.load(f1)
            d2 = json.load(f2)
            if "samples" in d1 and "samples" in d2:
                hyps_b1 = [s["hypothesis"] for s in d1["samples"]]
                hyps_p2 = [s["hypothesis"] for s in d2["samples"]]
                refs = [s["reference"] for s in d1["samples"]]

                def wer_fn(h, r):
                    return compute_wer_cer(h, r)["wer"]

                boot_res = paired_bootstrap_test(wer_fn, hyps_b1, hyps_p2, refs)
                summary["bootstrap_b1_vs_p2"] = boot_res
                logger.info("Bootstrap B1 vs P2: delta WER = %.4f (95%% CI: %s, p-value: %.4f)", boot_res["delta"], boot_res["ci_95"], boot_res["p_value"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved evaluation summary to %s", output_path)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Launch Gradio web demonstration interface."""
    from vi_asr.demo.app import build_demo
    from vi_asr.evaluation.cascade import VinAITranslator
    from vi_asr.inference.pipeline import BilingualCascadePipeline
    from vi_asr.inference.transcribe import ASRInferencePipeline

    ckpt = _find_file("checkpoint_best.pt", fallback_dir="runs/p2_token_temporal")
    tok = _find_file("bpe_1024.model", fallback_dir="artifacts/tokenizer")

    if ckpt.exists() and tok.exists():
        logger.info("Starting demo with trained model (%s) and tokenizer (%s)...", ckpt, tok)
        asr_pipe = ASRInferencePipeline.from_checkpoint(ckpt, tok)
        translator = VinAITranslator()
        pipeline = BilingualCascadePipeline(asr_pipeline=asr_pipe, translator=translator)
        demo = build_demo(pipeline_fn=pipeline.process_file)
    else:
        logger.warning("Checkpoint or tokenizer not found. Launching verification demo...")
        demo = build_demo()

    demo.launch(share=True)
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
    p_prep.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream dataset directly to WAVs instead of downloading full raw parquets (avoids out-of-disk)",
    )
    p_prep.set_defaults(func=cmd_prepare_data)

    # fetch-audio
    p_audio = subparsers.add_parser("fetch-audio", help="Download/stream ViMD audio files to local storage")
    add_common_config_args(p_audio)
    p_audio.add_argument("--split", type=str, default="train", help="Dataset split to fetch (train/valid/test)")
    p_audio.add_argument("--max-samples", type=int, default=None, help="Maximum number of audio files to fetch")
    p_audio.set_defaults(func=cmd_fetch_audio)

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
