"""Build and activate a verified Kaggle handoff between preparation and training.

Prepared WAVs are packed into a small number of uncompressed ZIP shards.  Training
reads individual WAV members randomly and directly from the read-only Notebook 01
output under ``/kaggle/input``; no audio is extracted into ``/kaggle/working``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


HANDOFF_FILENAME = "vi_asr_handoff.json"
SHARD_INDEX_FILENAME = "audio_shards_index.json"
SHARD_DIRECTORY = "audio_shards"
SPLITS = ("train", "valid", "test")
TEACHER_FILES = (
    "meta.json",
    "index.jsonl",
    "tokens-000.npy",
    "token_ids-000.npy",
    "offsets-000.npy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def _manifest_path(prepared_root: Path, split: str) -> Path:
    direct = prepared_root / f"{split}.jsonl"
    nested = prepared_root / "manifests" / f"{split}.jsonl"
    if direct.is_file():
        return direct
    if nested.is_file():
        return nested
    raise FileNotFoundError(f"Missing {split} manifest under {prepared_root}")


def _expected_audio(rows: Sequence[Mapping[str, Any]], split: str) -> List[str]:
    expected: List[str] = []
    for row in rows:
        rel = str(row.get("audio_relpath", ""))
        if not rel:
            raise ValueError(f"Manifest split {split} contains an empty audio_relpath")
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"Unsafe audio_relpath in {split}: {rel}")
        expected.append(rel_path.as_posix())
    if len(expected) != len(set(expected)):
        raise ValueError(f"Duplicate audio_relpath values in {split} manifest")
    return expected


def validate_prepared_audio(prepared_root: Path) -> Dict[str, Dict[str, Any]]:
    """Require an exact one-to-one mapping between every manifest row and WAV."""
    result: Dict[str, Dict[str, Any]] = {}
    for split in SPLITS:
        manifest = _manifest_path(prepared_root, split)
        rows = _read_jsonl(manifest)
        expected = _expected_audio(rows, split)
        missing = [rel for rel in expected if not (prepared_root / rel).is_file()]
        actual = {
            path.relative_to(prepared_root).as_posix()
            for path in (prepared_root / "audio" / split).glob("*.wav")
            if path.is_file()
        }
        extras = sorted(actual.difference(expected))
        if missing or extras:
            raise RuntimeError(
                f"Audio coverage failed for {split}: manifest={len(rows)}, "
                f"wav={len(actual)}, missing={len(missing)}, extras={len(extras)}; "
                f"missing_examples={missing[:3]}, extra_examples={extras[:3]}"
            )
        result[split] = {
            "manifest": manifest.name,
            "samples": len(rows),
            "audio_files": len(actual),
            "manifest_sha256": _sha256(manifest),
        }
    return result


def validate_teacher_cache(teacher_root: Path, train_manifest: Path) -> Dict[str, Any]:
    missing_files = [name for name in TEACHER_FILES if not (teacher_root / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Incomplete teacher cache at {teacher_root}: {missing_files}")

    train_rows = _read_jsonl(train_manifest)
    manifest_ids = {str(row["sample_id"]) for row in train_rows}
    index_rows = _read_jsonl(teacher_root / "index.jsonl")
    index_ids = {str(row["sample_id"]) for row in index_rows}
    if len(index_rows) != len(index_ids):
        raise RuntimeError("Teacher index contains duplicate sample_id values")
    if manifest_ids != index_ids:
        raise RuntimeError(
            "Teacher cache/sample mismatch: "
            f"train={len(manifest_ids)}, cache={len(index_ids)}, "
            f"missing={len(manifest_ids - index_ids)}, extra={len(index_ids - manifest_ids)}"
        )

    with (teacher_root / "meta.json").open("r", encoding="utf-8") as stream:
        meta = json.load(stream)
    if int(meta.get("total_samples", -1)) != len(train_rows):
        raise RuntimeError(
            f"Teacher meta total_samples={meta.get('total_samples')} != train={len(train_rows)}"
        )
    return {
        "samples": len(index_rows),
        "total_tokens": int(meta.get("total_tokens", 0)),
        "hidden_size": int(meta.get("hidden_size", 0)),
        "dtype": meta.get("dtype"),
        "index_sha256": _sha256(teacher_root / "index.jsonl"),
        "meta_sha256": _sha256(teacher_root / "meta.json"),
    }


def _copy_tokenizer(tokenizer_root: Path, artifact_root: Path) -> Dict[str, Any]:
    model_candidates = sorted(tokenizer_root.glob("bpe_*.model"))
    if len(model_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one bpe_*.model in {tokenizer_root}, found {model_candidates}"
        )
    model = model_candidates[0]
    vocab = model.with_suffix(".vocab")
    if not vocab.is_file():
        raise FileNotFoundError(f"Tokenizer vocab not found: {vocab}")

    dest = artifact_root / "tokenizer"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, dest / model.name)
    shutil.copy2(vocab, dest / vocab.name)
    return {
        "model": model.name,
        "model_sha256": _sha256(dest / model.name),
        "vocab": vocab.name,
        "vocab_sha256": _sha256(dest / vocab.name),
    }


def _iter_expected_audio(prepared_root: Path) -> Iterable[Path]:
    for split in SPLITS:
        rows = _read_jsonl(_manifest_path(prepared_root, split))
        for rel in sorted(_expected_audio(rows, split)):
            yield prepared_root / rel


def build_audio_shards(
    prepared_root: Path,
    shard_size_bytes: int,
    remove_source_audio: bool,
) -> Dict[str, Any]:
    """Pack WAVs into random-access ZIP_STORED shards and write an exact index.

    With ``remove_source_audio=True`` each WAV is deleted only after it has been
    added successfully.  Because ZIP_STORED does not compress the payload, the
    total disk use stays close to one copy of the audio during conversion.
    """
    if shard_size_bytes <= 0:
        raise ValueError("shard_size_bytes must be positive")

    final_root = prepared_root / SHARD_DIRECTORY
    partial_root = prepared_root / f".{SHARD_DIRECTORY}.partial"
    index_path = prepared_root / SHARD_INDEX_FILENAME
    index_tmp = prepared_root / f"{SHARD_INDEX_FILENAME}.tmp"
    for stale in (partial_root,):
        if stale.exists():
            shutil.rmtree(stale)
    if final_root.exists():
        shutil.rmtree(final_root)
    for stale_file in (index_path, index_tmp):
        if stale_file.exists():
            stale_file.unlink()
    partial_root.mkdir(parents=True)

    members: Dict[str, Dict[str, Any]] = {}
    shard_records: List[Dict[str, Any]] = []
    current_archive: zipfile.ZipFile | None = None
    current_name = ""
    current_bytes = 0
    current_files = 0

    def close_current() -> None:
        nonlocal current_archive, current_name, current_bytes, current_files
        if current_archive is None:
            return
        current_archive.close()
        shard_records.append(
            {
                "path": f"{SHARD_DIRECTORY}/{current_name}",
                "files": current_files,
                "payload_bytes": current_bytes,
            }
        )
        current_archive = None
        current_name = ""
        current_bytes = 0
        current_files = 0

    try:
        for audio_path in _iter_expected_audio(prepared_root):
            size_bytes = audio_path.stat().st_size
            if current_archive is None or (
                current_files > 0 and current_bytes + size_bytes > shard_size_bytes
            ):
                close_current()
                current_name = f"audio-{len(shard_records):05d}.zip"
                current_archive = zipfile.ZipFile(
                    partial_root / current_name,
                    mode="w",
                    compression=zipfile.ZIP_STORED,
                    allowZip64=True,
                )

            rel = audio_path.relative_to(prepared_root).as_posix()
            current_archive.write(audio_path, arcname=rel, compress_type=zipfile.ZIP_STORED)
            info = current_archive.getinfo(rel)
            members[rel] = {
                "shard": f"{SHARD_DIRECTORY}/{current_name}",
                "member": rel,
                "size_bytes": int(info.file_size),
                "crc32": f"{info.CRC:08x}",
            }
            current_bytes += size_bytes
            current_files += 1
            if remove_source_audio:
                audio_path.unlink()
        close_current()

        os.replace(partial_root, final_root)
        for record in shard_records:
            shard_path = prepared_root / record["path"]
            record["size_bytes"] = shard_path.stat().st_size
            record["sha256"] = _sha256(shard_path)

        index = {
            "schema_version": 1,
            "storage": "zip_stored_shards",
            "total_files": len(members),
            "total_payload_bytes": sum(int(item["size_bytes"]) for item in members.values()),
            "shard_size_target_bytes": shard_size_bytes,
            "shards": shard_records,
            "members": members,
        }
        with index_tmp.open("w", encoding="utf-8") as stream:
            json.dump(index, stream, ensure_ascii=False, separators=(",", ":"))
        os.replace(index_tmp, index_path)
    except Exception:
        if current_archive is not None:
            current_archive.close()
        if partial_root.exists():
            shutil.rmtree(partial_root)
        if index_tmp.exists():
            index_tmp.unlink()
        raise

    if remove_source_audio:
        audio_root = prepared_root / "audio"
        if audio_root.exists():
            for directory in sorted(audio_root.rglob("*"), reverse=True):
                if directory.is_dir():
                    try:
                        directory.rmdir()
                    except OSError:
                        pass
            try:
                audio_root.rmdir()
            except OSError:
                pass

    return {
        "type": "zip_stored_shards",
        "index": SHARD_INDEX_FILENAME,
        "index_sha256": _sha256(index_path),
        "files": len(members),
        "payload_bytes": sum(int(item["size_bytes"]) for item in members.values()),
        "shards": len(shard_records),
        "source_audio_removed": remove_source_audio,
    }


def validate_audio_shards(prepared_root: Path, verify_hashes: bool = True) -> Dict[str, Any]:
    """Validate manifest coverage, ZIP central directories, sizes, CRC metadata and hashes."""
    index_path = prepared_root / SHARD_INDEX_FILENAME
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing audio shard index: {index_path}")
    with index_path.open("r", encoding="utf-8") as stream:
        index = json.load(stream)
    if index.get("schema_version") != 1 or index.get("storage") != "zip_stored_shards":
        raise RuntimeError(f"Unsupported audio shard index at {index_path}")

    expected: List[str] = []
    split_counts: Dict[str, int] = {}
    for split in SPLITS:
        rows = _read_jsonl(_manifest_path(prepared_root, split))
        split_expected = _expected_audio(rows, split)
        expected.extend(split_expected)
        split_counts[split] = len(split_expected)

    members = index.get("members", {})
    if set(expected) != set(members):
        missing = sorted(set(expected).difference(members))
        extras = sorted(set(members).difference(expected))
        raise RuntimeError(
            f"Shard index coverage mismatch: expected={len(expected)}, indexed={len(members)}, "
            f"missing={missing[:3]}, extras={extras[:3]}"
        )

    records_by_shard: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for rel, record in members.items():
        records_by_shard.setdefault(str(record["shard"]), {})[rel] = record

    for shard in index.get("shards", []):
        relative_path = str(shard["path"])
        rel = Path(relative_path)
        if rel.is_absolute() or ".." in rel.parts or rel.parts[0] != SHARD_DIRECTORY:
            raise RuntimeError(f"Unsafe shard path: {relative_path}")
        shard_path = prepared_root / rel
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing audio shard: {shard_path}")
        if shard_path.stat().st_size != int(shard["size_bytes"]):
            raise RuntimeError(f"Audio shard size mismatch: {shard_path}")
        if verify_hashes and _sha256(shard_path) != shard["sha256"]:
            raise RuntimeError(f"Audio shard SHA256 mismatch: {shard_path}")

        expected_in_shard = records_by_shard.get(relative_path, {})
        with zipfile.ZipFile(shard_path, mode="r") as archive:
            infos = {info.filename: info for info in archive.infolist() if not info.is_dir()}
        if set(infos) != set(expected_in_shard):
            raise RuntimeError(f"ZIP member coverage mismatch in {shard_path}")
        for member, record in expected_in_shard.items():
            info = infos[member]
            if info.compress_type != zipfile.ZIP_STORED:
                raise RuntimeError(f"Compressed member is not random-access optimized: {member}")
            if info.file_size != int(record["size_bytes"]) or f"{info.CRC:08x}" != record["crc32"]:
                raise RuntimeError(f"ZIP member metadata mismatch: {member}")

    return {
        "index": str(index_path),
        "index_sha256": _sha256(index_path),
        "files": len(members),
        "shards": len(index.get("shards", [])),
        "payload_bytes": int(index.get("total_payload_bytes", 0)),
        "split_counts": split_counts,
        "hashes_verified": verify_hashes,
    }


def build_handoff(args: argparse.Namespace) -> Dict[str, Any]:
    prepared_root = Path(args.prepared_root).resolve()
    teacher_root = Path(args.teacher_root).resolve()
    tokenizer_root = Path(args.tokenizer_root).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    audit_root = Path(args.audit_root).resolve()

    split_info = validate_prepared_audio(prepared_root)
    teacher_info = validate_teacher_cache(teacher_root, _manifest_path(prepared_root, "train"))
    tokenizer_info = _copy_tokenizer(tokenizer_root, artifact_root)
    if not (audit_root / "audit.json").is_file():
        raise FileNotFoundError(f"Audit report missing: {audit_root / 'audit.json'}")

    audio_storage = build_audio_shards(
        prepared_root=prepared_root,
        shard_size_bytes=int(getattr(args, "shard_size_mib", 512)) * 1024 * 1024,
        remove_source_audio=bool(args.remove_source_audio),
    )
    handoff = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_revision": args.dataset_revision,
        "splits": split_info,
        "audio_storage": audio_storage,
        "teacher_cache": teacher_info,
        "tokenizer": tokenizer_info,
        "paths": {
            "prepared": "prepared",
            "teacher_cache": "teacher_cache",
            "tokenizer": "tokenizer",
            "audit": audit_root.relative_to(artifact_root).as_posix(),
        },
        "audit_sha256": _sha256(audit_root / "audit.json"),
    }
    handoff_path = artifact_root / HANDOFF_FILENAME
    with handoff_path.open("w", encoding="utf-8") as stream:
        json.dump(handoff, stream, indent=2, ensure_ascii=False)
    return {"handoff": str(handoff_path), **handoff}


def discover_handoff(input_root: Path) -> Path:
    candidates = sorted(path for path in input_root.rglob(HANDOFF_FILENAME) if path.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No {HANDOFF_FILENAME} found under {input_root}. Attach the saved output of Notebook 01."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple Notebook 01 handoffs found; detach old versions and keep one: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def restore_handoff(args: argparse.Namespace) -> Dict[str, Any]:
    handoff_path = discover_handoff(Path(args.input_root))
    artifact_root = handoff_path.parent
    with handoff_path.open("r", encoding="utf-8") as stream:
        handoff = json.load(stream)
    if handoff.get("schema_version") != 2:
        raise RuntimeError(f"Unsupported handoff schema: {handoff.get('schema_version')}")

    working_root = Path(args.working_root).resolve()
    source_prepared = artifact_root / handoff["paths"]["prepared"]
    skip_hash = bool(
        getattr(args, "skip_shard_hash", getattr(args, "skip_archive_hash", False))
    )
    audio_info = validate_audio_shards(
        source_prepared,
        verify_hashes=not skip_hash,
    )
    if audio_info["index_sha256"] != handoff["audio_storage"]["index_sha256"]:
        raise RuntimeError(
            "Audio shard index SHA256 differs from the Notebook 01 handoff"
        )
    split_info = handoff["splits"]
    for split in SPLITS:
        actual_manifest_hash = _sha256(_manifest_path(source_prepared, split))
        if actual_manifest_hash != split_info[split]["manifest_sha256"]:
            raise RuntimeError(f"Manifest SHA256 mismatch for split {split}")

    teacher_root = artifact_root / handoff["paths"]["teacher_cache"]
    teacher_info = validate_teacher_cache(teacher_root, _manifest_path(source_prepared, "train"))
    for key in ("index_sha256", "meta_sha256"):
        if teacher_info[key] != handoff["teacher_cache"][key]:
            raise RuntimeError(f"Teacher cache {key} mismatch")

    source_tokenizer = artifact_root / handoff["paths"]["tokenizer"]
    for name_key, hash_key in (("model", "model_sha256"), ("vocab", "vocab_sha256")):
        tokenizer_file = source_tokenizer / handoff["tokenizer"][name_key]
        if not tokenizer_file.is_file() or _sha256(tokenizer_file) != handoff["tokenizer"][hash_key]:
            raise RuntimeError(f"Tokenizer integrity check failed: {tokenizer_file}")
    audit_file = artifact_root / handoff["paths"]["audit"] / "audit.json"
    if not audit_file.is_file() or _sha256(audit_file) != handoff["audit_sha256"]:
        raise RuntimeError(f"Audit report integrity check failed: {audit_file}")
    project_root = working_root / "vi_asr_project"
    target_tokenizer = project_root / "artifacts" / "tokenizer"
    target_tokenizer.mkdir(parents=True, exist_ok=True)
    for filename in (handoff["tokenizer"]["model"], handoff["tokenizer"]["vocab"]):
        shutil.copy2(source_tokenizer / filename, target_tokenizer / filename)

    runtime_paths = Path(args.runtime_paths)
    if not runtime_paths.is_absolute():
        runtime_paths = project_root / runtime_paths
    runtime_paths.parent.mkdir(parents=True, exist_ok=True)
    runtime_config = {
        "paths": {
            "project_root": str(project_root),
            "data_root": str(source_prepared),
            "teacher_root": str(teacher_root),
            "models_root": str(working_root / "artifacts" / "models"),
            "output_root": str(working_root / "runs"),
        }
    }
    with runtime_paths.open("w", encoding="utf-8") as stream:
        json.dump(runtime_config, stream, indent=2)

    return {
        "handoff": str(handoff_path),
        "runtime_paths": str(runtime_paths),
        "splits": split_info,
        "teacher_cache": teacher_info,
        "audio_storage": audio_info,
        "audio_shards_sha256_verified": not skip_hash,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Validate and package Notebook 01 outputs")
    build.add_argument("--prepared-root", required=True)
    build.add_argument("--teacher-root", required=True)
    build.add_argument("--tokenizer-root", required=True)
    build.add_argument("--audit-root", required=True)
    build.add_argument("--artifact-root", required=True)
    build.add_argument("--dataset-revision", required=True)
    build.add_argument("--remove-source-audio", action="store_true")
    build.add_argument("--shard-size-mib", type=int, default=512)
    build.set_defaults(func=build_handoff)

    restore = subparsers.add_parser("restore", help="Verify and mount Notebook 01 output in Notebook 02")
    restore.add_argument("--input-root", default="/kaggle/input")
    restore.add_argument("--working-root", default="/kaggle/working")
    restore.add_argument("--runtime-paths", default="configs/paths/kaggle_runtime.yaml")
    restore.add_argument(
        "--skip-shard-hash",
        "--skip-archive-hash",
        dest="skip_shard_hash",
        action="store_true",
        help="Skip full shard SHA256 checks",
    )
    restore.set_defaults(func=restore_handoff)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = args.func(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
