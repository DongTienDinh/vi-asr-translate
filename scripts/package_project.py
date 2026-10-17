"""Packaging script to create clean, reproducible vi_asr_project.zip for Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import List, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("package_project")

# Folders to include in the code archive
INCLUDE_DIRS = [
    "src",
    "configs",
    "tests",
    "scripts",
    "metadata",
    "docs",
    "notebooks",
    "third_party/chunkformer",
    "third_party/patches",
]

# Top-level files to include
INCLUDE_FILES = [
    "pyproject.toml",
    "requirements-kaggle.in",
    "requirements-extra-kaggle.lock.txt",
    "README.md",
    "AGENTS.md",
    "IMPLEMENTATION_STATUS.md",
    "THIRD_PARTY.md",
    ".gitignore",
]

# File/folder patterns to strictly exclude
EXCLUDE_PATTERNS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "runs",
    "artifacts",
    "checkpoints",
    ".cache",
    ".ipynb_checkpoints",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".npy",
    ".wav",
    ".flac",
    ".mp3",
    ".zip",
    ".tar",
    ".gz",
}


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def should_exclude(rel_path: Path) -> bool:
    parts = set(rel_path.parts)
    if parts.intersection(EXCLUDE_PATTERNS):
        return True
    if rel_path.suffix in EXCLUDE_EXTENSIONS:
        return True
    return False


def build_package(project_root: Path, output_zip: Path) -> Path:
    logger.info("Packaging project from %s into %s...", project_root, output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    archived_files: List[str] = []

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Add top-level files
        for fname in INCLUDE_FILES:
            fpath = project_root / fname
            if fpath.exists() and fpath.is_file():
                arcname = f"vi_asr_project/{fname}"
                zf.write(fpath, arcname)
                archived_files.append(arcname)
                logger.debug("Added file: %s", arcname)

        # 2. Add directories
        for dname in INCLUDE_DIRS:
            dpath = project_root / dname
            if not dpath.exists():
                logger.warning("Directory %s does not exist, skipping.", dpath)
                continue

            for root, dirs, files in os.walk(dpath):
                # Filter out excluded directory names in-place
                dirs[:] = [d for d in dirs if d not in EXCLUDE_PATTERNS]

                for file in files:
                    file_path = Path(root) / file
                    rel_to_root = file_path.relative_to(project_root)

                    if should_exclude(rel_to_root):
                        continue

                    arcname = f"vi_asr_project/{rel_to_root.as_posix()}"
                    zf.write(file_path, arcname)
                    archived_files.append(arcname)

    zip_sha256 = compute_file_sha256(output_zip)
    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)

    logger.info(
        "Successfully packaged %d files into %s (Size: %.2f MB, SHA256: %s)",
        len(archived_files),
        output_zip,
        zip_size_mb,
        zip_sha256,
    )

    # Save manifest
    manifest_path = output_zip.parent / "packaging_manifest.json"
    manifest_data = {
        "output_zip": output_zip.name,
        "sha256": zip_sha256,
        "total_files": len(archived_files),
        "size_mb": round(zip_size_mb, 2),
        "files": sorted(archived_files),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return output_zip


def main() -> int:
    parser = argparse.ArgumentParser(description="Package vi-asr project for Kaggle deployment")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/vi_asr_project.zip",
        help="Target output zip file path",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_zip = Path(args.output).resolve()
    build_package(project_root, output_zip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
