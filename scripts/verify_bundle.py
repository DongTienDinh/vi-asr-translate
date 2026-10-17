"""Bundle verification script inspecting vi_asr_project.zip (Section 22)."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("verify_bundle")

REQUIRED_ENTRIES = [
    "vi_asr_project/pyproject.toml",
    "vi_asr_project/README.md",
    "vi_asr_project/AGENTS.md",
    "vi_asr_project/src/vi_asr/__init__.py",
    "vi_asr_project/src/vi_asr/cli.py",
    "vi_asr_project/configs/base.yaml",
    "vi_asr_project/configs/variants/b1.yaml",
    "vi_asr_project/configs/profiles/t4x2.yaml",
    "vi_asr_project/third_party/chunkformer",
]

FORBIDDEN_PATTERNS = [
    ".git/",
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".npy",
    ".wav",
    "__pycache__",
]


def verify_bundle(zip_path: Path) -> bool:
    logger.info("Inspecting bundle: %s...", zip_path)
    if not zip_path.exists():
        logger.error("ZIP file %s does not exist!", zip_path)
        return False

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        logger.info("Total entries in ZIP: %d", len(namelist))

        # 1. Check required files
        for req in REQUIRED_ENTRIES:
            found = any(name.startswith(req) or name == req for name in namelist)
            if not found:
                logger.error("Missing required entry in ZIP: %s", req)
                return False

        # 2. Check forbidden entries
        for name in namelist:
            for pattern in FORBIDDEN_PATTERNS:
                if pattern in name:
                    logger.error("Forbidden file found in ZIP: %s (matches pattern '%s')", name, pattern)
                    return False

    # 3. Test extraction and CLI execution
    with tempfile.TemporaryDirectory() as tmpdir:
        extract_root = Path(tmpdir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)

        unpacked_root = extract_root / "vi_asr_project"
        logger.info("Extracted into temporary directory: %s", unpacked_root)

        # Run CLI verification
        cmd = [sys.executable, "-m", "vi_asr", "--help"]
        res = subprocess.run(cmd, cwd=str(unpacked_root), capture_output=True, text=True)
        if res.returncode != 0:
            logger.error("Failed to run vi-asr CLI in extracted bundle:\n%s", res.stderr)
            return False

        logger.info("Successfully verified vi-asr CLI inside extracted bundle.")

    logger.info("All bundle verification checks PASSED!")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify vi_asr_project.zip archive")
    parser.add_argument(
        "--bundle",
        type=str,
        default="artifacts/vi_asr_project.zip",
        help="Path to zip bundle to verify",
    )
    args = parser.parse_args()

    zip_path = Path(args.bundle).resolve()
    success = verify_bundle(zip_path)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
