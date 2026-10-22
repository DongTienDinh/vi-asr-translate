"""Regenerate the Kaggle data-preparation, training, and evaluation notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parent.parent


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def write_notebook(path: Path, cells: Iterable[dict]) -> None:
    metadata = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream).get("metadata", {})
    notebook_cells = list(cells)
    for index, cell in enumerate(notebook_cells):
        cell["id"] = f"cell-{index:02d}"
    notebook = {
        "cells": notebook_cells,
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(notebook, stream, indent=1, ensure_ascii=False)
        stream.write("\n")


COMMON_SETUP = r'''%cd /kaggle/working
%env HF_HOME=/kaggle/working/.cache/huggingface
%env HF_DATASETS_CACHE=/kaggle/working/.cache/huggingface/datasets
%env TRANSFORMERS_CACHE=/kaggle/working/.cache/huggingface/hub
%env TORCH_HOME=/kaggle/working/.cache/torch

import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

work_dir = Path("/kaggle/working/vi_asr_project")
preferred_zip = Path("/kaggle/input/vi-asr-code/vi_asr_project.zip")
preferred_project = Path("/kaggle/input/vi-asr-code/vi_asr_project")
zip_candidates = sorted(Path("/kaggle/input").rglob("vi_asr_project.zip"))
project_candidates = sorted(
    path.parent for path in Path("/kaggle/input").rglob("vi_asr_project/pyproject.toml")
)
dataset_projects = [path for path in project_candidates if "vi-asr-code" in path.parts]
dataset_zips = [path for path in zip_candidates if "vi-asr-code" in path.parts]
handoff_projects = [
    path for path in project_candidates
    if (path.parent / "artifacts" / "vi_asr_handoff.json").is_file()
]

source_path = None
source_kind = None
if (preferred_project / "pyproject.toml").is_file():
    # Kaggle thường bung ZIP đã upload thành thư mục read-only này.
    source_path, source_kind = preferred_project, "directory"
elif len(dataset_projects) == 1:
    # Kaggle image mới mount theo /kaggle/input/datasets/<owner>/<slug>/...
    source_path, source_kind = dataset_projects[0], "directory"
elif len(dataset_projects) > 1:
    raise RuntimeError(f"Có nhiều thư mục source trong dataset vi-asr-code: {dataset_projects}")
elif preferred_zip.is_file():
    source_path, source_kind = preferred_zip, "zip"
elif len(dataset_zips) == 1:
    source_path, source_kind = dataset_zips[0], "zip"
elif len(dataset_zips) > 1:
    raise RuntimeError(f"Có nhiều source ZIP trong dataset vi-asr-code: {dataset_zips}")
elif len(zip_candidates) == 1:
    source_path, source_kind = zip_candidates[0], "zip"
elif len(zip_candidates) > 1:
    raise RuntimeError(f"Có nhiều source ZIP, chỉ giữ một bản: {zip_candidates}")
elif len(handoff_projects) == 1:
    source_path, source_kind = handoff_projects[0], "directory"
elif not handoff_projects and len(project_candidates) == 1:
    source_path, source_kind = project_candidates[0], "directory"
else:
    raise FileNotFoundError(
        "Không xác định được duy nhất source project. Hãy attach dataset vi-asr-code "
        "hoặc output Notebook 01."
    )

if not (work_dir / "pyproject.toml").is_file():
    if source_kind == "zip":
        with zipfile.ZipFile(source_path, "r") as archive:
            archive.extractall("/kaggle/working")
        print("Đã giải nén source:", source_path)
    else:
        shutil.copytree(source_path, work_dir)
        print("Đã copy source:", source_path)

def sha256_source(path):
    path = Path(path)
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()

source_sha256 = sha256_source(source_path)
print({"source": str(source_path), "kind": source_kind, "sha256": source_sha256})

os.environ["PYTHONPATH"] = str(work_dir / "src")
if str(work_dir / "src") not in sys.path:
    sys.path.insert(0, str(work_dir / "src"))

%cd /kaggle/working/vi_asr_project
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.in"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

import vi_asr
print("vi_asr:", vi_asr.__file__)
'''


def build_notebook_01() -> None:
    cells: List[dict] = [
        markdown(
            """# Notebook 01 — Chuẩn bị dữ liệu ViMD cho Kaggle Input

Notebook stream revision ViMD đã pin từ Hugging Face đúng một lần, chuẩn hóa audio thành
WAV mono 16 kHz PCM-16, tạo tokenizer và PhoBERT cache. WAV sau đó được đóng vào một số
ZIP shard **không nén** để Notebook 02 đọc ngẫu nhiên trực tiếp từ `/kaggle/input`, không
giải nén sang `/kaggle/working`.

> Bật Internet và một GPU. Chỉ Save Version khi cell cuối báo `PASS`.
"""
        ),
        code(
            r'''# 1. Kiểm tra môi trường
!nvidia-smi
import shutil
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("Disk /kaggle/working:", shutil.disk_usage("/kaggle/working"))
assert torch.cuda.is_available(), "Cần GPU cho bước tạo PhoBERT teacher cache"
'''
        ),
        code("# 2. Nạp source và cài dependencies\n" + COMMON_SETUP),
        code(
            r'''# 3. Clean rebuild — chỉ xóa artifact thuộc pipeline trong /kaggle/working
import shutil
from pathlib import Path

targets = [
    Path("/kaggle/working/artifacts/prepared"),
    Path("/kaggle/working/artifacts/teacher_cache"),
    Path("/kaggle/working/artifacts/audit_report"),
    Path("/kaggle/working/artifacts/tokenizer"),
    Path("/kaggle/working/artifacts/prepared_audio.tar.gz"),
    Path("/kaggle/working/artifacts/vi_asr_handoff.json"),
    Path("/kaggle/working/vi_asr_project/artifacts/tokenizer"),
]
for target in targets:
    resolved = target.resolve()
    assert str(resolved).startswith("/kaggle/working/"), resolved
    if resolved.is_dir():
        shutil.rmtree(resolved)
    elif resolved.exists():
        resolved.unlink()
Path("/kaggle/working/artifacts").mkdir(parents=True, exist_ok=True)
print("Clean rebuild directory is ready.")
'''
        ),
        code(
            r'''# 4. Stream Hugging Face và materialize đầy đủ train/valid/test
import subprocess, sys
subprocess.run([
    sys.executable, "-m", "vi_asr", "prepare-data",
    "--config", "configs/base.yaml",
    "--paths", "configs/paths/kaggle_prepare.yaml",
], check=True)
'''
        ),
        code(
            r'''# 5. Xác nhận coverage WAV trước các bước tốn GPU
import json
from pathlib import Path

prepared = Path("/kaggle/working/artifacts/prepared")
split_counts = {}
for split in ("train", "valid", "test"):
    rows = [json.loads(line) for line in (prepared / f"{split}.jsonl").open(encoding="utf-8") if line.strip()]
    missing = [row["audio_relpath"] for row in rows if not (prepared / row["audio_relpath"]).is_file()]
    wav_count = len(list((prepared / "audio" / split).glob("*.wav")))
    split_counts[split] = len(rows)
    print(f"{split:5s}: manifest={len(rows):6d} | wav={wav_count:6d} | missing={len(missing)}")
    assert wav_count == len(rows) and not missing, f"Audio chưa đủ cho split {split}"
assert all(split_counts.values())
'''
        ),
        code(
            r'''# 6. Huấn luyện SentencePiece BPE chỉ từ train
subprocess.run([
    sys.executable, "-m", "vi_asr", "build-tokenizer",
    "--config", "configs/base.yaml",
    "--paths", "configs/paths/kaggle_prepare.yaml",
], check=True)
'''
        ),
        code(
            r'''# 7. Tạo PhoBERT content-token cache cho toàn bộ train
subprocess.run([
    sys.executable, "-m", "vi_asr", "cache-teacher",
    "--config", "configs/base.yaml",
    "--paths", "configs/paths/kaggle_prepare.yaml",
    "--split", "train", "--device", "cuda:0",
], check=True)
'''
        ),
        code(
            r'''# 8. Gate G1: leakage audit và CTC validity
subprocess.run([
    sys.executable, "-m", "vi_asr", "audit",
    "--config", "configs/base.yaml",
    "--paths", "configs/paths/kaggle_prepare.yaml",
    "--report-dir", "/kaggle/working/artifacts/audit_report",
], check=True)
'''
        ),
        code(
            r'''# 9. Đóng WAV thành ZIP_STORED shards (~512 MiB/shard)
# Mỗi WAV được xóa sau khi ghi thành công, nên không tạo thêm một bản audio 9.4 GiB.
subprocess.run([
    sys.executable, "scripts/kaggle_handoff.py", "build",
    "--prepared-root", "/kaggle/working/artifacts/prepared",
    "--teacher-root", "/kaggle/working/artifacts/teacher_cache",
    "--tokenizer-root", "/kaggle/working/vi_asr_project/artifacts/tokenizer",
    "--audit-root", "/kaggle/working/artifacts/audit_report",
    "--artifact-root", "/kaggle/working/artifacts",
    "--dataset-revision", "3a5b30157034e7eadd5c75fae1a820c6f9383398",
    "--shard-size-mib", "512",
    "--remove-source-audio",
], check=True)
'''
        ),
        code(
            r'''# 10. Final gate — chỉ Save Version khi cell này PASS
import json
import shutil
from pathlib import Path
from scripts.kaggle_handoff import validate_audio_shards

artifact_root = Path("/kaggle/working/artifacts")
handoff_path = artifact_root / "vi_asr_handoff.json"
handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
assert handoff["schema_version"] == 2

prepared = artifact_root / "prepared"
verified = validate_audio_shards(prepared, verify_hashes=True)
assert verified["files"] == 15023 + 1900 + 2026
assert not list(prepared.rglob("*.wav")), "WAV rời còn sót lại sẽ làm output quá nhiều file"

print(json.dumps({"handoff": handoff, "verified_audio": verified}, indent=2, ensure_ascii=False))
print("Working used:", shutil.disk_usage("/kaggle/working").used / 1024**3, "GiB")

cache_root = Path("/kaggle/working/.cache")
if cache_root.exists():
    shutil.rmtree(cache_root)
    print("Đã xóa download cache khỏi notebook output.")

# Giữ source dưới dạng một ZIP thay vì hàng nghìn file trong project tree.
project_copy = Path("/kaggle/working/vi_asr_project")
if project_copy.exists():
    subprocess.run([
        sys.executable, str(project_copy / "scripts" / "package_project.py"),
        "--output", "/kaggle/working/vi_asr_project.zip",
    ], check=True)
    os.chdir("/kaggle/working")
    shutil.rmtree(project_copy)
    print("Đã thay working project bằng vi_asr_project.zip.")
else:
    assert Path("/kaggle/working/vi_asr_project.zip").is_file()

print("\nPASS: Save Version rồi attach output này làm Input của Notebook 02.")
'''
        ),
    ]
    write_notebook(ROOT / "notebooks" / "01_prepare.ipynb", cells)


def build_notebook_02() -> None:
    cells: List[dict] = [
        markdown(
            """# Notebook 02 — Baseline B1 tối ưu pipeline trên Kaggle T4 ×2

Notebook xác minh output NB1 rồi đọc WAV trực tiếp trong ZIP shard tại `/kaggle/input`.
Không copy hoặc giải nén audio vào `/kaggle/working`. Notebook này chỉ chạy **B1 = ViP-VL
+ Linear CTC**, seed 42, fixed budget 20 epoch. Mỗi Save Version chạy một số epoch nguyên vẹn
rồi resume ở session kế tiếp. Protocol v2 dùng 1 worker/rank, prefetch, pinned memory,
microbatch 4 và accumulation 4 để giữ effective batch 32. Bật Internet để lấy checkpoint
và `global_cmvn` đúng revision đã pin.
"""
        ),
        code(
            r'''# 1. Kiểm tra đúng 2 GPU
!nvidia-smi
import torch
assert torch.cuda.device_count() == 2, f"NB2 yêu cầu 2 GPU, hiện có {torch.cuda.device_count()}"
print([torch.cuda.get_device_name(i) for i in range(2)])
'''
        ),
        code("# 2. Nạp source và cài dependencies\n" + COMMON_SETUP),
        code(
            r'''# 3. Xác minh và kích hoạt artifact NB1 — không giải nén audio
import subprocess, sys
assert "vi-asr-code" in source_path.parts, (
    f"NB2 từ chối source fallback ngoài dataset vi-asr-code: {source_path}"
)
subprocess.run([
    sys.executable, "scripts/kaggle_handoff.py", "restore",
    "--input-root", "/kaggle/input",
    "--working-root", "/kaggle/working",
    "--runtime-paths", "configs/paths/kaggle_runtime.yaml",
], check=True)
'''
        ),
        code(
            r'''# 4. Preflight: data_root phải nằm trong Input và mọi shard phải hợp lệ
import json
from pathlib import Path
from scripts.kaggle_handoff import validate_audio_shards

runtime = json.loads(Path("configs/paths/kaggle_runtime.yaml").read_text(encoding="utf-8"))
data_root = Path(runtime["paths"]["data_root"])
assert str(data_root).startswith("/kaggle/input/"), data_root
assert not Path("/kaggle/working/artifacts/prepared/audio").exists()

verified = validate_audio_shards(data_root, verify_hashes=False)  # SHA đã kiểm tra ở cell 3
print(json.dumps(verified, indent=2))
assert verified["split_counts"] == {"train": 15023, "valid": 1900, "test": 2026}
assert Path("artifacts/tokenizer/bpe_1024.model").is_file()

subprocess.run([
    sys.executable, "-m", "vi_asr", "audit",
    "--config", "configs/base.yaml",
    "--paths", "configs/paths/kaggle_runtime.yaml",
    "--report-dir", "/kaggle/working/preflight_audit",
], check=True)
'''
        ),
        code(
            r'''# 5. Smoke tùy chọn — chỉ cần bật ở session B1 đầu tiên
RUN_SMOKE = False
if RUN_SMOKE:
    subprocess.run([
        sys.executable, "-m", "vi_asr", "smoke",
        "--config", "configs/base.yaml",
        "--paths", "configs/paths/kaggle_runtime.yaml",
    ], check=True)
else:
    print("Bỏ qua smoke đã PASS trước đó; bắt đầu đường chạy B1 cố định.")
'''
        ),
        code(
            r'''# 6. Protocol B1 đã khóa — không đổi giữa các session
RUN_NAME = "b1_ctc_v2"
SEED = 42
TOTAL_EPOCHS = 20
EPOCHS_THIS_SESSION = 3         # giảm còn 1–2 nếu gần giới hạn thời gian Kaggle
RESUME_CHECKPOINT = "auto"      # auto | None | đường dẫn checkpoint cụ thể

assert EPOCHS_THIS_SESSION > 0
print({
    "run": RUN_NAME,
    "variant": "configs/variants/b1.yaml",
    "seed": SEED,
    "total_epochs": TOTAL_EPOCHS,
    "epochs_this_session": EPOCHS_THIS_SESSION,
    "resume": RESUME_CHECKPOINT,
})
'''
        ),
        code(
            r'''# 7. Chạy/resume baseline B1; subprocess lỗi sẽ dừng notebook ngay
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import torch

common = [
    "--config", "configs/base.yaml",
    "--profile", "configs/profiles/t4x2_fast.yaml",
    "--paths", "configs/paths/kaggle_runtime.yaml",
]

command = [
    "torchrun", "--standalone", "--nproc_per_node=2", "-m", "vi_asr", "train",
    *common,
    "--variant", "configs/variants/b1.yaml",
    "--run-dir", f"/kaggle/working/runs/{RUN_NAME}",
    "--seed", str(SEED),
    "--epochs-this-session", str(EPOCHS_THIS_SESSION),
]

resume_path = None
if RESUME_CHECKPOINT == "auto":
    candidates = sorted(
        path for path in Path("/kaggle/input").rglob("checkpoint_latest.pt")
        if path.parent.name == RUN_NAME
    )
    if len(candidates) > 1:
        raise RuntimeError(f"Chỉ attach output session B1 mới nhất; đang có: {candidates}")
    resume_path = candidates[0] if candidates else None
elif RESUME_CHECKPOINT:
    resume_path = Path(RESUME_CHECKPOINT)

start_epoch = 1
if resume_path is not None:
    assert resume_path.is_file(), resume_path
    resume_meta = torch.load(resume_path, map_location="cpu", weights_only=False)
    assert resume_meta.get("config", {}).get("experiment", {}).get("name") == "b1_ctc"
    assert resume_meta["config"]["experiment"].get("protocol_version") == "b1_repro_v2"
    assert int(resume_meta["config"]["experiment"]["seed"]) == SEED
    start_epoch = int(resume_meta["epoch"]) + 1
    assert start_epoch <= TOTAL_EPOCHS, "B1 đã đủ 20 epoch; chuyển sang Notebook 03."
    command += ["--resume", str(resume_path)]
    print(f"Resume epoch {start_epoch}/{TOTAL_EPOCHS} từ:", resume_path)
else:
    print("Bắt đầu B1 mới từ pretrained ViP-VL, epoch 1/20.")

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

protocol = {
    "run": RUN_NAME,
    "variant": "b1",
    "protocol_version": "b1_repro_v2",
    "profile": "configs/profiles/t4x2_fast.yaml",
    "seed": SEED,
    "total_epochs": TOTAL_EPOCHS,
    "session_start_epoch": start_epoch,
    "epochs_this_session": EPOCHS_THIS_SESSION,
    "source_path": str(source_path),
    "source_kind": source_kind,
    "source_sha256": source_sha256,
    "data_root": str(data_root),
    "audio_files": verified["files"],
    "split_counts": verified["split_counts"],
}
protocol_path = Path(f"/kaggle/working/runs/{RUN_NAME}/notebook_protocol.json")
protocol_path.parent.mkdir(parents=True, exist_ok=True)
protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(protocol, indent=2, ensure_ascii=False))

free_gib = shutil.disk_usage("/kaggle/working").free / 1024**3
print("Free working before training:", round(free_gib, 2), "GiB")
assert free_gib >= 8.0, "Working còn dưới 8 GiB; hãy xóa output/cache không cần thiết"
print("Command:", " ".join(command), flush=True)
training_env = os.environ.copy()
training_env["PYTHONUNBUFFERED"] = "1"
subprocess.run(command, check=True, env=training_env)
'''
        ),
        code(
            r'''# 8. Kiểm tra output và dọn cache trước Save Version
import os
import shutil
from pathlib import Path

run_root = Path("/kaggle/working/runs")
for checkpoint in sorted(run_root.rglob("*.pt")):
    stat = checkpoint.stat()
    print(f"{checkpoint}: {stat.st_size / 1024**3:.2f} GiB | inode={getattr(stat, 'st_ino', 'n/a')}")

cache_root = Path("/kaggle/working/.cache")
if cache_root.exists():
    shutil.rmtree(cache_root)

# Source luôn lấy lại từ NB1 input ở session kế tiếp; không lưu bản giải nén trong output NB2.
os.chdir("/kaggle/working")
project_copy = Path("/kaggle/working/vi_asr_project")
if project_copy.exists():
    shutil.rmtree(project_copy)
print("Working free before Save Version:", round(shutil.disk_usage("/kaggle/working").free / 1024**3, 2), "GiB")

latest = run_root / RUN_NAME / "checkpoint_latest.pt"
best = run_root / RUN_NAME / "checkpoint_best.pt"
assert latest.is_file() and best.is_file(), (latest, best)
meta = torch.load(latest, map_location="cpu", weights_only=False)
finished_epoch = int(meta["epoch"])
assert 1 <= finished_epoch <= TOTAL_EPOCHS
print(f"B1 đã hoàn tất epoch {finished_epoch}/{TOTAL_EPOCHS}; best epoch={meta['metrics']['best_epoch']}")
if finished_epoch < TOTAL_EPOCHS:
    print("PASS SESSION: Save Version, session sau chỉ attach output B1 mới nhất và resume auto.")
else:
    print("B1 TRAINING COMPLETE: Save Version rồi chuyển sang Notebook 03 đánh giá.")
'''
        ),
    ]
    write_notebook(ROOT / "notebooks" / "02_train_t4x2.ipynb", cells)


def build_notebook_03() -> None:
    cells: List[dict] = [
        markdown(
            """# Notebook 03 — Tái hiện và đánh giá baseline B1

Notebook chỉ nhận run `b1_ctc_v2` đã hoàn thành đủ 20 epoch. `checkpoint_best.pt` được chọn bằng
validation WER (tie-break CER, rồi epoch sớm hơn). Audio test được đọc trực tiếp từ ZIP shard
của Notebook 01; thiếu một mẫu sẽ dừng. Output báo corpus WER/CER tổng và theo ba miền.
"""
        ),
        code("# 1. Nạp source và cài dependencies\n" + COMMON_SETUP),
        code(
            r'''# 2. Kích hoạt artifact NB1 — không giải nén audio vào /kaggle/working
import subprocess
import sys

subprocess.run([
    sys.executable, "scripts/kaggle_handoff.py", "restore",
    "--input-root", "/kaggle/input",
    "--working-root", "/kaggle/working",
    "--runtime-paths", "configs/paths/kaggle_runtime.yaml",
], check=True)
'''
        ),
        code(
            r'''# 3. Xác minh run B1 đã hoàn thành đủ fixed budget 20 epoch
import hashlib
import json
from pathlib import Path
import torch

def find_one(filename):
    candidates = sorted(
        path for path in Path("/kaggle/input").rglob(filename)
        if path.parent.name == "b1_ctc_v2"
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Cần đúng 1 {filename} cho b1_ctc_v2, tìm thấy {len(candidates)}: {candidates}"
        )
    return candidates[0]

BEST_CHECKPOINT = find_one("checkpoint_best.pt")
LATEST_CHECKPOINT = find_one("checkpoint_latest.pt")
latest = torch.load(LATEST_CHECKPOINT, map_location="cpu", weights_only=False)
best = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
assert int(latest["epoch"]) == 20, f"B1 mới chạy đến epoch {latest['epoch']}/20"
assert latest["config"]["experiment"]["name"] == "b1_ctc"
assert latest["config"]["experiment"]["protocol_version"] == "b1_repro_v2"
assert int(latest["config"]["experiment"]["seed"]) == 42
assert best["metrics"]["best_epoch"] == best["epoch"]
print({
    "latest_epoch": latest["epoch"],
    "best_epoch": best["epoch"],
    "best_valid_wer": best["metrics"]["best_wer"],
    "best_valid_cer": best["metrics"]["best_cer"],
    "best_checkpoint": str(BEST_CHECKPOINT),
})
'''
        ),
        code(
            r'''# 4. Decode frozen test split đúng một lần bằng best validation checkpoint
subprocess.run([
    sys.executable, "-m", "vi_asr", "evaluate-asr",
    "--config", "configs/base.yaml",
    "--paths", "configs/paths/kaggle_runtime.yaml",
    "--checkpoint", str(BEST_CHECKPOINT),
    "--split", "test",
    "--run-dir", "/kaggle/working/eval/asr_b1",
], check=True)
'''
        ),
        code(
            r'''# 5. Gate cuối: coverage 100%, provenance và báo cáo tái hiện B1
eval_path = Path("/kaggle/working/eval/asr_b1/eval_test.json")
result = json.loads(eval_path.read_text(encoding="utf-8"))
assert result["coverage"] == {"decoded": 2026, "manifest": 2026}
assert result["checkpoint_epoch"] == best["epoch"]
assert result["checkpoint_selection_metrics"]["best_epoch"] == best["epoch"]
assert set(result["metrics_by_region"]) == {"north", "central", "south"}

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

report = {
    "status": "completed",
    "baseline": "B1: ViP-VL + Linear CTC",
    "seed": 42,
    "fixed_budget_epochs": 20,
    "selected_epoch": best["epoch"],
    "selection_metric": "validation WER; tie-break CER then earliest epoch",
    "test_samples": result["coverage"]["decoded"],
    "test_metrics": result["metrics"],
    "metrics_by_region": result["metrics_by_region"],
    "checkpoint_sha256": result["checkpoint_sha256"],
    "training_config_sha256": result["training_config_sha256"],
    "source_sha256": source_sha256,
}
report_path = Path("/kaggle/working/artifacts/b1_reproduction_report.json")
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
print("PASS B1 REPRODUCTION:", report_path)
'''
        ),
    ]
    write_notebook(ROOT / "notebooks" / "03_evaluate.ipynb", cells)


if __name__ == "__main__":
    build_notebook_01()
    build_notebook_02()
    build_notebook_03()
    print("Regenerated notebooks/01_prepare.ipynb through notebooks/03_evaluate.ipynb")
