"""Update notebook cells to safely cd to /kaggle/working, set writable HF cache dirs, and auto-discover files."""

import json
from pathlib import Path

COMMON_INIT_SNIPPET = [
    "%cd /kaggle/working\n",
    "# Cấu hình thư mục cache ghi vào /kaggle/working/.cache để không làm đầy phân vùng root /tmp (~5GB)\n",
    "%env HF_HOME=/kaggle/working/.cache/huggingface\n",
    "%env HF_DATASETS_CACHE=/kaggle/working/.cache/huggingface/datasets\n",
    "%env TRANSFORMERS_CACHE=/kaggle/working/.cache/huggingface/hub\n",
    "%env TORCH_HOME=/kaggle/working/.cache/torch\n",
    "%env PYTHONPATH=/kaggle/working/vi_asr_project/src\n",
    "import os, sys, zipfile\n",
    "from pathlib import Path\n",
    "\n",
    "work_dir = Path(\"/kaggle/working/vi_asr_project\")\n",
    "zip_source = Path(\"/kaggle/input/vi-asr-code/vi_asr_project.zip\")\n",
    "if not zip_source.exists():\n",
    "    zip_source = Path(\"vi_asr_project.zip\")\n",
    "\n",
    "if not (work_dir / \"pyproject.toml\").exists():\n",
    "    if zip_source.exists():\n",
    "        with zipfile.ZipFile(zip_source, 'r') as zf:\n",
    "            zf.extractall(\"/kaggle/working\")\n",
    "        print(\"Unpacked vi_asr_project from zip successfully.\")\n",
    "    else:\n",
    "        print(\"Cloning repository from GitHub...\")\n",
    "        !rm -rf /kaggle/working/vi_asr_project\n",
    "        !git clone --recurse-submodules https://github.com/DongTienDinh/vi-asr-translate.git /kaggle/working/vi_asr_project\n",
    "\n",
    "%cd /kaggle/working/vi_asr_project\n",
    "!pip install -q -r requirements-kaggle.in\n",
    "!pip install -q -e .\n",
    "\n",
    "src_dir = str(work_dir / \"src\")\n",
    "if src_dir not in sys.path:\n",
    "    sys.path.insert(0, src_dir)\n",
    "\n",
    "import vi_asr\n",
    "print('vi_asr successfully loaded from:', vi_asr.__file__)\n",
]

def patch_01():
    p = Path("notebooks/01_prepare.ipynb")
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    nb["cells"][2]["source"] = [
        "# 2. Nạp mã nguồn vi_asr_project (từ Zip hoặc Git Clone) & Cài đặt dependencies\n",
        *COMMON_INIT_SNIPPET,
    ]
    nb["cells"][3]["source"] = [
        "# 3. Chuẩn bị manifests dữ liệu từ Hugging Face (mặc định streaming để tránh tràn đĩa)\n",
        "!python -m vi_asr prepare-data \\\n",
        "    --config configs/base.yaml \\\n",
        "    --paths configs/paths/kaggle_prepare.yaml\n"
    ]
    nb["cells"][4]["source"] = [
        "# 4. Huấn luyện SentencePiece BPE Tokenizer (Train-only)\n",
        "!python -m vi_asr build-tokenizer \\\n",
        "    --config configs/base.yaml \\\n",
        "    --paths configs/paths/kaggle_prepare.yaml\n"
    ]
    nb["cells"][5]["source"] = [
        "# 5. Sinh PhoBERT Offline Token Cache cho tập Train\n",
        "!python -m vi_asr cache-teacher \\\n",
        "    --config configs/base.yaml \\\n",
        "    --paths configs/paths/kaggle_prepare.yaml \\\n",
        "    --split train \\\n",
        "    --device cuda:0\n"
    ]
    nb["cells"][6]["source"] = [
        "# 6. Kiểm toán toàn diện phân chia dữ liệu và rà soát Leakage (Gate G1)\n",
        "!python -m vi_asr audit \\\n",
        "    --config configs/base.yaml \\\n",
        "    --paths configs/paths/kaggle_prepare.yaml \\\n",
        "    --report-dir artifacts/audit_report\n"
    ]
    # Cell 7: Archive audio to single tar.gz to protect against Kaggle's 500-file limit
    archive_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7. Đóng gói Audio thành file archive duy nhất (Bảo vệ dữ liệu không bị giới hạn 500 file của Kaggle)\n",
            "!tar -czf /kaggle/working/artifacts/prepared_audio.tar.gz -C /kaggle/working/artifacts/prepared audio\n",
            "print('Đã đóng gói prepared_audio.tar.gz thành công để đính kèm sang Notebook 02!')\n"
        ]
    }
    if len(nb["cells"]) <= 7:
        nb["cells"].append(archive_cell)
    else:
        nb["cells"][7] = archive_cell

    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Patched 01_prepare.ipynb")

def patch_02():
    p = Path("notebooks/02_train_t4x2.ipynb")
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    nb["cells"][2]["source"] = [
        "# 2. Chuyển vào thư mục dự án và kiểm tra CLI\n",
        *COMMON_INIT_SNIPPET,
        "!python -m vi_asr --help\n"
    ]

    fetch_audio_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Chuẩn bị Audio Thực tế vào /kaggle/working/artifacts/prepared/audio\n",
            "# Tải trước các mẫu âm thanh thật (hoặc giải nén nếu có archive) để chạy mượt mà Pilot Test và Huấn luyện\n",
            "!python -m vi_asr fetch-audio \\\n",
            "    --config configs/base.yaml \\\n",
            "    --paths configs/paths/kaggle.yaml \\\n",
            "    --split train \\\n",
            "    --max-samples 3500\n"
        ]
    }

    # Check if fetch-audio is already present
    has_fetch = any("fetch-audio" in "".join(c.get("source", [])) for c in nb["cells"])
    if not has_fetch:
        # Insert after cell 2
        nb["cells"].insert(3, fetch_audio_cell)
    else:
        for i, c in enumerate(nb["cells"]):
            if "fetch-audio" in "".join(c.get("source", [])):
                nb["cells"][i] = fetch_audio_cell
                break

    # Renumber subsequent cells comments
    num = 4
    for i in range(4, len(nb["cells"])):
        src = nb["cells"][i].get("source", [])
        if src and src[0].startswith("# "):
            old_title = src[0].split(". ", 1)[-1] if ". " in src[0] else src[0][2:]
            src[0] = f"# {num}. {old_title}"
            num += 1

    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Patched 02_train_t4x2.ipynb")

def patch_03():
    p = Path("notebooks/03_evaluate.ipynb")
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    nb["cells"][1]["source"] = [
        "# Kiểm tra mã nguồn dự án và thiết bị\n",
        *COMMON_INIT_SNIPPET,
        "import torch\n",
        "print(\"Ready for evaluation on device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\")\n"
    ]
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Patched 03_evaluate.ipynb")

def patch_04():
    p = Path("notebooks/04_demo.ipynb")
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    nb["cells"][1]["source"] = [
        "# Chuẩn bị mã nguồn dự án và cài đặt dependencies cho Web UI\n",
        *COMMON_INIT_SNIPPET,
        "!pip install -q gradio soundfile\n",
    ]
    nb["cells"][2]["source"] = [
        "# Khởi chạy Demo Web UI\n",
        "from vi_asr.demo.app import build_demo\n",
        "from vi_asr.inference.pipeline import BilingualCascadePipeline\n",
        "from vi_asr.inference.transcribe import ASRInferencePipeline\n",
        "from vi_asr.evaluation.cascade import VinAITranslator\n",
        "from vi_asr.cli import _find_file\n",
        "from pathlib import Path\n",
        "\n",
        "# Tự động tìm kiếm checkpoint và tokenizer\n",
        "ckpt_path = _find_file(\"checkpoint_best.pt\", fallback_dir=\"runs/p2_token_temporal\")\n",
        "tok_path = _find_file(\"bpe_1024.model\", fallback_dir=\"artifacts/tokenizer\")\n",
        "\n",
        "print(f\"Checkpoint location: {ckpt_path} (exists: {ckpt_path.exists()})\")\n",
        "print(f\"Tokenizer location:  {tok_path} (exists: {tok_path.exists()})\")\n",
        "\n",
        "if ckpt_path.exists() and tok_path.exists():\n",
        "    print(\"Loading trained ASR model and VinAI translator...\")\n",
        "    asr_pipe = ASRInferencePipeline.from_checkpoint(ckpt_path, tok_path)\n",
        "    translator = VinAITranslator()\n",
        "    pipeline = BilingualCascadePipeline(asr_pipeline=asr_pipe, translator=translator)\n",
        "    demo = build_demo(pipeline_fn=pipeline.process_file)\n",
        "else:\n",
        "    print(\"Checkpoint not found. Running demo in interactive verification mode...\")\n",
        "    demo = build_demo()\n",
        "\n",
        "demo.launch(share=True, debug=True)\n"
    ]
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Patched 04_demo.ipynb")

if __name__ == "__main__":
    patch_01()
    patch_02()
    patch_03()
    patch_04()
