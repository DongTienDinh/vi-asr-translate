"""Update notebook cells to auto-clone repository if not already present."""

import json
from pathlib import Path

def patch_01():
    p = Path("notebooks/01_prepare.ipynb")
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    nb["cells"][2]["source"] = [
        "# 2. Nạp mã nguồn vi_asr_project (từ Zip hoặc Git Clone trực tiếp)\n",
        "import os, zipfile\n",
        "from pathlib import Path\n",
        "\n",
        "work_dir = Path(\"/kaggle/working/vi_asr_project\")\n",
        "zip_source = Path(\"/kaggle/input/vi-asr-code/vi_asr_project.zip\")\n",
        "if not zip_source.exists():\n",
        "    zip_source = Path(\"vi_asr_project.zip\")\n",
        "\n",
        "if zip_source.exists():\n",
        "    with zipfile.ZipFile(zip_source, 'r') as zf:\n",
        "        zf.extractall(\"/kaggle/working\")\n",
        "    print(\"Unpacked vi_asr_project from zip successfully.\")\n",
        "elif not (work_dir / \"pyproject.toml\").exists():\n",
        "    print(\"Cloning repository directly from GitHub...\")\n",
        "    !rm -rf /kaggle/working/vi_asr_project\n",
        "    !git clone --recurse-submodules https://github.com/DongTienDinh/vi-asr-translate.git /kaggle/working/vi_asr_project\n",
        "else:\n",
        "    print(\"Using existing repository at:\", work_dir)\n",
        "\n",
        "# Cài đặt package vi-asr ở chế độ editable\n",
        "%cd /kaggle/working/vi_asr_project\n",
        "!pip install -q -e .\n"
    ]
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Patched 01_prepare.ipynb")

def patch_02():
    p = Path("notebooks/02_train_t4x2.ipynb")
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    nb["cells"][2]["source"] = [
        "# 2. Chuyển vào thư mục dự án và kiểm tra CLI\n",
        "from pathlib import Path\n",
        "\n",
        "work_dir = Path(\"/kaggle/working/vi_asr_project\")\n",
        "if not (work_dir / \"pyproject.toml\").exists():\n",
        "    !rm -rf /kaggle/working/vi_asr_project\n",
        "    !git clone --recurse-submodules https://github.com/DongTienDinh/vi-asr-translate.git /kaggle/working/vi_asr_project\n",
        "    %cd /kaggle/working/vi_asr_project\n",
        "    !pip install -q -e .\n",
        "else:\n",
        "    %cd /kaggle/working/vi_asr_project\n",
        "\n",
        "!python -m vi_asr --help\n"
    ]
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Patched 02_train_t4x2.ipynb")

def patch_03():
    p = Path("notebooks/03_evaluate.ipynb")
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    nb["cells"][1]["source"] = [
        "# Kiểm tra mã nguồn dự án và thiết bị\n",
        "from pathlib import Path\n",
        "import torch\n",
        "\n",
        "work_dir = Path(\"/kaggle/working/vi_asr_project\")\n",
        "if not (work_dir / \"pyproject.toml\").exists():\n",
        "    !rm -rf /kaggle/working/vi_asr_project\n",
        "    !git clone --recurse-submodules https://github.com/DongTienDinh/vi-asr-translate.git /kaggle/working/vi_asr_project\n",
        "    %cd /kaggle/working/vi_asr_project\n",
        "    !pip install -q -e .\n",
        "else:\n",
        "    %cd /kaggle/working/vi_asr_project\n",
        "\n",
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
        "from pathlib import Path\n",
        "\n",
        "work_dir = Path(\"/kaggle/working/vi_asr_project\")\n",
        "if not (work_dir / \"pyproject.toml\").exists():\n",
        "    !rm -rf /kaggle/working/vi_asr_project\n",
        "    !git clone --recurse-submodules https://github.com/DongTienDinh/vi-asr-translate.git /kaggle/working/vi_asr_project\n",
        "\n",
        "%cd /kaggle/working/vi_asr_project\n",
        "!pip install -q -e .\n",
        "!pip install -q gradio soundfile\n"
    ]
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Patched 04_demo.ipynb")

if __name__ == "__main__":
    patch_01()
    patch_02()
    patch_03()
    patch_04()
