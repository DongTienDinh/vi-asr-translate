"""Vietnamese ASR to English Translation package."""

import os

# Ensure writable cache directories in Kaggle / restricted environments
if os.path.exists("/kaggle") or not os.access(os.path.expanduser("~"), os.W_OK):
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface/hub")
    os.environ.setdefault("TORCH_HOME", "/tmp/torch")

__version__ = "0.1.0"
