# Vietnamese ASR → English Translation on Kaggle T4 ×2

## Overview
This repository implements an end-to-end Vietnamese Automatic Speech Recognition (ASR) system with cascade English Machine Translation (MT), designed specifically for efficient distributed training on dual Nvidia T4 GPUs on Kaggle.

### Architecture Highlights
- **Acoustic Backbone:** Pretrained `khanhld/vip-vl-base-vie` (ChunkFormer architecture with 12 encoder blocks, 512 hidden dimension, 8 attention heads).
- **ASR Head:** Pure Linear CTC layer paired with a SentencePiece BPE subword tokenizer trained exclusively on the training split.
- **Text-Guided Token-Frame Pooling:** Uses representations from frozen `vinai/phobert-base` cached per content-token. Acoustic frames are softly aligned to text tokens via Cosine Attention with an optional weak Gaussian temporal prior.
- **Cascade Translation:** Integrated with frozen `vinai/vinai-translate-vi2en-v2` post-ASR.
- **Distributed Training:** PyTorch DDP (`torchrun`), FP16 AMP + GradScaler, duration-aware global batching, effective batch size of 32 utterances across 2× T4 GPUs.

## Quickstart

### Installation (Local Development)
```bash
pip install -e .
```

### CLI Interface
The package provides a unified CLI under `python -m vi_asr`:
```bash
# View all available subcommands
python -m vi_asr --help

# Audit environment and data configuration
python -m vi_asr audit --config configs/base.yaml --paths configs/paths/local.yaml

# Run single-GPU smoke test
python -m vi_asr smoke --config configs/base.yaml --variant configs/variants/p2.yaml --profile configs/profiles/t4_single.yaml
```

## Directory Layout
- `configs/`: Hierarchical YAML configurations (`base`, `variants`, `profiles`, `paths`).
- `src/vi_asr/`: Package source code.
- `notebooks/`: Sequential orchestration notebooks for Kaggle.
- `tests/`: Quality gate verification test suite.
- `third_party/`: Pinned upstream dependencies (`chunkformer`).
