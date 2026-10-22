# Vietnamese ASR → English Translation on Kaggle T4 ×2

## Overview
This repository implements an end-to-end Vietnamese Automatic Speech Recognition (ASR) system with cascade English Machine Translation (MT), designed specifically for efficient distributed training on dual Nvidia T4 GPUs on Kaggle.

### Architecture Highlights
- **Acoustic Backbone:** Pretrained `khanhld/vip-vl-base-vie` (ChunkFormer architecture with 12 encoder blocks, 512 hidden dimension, 8 attention heads).
- **ASR Head:** Pure Linear CTC layer paired with a SentencePiece BPE subword tokenizer trained exclusively on the training split.
- **Text-Guided Token-Frame Pooling:** Uses representations from frozen `vinai/phobert-base` cached per content-token. Acoustic frames are softly aligned to text tokens via Cosine Attention with an optional weak Gaussian temporal prior.
- **Cascade Translation:** Integrated with frozen `vinai/vinai-translate-vi2en-v2` post-ASR.
- **Distributed Training:** PyTorch DDP (`torchrun`), FP16 AMP + GradScaler, duration-aware global batching, effective batch size of 32 utterances across 2× T4 GPUs.

## Experimental baseline

`B1` is the sole ASR baseline: the pretrained `khanhld/vip-vl-base-vie` encoder is fine-tuned with a newly initialized Linear CTC head and the train-only SentencePiece tokenizer. `P1` and `P2` retain the same encoder, head initialization, data splits, frontend, training budget, and decoding protocol; they add only the proposed training objectives.

Checkpoint selection uses validation WER. Final ASR results use corpus-level WER and CER on the frozen ViMD test split, with substitution/deletion/insertion counts and regional breakdowns. No test result is reported until a main checkpoint has actually been trained and evaluated; unrun experiments remain `not_run`.

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

### Kaggle notebook order

1. Upload the generated `vi_asr_project.zip` and run `notebooks/01_prepare.ipynb` from top to bottom.
2. Save a Notebook 01 version only after its final handoff cell reports `PASS`.
3. Attach the saved Notebook 01 output and the newest code ZIP to `notebooks/02_train_t4x2.ipynb`; use 2× T4 and Internet on.
4. Notebook 02 is frozen to optimized B1 protocol `b1_repro_v2`, seed 42 and 20 epochs. Save each session, then attach only the newest `b1_ctc_v2` output to the next session for automatic epoch-boundary resume. Do not resume a v1 checkpoint.
5. After `checkpoint_latest.pt` reaches epoch 20, attach that output with Notebook 01 data to `notebooks/03_evaluate.ipynb`. Notebook 03 evaluates only `checkpoint_best.pt` and writes `b1_reproduction_report.json` after decoding all 2,026 test samples.

Notebook 01 verifies one audio file per manifest row, the full train teacher-cache index,
the train-only tokenizer, and Gate G1 before creating checksummed ZIP_STORED audio shards
plus `vi_asr_handoff.json`. Notebook 02 reads individual WAV members directly from those
read-only `/kaggle/input` shards, without extracting audio into `/kaggle/working`, and
refuses to continue when coverage or integrity checks fail.

For B1 reproducibility, both DDP ranks use the same sampler seed, model initialization is
identical before DDP broadcast, training/validation use separate frontend instances,
validation and inference force dither 0, and the pinned ViP-VL `global_cmvn` is applied
exactly once. The linear schedule is counted in optimizer updates rather than microbatches.
Every run saves the resolved configuration hash, encoder checkpoint hash, seed, update
counts, and best-validation selection metadata.

The T4×2 v2 profile uses one spawned input worker per rank, pinned-memory prefetch,
non-blocking device copies, four samples per rank and four accumulation steps. This keeps
the nominal effective batch at 32 while overlapping ZIP/WAV/fbank preparation with GPU
work. Rank 0 reports loss, throughput, data-wait ratio, learning rate and peak GPU memory
every 20 optimizer updates. Validation reuses its first forward pass for greedy CTC decode.

## Directory Layout
- `configs/`: Hierarchical YAML configurations (`base`, `variants`, `profiles`, `paths`).
- `src/vi_asr/`: Package source code.
- `notebooks/`: Sequential orchestration notebooks for Kaggle.
- `tests/`: Quality gate verification test suite.
- `third_party/`: Pinned upstream dependencies (`chunkformer`).
