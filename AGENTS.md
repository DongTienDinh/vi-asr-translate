# AGENTS.md — Global Context & Agent Progress Tracker

## 1. Project Overview & Scope
- **Project Name:** Vietnamese ASR → English Translation on Kaggle T4 ×2 (Dual-GPU DDP).
- **Architecture Core:**
  - **Acoustic Backbone:** `khanhld/vip-vl-base-vie` (Pretrained ChunkFormer 12 blocks, 512-dim, 8 heads, macaron FFN).
  - **ASR Head:** Pure Linear CTC `Linear(512, V)` + SentencePiece BPE tokenizer (~1,024 vocab, train-only).
  - **Text Teacher (Train-only):** `vinai/phobert-base` (frozen, extract content-token representations `[B, N, 768]` cached offline, NO single-vector mean pooling).
  - **Alignment Module:** Token–frame pooling via Cosine Attention ($H \to 256$, $E \to 256$) + soft Gaussian temporal prior + audio-only weighted sum $C = A \cdot H$ + projection to teacher space $P_v(C) \to 768$.
  - **Machine Translation:** `vinai/vinai-translate-vi2en-v2` (frozen cascade after ASR, not joint-trained).
  - **Baselines:** External `Whisper-small` multilingual zero-shot; Internal `B1` (ViP-VL + CTC thuần).
  - **Target Hardware:** Kaggle 2× T4 GPUs (DDP with torchrun, FP16 AMP + GradScaler, duration-aware global batching, effective batch = 32 utterances).

---

## 2. Implementation Progress & Roadmap

| Step | Scope | Status | Notes / Gate |
|---|---|---|---|
| **Step 1** | Project setup, `pyproject.toml`, CLI skeleton, YAML config hierarchy, `AGENTS.md` | **COMPLETED** | **Gate G0 Passed**: CLI routing, typed config validation, schema inheritance. |
| **Step 2** | Clone & Vendor `chunkformer`, Source Audit, Activation Checkpointing Patch | **COMPLETED** | **Gate G0/G2 Passed**: SHA `24b12583...` pinned, non-reentrant checkpoint patch verified (diff < 1e-5), lazy import fixed, audit doc created. |
| **Step 3** | Audio Frontend & Kaldi Parity (`povey`, 80 mel, scale $32768$, CMVN once, SpecAugment) | **COMPLETED** | **Gate G2 (Frontend) Passed**: Kaldi parity diff < 1e-5, scale 32768 confirmed, SpecAugment & GlobalCMVN verified. |
| **Step 4** | Data Adapter for HF `ViMD_Dataset`, Manifests, SentencePiece BPE Tokenizer | **COMPLETED** | **Gate G1 (Data & Tokenizer) Passed**: Manifest schema, train-only BPE, CTC collapse, inequality $T_{\text{out}} \ge U + R$, audit leakage suite verified. |
| **Step 5** | PhoBERT Offline Token Cache Generator (VnCoreNLP wseg, sliding windows > 256) | **COMPLETED** | **Gate G1 (Cache) Passed**: Sliding-window non-truncating stitching, `[N, 768]` float16 sharded npy + zero-copy `TeacherCacheReader`, KeyError on missing sample. |
| **Step 6** | ViP-VL Encoder Wrapper (clean CTC without upstream decoder) & Checkpoint Loader | *Pending* | Gate G2: Zero missing/unexpected backbone keys, `load_report.json`. |
| **Step 7** | Token–Frame Pooling & Gaussian Temporal Prior Module, Loss Functions | *Pending* | Gate G3: Mask invariance, Attention rows sum=1, teacher zero-grad. |
| **Step 8** | Gate G4 Tiny Overfit Suite (16–32 samples for B1, P1, P2) | *Pending* | Gate G4: Verified learning convergence on toy dataset. |
| **Step 9** | DDP Engine (Global Duration Sampler, Loss Normalization $\frac{R \cdot D}{W}$, Distributed Eval) | *Pending* | Gates G5, G6, G7: Gradient equivalence test, atomic resume. |
| **Step 10**| Whisper Baseline & VinAI Translate Cascade Evaluator (sacreBLEU, chrF++) | *Pending* | Evaluation suite with data contamination tracking. |
| **Step 11**| Demo Web UI (Gradio) & VAD Segmenter for Long Audio / Microphone | *Pending* | Bounded queue, parallel bilingual display. |
| **Step 12**| Packaging Script (`scripts/package_project.py`) & 4 Kaggle Notebooks | *Pending* | Code ZIP (`vi_asr_project.zip`) + Kaggle runbook. |

---

## 3. Directory Layout Conventions
```text
ASR/
  AGENTS.md                     # Global context & tracker (always updated per step)
  README.md                     # Project documentation
  IMPLEMENTATION_STATUS.md      # Gate & experimental status tracker
  THIRD_PARTY.md                # Licenses, pinned hashes, upstream credits
  pyproject.toml                # Build system & package specification
  requirements-kaggle.in        # Top-level dependencies
  requirements-extra-kaggle.lock.txt # Kaggle specific package list
  configs/
    base.yaml                   # Core configuration schema
    variants/                   # Experiment variants (b1, p1, p2, d1, p2d)
    profiles/                   # Hardware profiles (cpu_smoke, t4_single, t4x2)
    paths/                      # Environment paths (local, kaggle, kaggle_prepare)
    evaluation.yaml             # Evaluation settings
  src/
    vi_asr/                     # Main Python package
      __init__.py
      __main__.py               # python -m vi_asr entry point
      cli.py                    # Subcommand dispatcher
      config.py                 # Hierarchical YAML loader & validator
      data/                     # Audio frontend, tokenizer, manifests, caching
      models/                   # ViP-VL wrapper, CTC head, alignment, MT
      training/                 # DDP trainer, samplers, loss, schedule, checkpoint
      evaluation/               # ASR & MT metrics, bootstrap, cascade eval
      inference/                # VAD segmenter, transcription pipeline
      utils/                    # Logging, hashing, environment diagnostics
  tests/                        # Comprehensive unit tests (unittest based)
  scripts/                      # Packaging, bundle verification
  notebooks/                    # Kaggle runnable notebooks (01 to 04)
  docs/                         # Architecture specifications & audit docs
  third_party/                  # Vendored dependencies (chunkformer)
```

---

## 4. Key Engineering Invariants
1. **Frontend Scale:** Waveform MUST be multiplied by $2^{15} = 32768.0$ before Kaldi fbank; CMVN applied exactly ONCE.
2. **CTC Loss Contract:** Targets normalized per example by target length $U_i$; loss computed in FP32; $T_{\text{out}} \ge U + R$.
3. **Teacher States:** PhoBERT states frozen `[B, N, 768]` cached offline per content token; NO teacher gradients; NO text tokens injected into inference path.
4. **DDP Gradient Normalization:** Microbatch backward scaled by $1/D$; microstep sync with `model.no_sync()`; final step all-reduce total sample count $W$ and rescale gradients by $\frac{R \cdot D}{W}$ ($R=2, D=32$).
5. **Validation Integrity:** Unwrapped model evaluation with sample sharding `indices[rank::world_size]`; no drop, no duplicate; global WER from aggregated $(\sum S + \sum D + \sum I) / \sum N$.
6. **Scientific Honesty:** No fabricated metrics; untried runs marked as `not_run`; strict test set freeze before final evaluation.
