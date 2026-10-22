# AGENTS.md — Global Context & Agent Progress Tracker

## 1. Project Overview & Scope
- **Project Name:** Vietnamese ASR → English Translation on Kaggle T4 ×2 (Dual-GPU DDP).
- **Architecture Core:**
  - **Acoustic Backbone:** `khanhld/vip-vl-base-vie` (Pretrained ChunkFormer 12 blocks, 512-dim, 8 heads, macaron FFN).
  - **ASR Head:** Pure Linear CTC `Linear(512, V)` + SentencePiece BPE tokenizer (~1,024 vocab, train-only).
  - **Text Teacher (Train-only):** `vinai/phobert-base` (frozen, extract content-token representations `[B, N, 768]` cached offline, NO single-vector mean pooling).
  - **Alignment Module:** Token–frame pooling via Cosine Attention ($H \to 256$, $E \to 256$) + soft Gaussian temporal prior + audio-only weighted sum $C = A \cdot H$ + projection to teacher space $P_v(C) \to 768$.
  - **Machine Translation:** `vinai/vinai-translate-vi2en-v2` (frozen cascade after ASR, not joint-trained).
  - **Baseline:** `B1` (ViP-VL + Linear CTC thuần), là đối chứng nội bộ duy nhất cho P1/P2.
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
| **Step 6** | ViP-VL Encoder Wrapper (clean CTC without upstream decoder) & Checkpoint Loader | **COMPLETED** | **Gate G2 (Encoder & CTC) Passed**: Pure CTC without upstream decoder, zero missing/unexpected backbone keys, FP32 target-normalized CTC loss, `load_report.json` verified. |
| **Step 7** | Token–Frame Pooling & Gaussian Temporal Prior Module, Loss Functions | **COMPLETED** | **Gate G3 (Alignment & Losses) Passed**: Mask invariance, attention rows sum=1, padding weights=0, constant values invariant, teacher zero-grad verified. |
| **Step 8** | Gate G4 Tiny Overfit Suite (16–32 samples for B1, P1, P2) | **COMPLETED** | **Gate G4 Passed**: Loss reduction > 50% verified across B1, P1, P2 toy overfit in unit tests. |
| **Step 9** | DDP Engine (Global Duration Sampler, Loss Normalization $\frac{R \cdot D}{W}$, Distributed Eval) | **COMPLETED** | **Gates G5, G7 Passed**: Gradient equivalence verified (< 1e-5), exact sharding coverage (0 drops, 0 duplicates), atomic resume test verified. |
| **Step 10**| B1/P1/P2 Evaluation & VinAI Translate Cascade Evaluator (sacreBLEU, chrF++) | **COMPLETED** | Evaluation suite with corpus WER/CER, sacreBLEU, chrF++, paired bootstrap hypothesis test (1000 resamples), and VinAI translator wrapper verified. |
| **Step 11**| Demo Web UI (Gradio) & VAD Segmenter for Long Audio / Microphone | **COMPLETED** | Audio-only inference pipeline, energy VAD segmenter, bilingual cascade pipeline, Gradio 6.x demo UI verified. |
| **Step 12**| Packaging Script (`scripts/package_project.py`) & 4 Kaggle Notebooks | **COMPLETED** | Kaggle Notebook 01→02 handoff uses checksummed ZIP_STORED audio shards read directly from `/kaggle/input`, exact artifact validation, bounded checkpoint storage, epoch-boundary resume, and source discovery for both legacy and current nested Kaggle mounts. |

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

---

## 5. Scientific Positioning and Terminology

- The project **does not pretrain ViP-VL again**. It downloads the released encoder-only
  checkpoint `khanhld/vip-vl-base-vie` at pinned revision
  `4b259337d96450d57403acdee82572c587151e22`, loads all backbone parameters with
  `strict=True`, initializes a new Linear CTC head, and then performs supervised fine-tuning.
- `B1` is an **internal project baseline**, not an exact reproduction of the ViP-VL paper's
  ASR experiment. The paper fine-tunes on VLSP 2020 (~250 h) for 150 epochs on 4× H200;
  this project fine-tunes on the audited ViMD split for a fixed 20-epoch budget on 2× T4.
- The paper's reported ViMD WER must be treated only as an external reference. It must not
  be presented as a result obtained by this repository.
- `P1` and `P2` must each start independently from the same pinned pretrained ViP-VL
  checkpoint and shared initialization/seed protocol. They must **not** continue training
  from a completed B1 checkpoint, because that would give them a larger training exposure.
- Development and checkpoint selection use validation only. The frozen test split is decoded
  once after the protocol and variants are frozen.

---

## 6. Current Kaggle Data State (Verified)

Notebook 01 has already produced a valid immutable Kaggle output and **does not need to be
rerun** for the current B1 attempt.

| Artifact | Verified value |
|---|---:|
| Train samples/audio | 15,023 / 15,023 |
| Validation samples/audio | 1,900 / 1,900 |
| Test samples/audio | 2,026 / 2,026 |
| Total audio files | 18,949 |
| Audio shard count | 23 |
| Uncompressed shard payload | 11,816,148,360 bytes |
| Audio index SHA256 | `b1ca837a32cf0e4a41f46f8e3ce7f2eb3d6a9c3547dc3b645d309067ef947199` |
| Train manifest SHA256 | `3db82f7bf900c3bce80a394669889d9667c0bd40a7252c5d0050f32ff7d3d88b` |
| Valid manifest SHA256 | `632395e5057271f9bad002095909418dc72bd4c57a1a11c28a184b00cc955e48` |
| Test manifest SHA256 | `e34ab0a6231ecdd40eea9dcbf66b3a7d2cd90b480bc062cd41430b094b592492` |
| Teacher cache samples/tokens | 15,023 / 1,078,433 |

Notebook 02 reads individual WAV members directly from the read-only ZIP_STORED shards under
`/kaggle/input`; it must not extract the 11.8 GB audio payload into `/kaggle/working`.

The second preflight display may show `"hashes_verified": false` intentionally because that
specific call uses `verify_hashes=False` after the preceding restore step has already reported
`audio_shards_sha256_verified: true`. This is not a data-integrity failure.

---

## 7. Current B1 Execution State

- Official B1 status remains **`not_run`**.
- Previous Kaggle sessions reached epoch 3/20 and produced checkpoints, but they selected a
  fallback source ZIP embedded in Notebook 01 output instead of the dedicated `vi-asr-code`
  Dataset. Observed fallback source hashes included `c71492dd...` and `cae5a10d...`.
- Those fallback checkpoints are diagnostic only and must **not** be resumed or reported as
  official B1 results.
- Their displayed `best_wer = 1.0`, `best_loss ≈ 13.7`, and `best_epoch = 1` are not official
  metrics. WER 1.0 means 100% validation WER, which may reflect early CTC blank collapse, but
  the runs are excluded first because their source provenance is not the frozen package.
- Warnings about frozen Python modules, unauthenticated Hugging Face requests,
  `OMP_NUM_THREADS=1`, Mistune/nbconvert escape sequences, and duplicate rank log messages
  are non-fatal. A successful session is identified by `PASS SESSION` plus the required
  checkpoint files.

### Root cause of the source-selection issue

The current Kaggle image can mount a user Dataset at:

```text
/kaggle/input/datasets/<owner>/vi-asr-code/vi_asr_project
```

rather than only at the older short path:

```text
/kaggle/input/vi-asr-code/vi_asr_project
```

The latest Notebook 02 recursively discovers either layout, gives any path containing the
Dataset slug `vi-asr-code` priority over Notebook 01 fallback ZIPs, and fails before expensive
shard verification if the selected source is outside `vi-asr-code`.

---

## 8. Latest Files to Upload

| Purpose | Local file | Required identity |
|---|---|---|
| Kaggle source Dataset | `artifacts/vi_asr_project.zip` | Use SHA256 from the generated `artifacts/packaging_manifest.json` |
| B1 training notebook | `notebooks/02_train_t4x2.ipynb` | Latest generated notebook with nested Kaggle Dataset discovery and early fail-closed source gate |

The Kaggle Dataset should use slug `vi-asr-code`. Kaggle may expose the uploaded archive as an
expanded `vi_asr_project` directory; this is supported. For a fresh official B1 session, attach
only:

1. the existing verified Notebook 01 output;
2. the latest `vi-asr-code` Dataset;
3. no old Notebook 02 output/checkpoint.

The beginning of the official log must report a source such as:

```text
source: /kaggle/input/datasets/<owner>/vi-asr-code/vi_asr_project
kind: directory
```

or the equivalent legacy short path. It must never report the Notebook 01 embedded
`vi_asr_project.zip` as the training source.

---

## 9. B1 Training Protocol on Kaggle

Notebook 02 is frozen to:

```text
RUN_NAME = b1_ctc_v2
protocol_version = b1_repro_v2
seed = 42
TOTAL_EPOCHS = 20
EPOCHS_THIS_SESSION = 3
RESUME_CHECKPOINT = auto
RUN_SMOKE = False
```

- Use `Save Version → Save & Run All`, T4×2, Internet ON.
- The pretrained ViP-VL `pytorch_model.pt` (~315 MB) and `global_cmvn` are downloaded
  automatically from the pinned Hugging Face revision. Failure to load them aborts training;
  the code refuses to train a silently random backbone.
- Each version runs at most three complete epochs and saves an epoch-boundary checkpoint.
- Session plan: `1–3`, `4–6`, `7–9`, `10–12`, `13–15`, `16–18`, `19–20`.
- From the second session onward, attach only the newest B1 Notebook output in addition to
  Notebook 01 and `vi-asr-code`; automatic resume must print the expected next epoch.
- `RUN_SMOKE=False` skips the toy smoke test. Validation on all 1,900 validation samples still
  runs after every epoch and must not be confused with frozen test-set evaluation.
- Notebook 03 is used only after `checkpoint_latest.pt` reaches epoch 20. It evaluates
  `checkpoint_best.pt`, chosen by validation WER, then CER, then earliest epoch.

Expected output directory per successful session:

```text
/kaggle/working/runs/b1_ctc_v2/
  checkpoint_latest.pt
  checkpoint_best.pt
  checkpoint_epoch_XXX.pt
  resolved_config.yaml
  run_metadata.json
  load_report.json
  notebook_protocol.json
```

---

## 10. Runtime Observations and Next Action

- The older v1 T4×2 run spent about 19 minutes verifying the immutable 11.8 GB shard payload
  and roughly 33–42 minutes per epoch including validation. These timings are not v2 results.
- ViP-VL has about 78M parameters. Download time is small relative to full-encoder forward,
  backward, activation recomputation, validation, and CPU WAV/fbank work.
- Protocol v2 uses AMP FP16, GradScaler, activation checkpointing, maximum four samples per
  rank, four-step accumulation, and effective batch size 32.
- `configs/profiles/t4x2_fast.yaml` is now honored by the CLI: one spawned worker per rank,
  prefetch factor 2, pinned memory, non-blocking transfers, and deterministic epoch/rank worker
  seeds. Rank 0 logs throughput/data-wait/GPU-memory every 20 optimizer updates.
- Validation now reuses the logits from its loss forward pass for greedy CTC decoding instead
  of running the encoder twice. v2 throughput and peak-memory numbers remain `not_run` until
  measured on Kaggle; do not claim a speedup before that measurement.

**Immediate next action:** upload the new package and Notebook 02, attach no v1 B1 checkpoint,
begin `b1_ctc_v2` at epoch 1, record the first-epoch throughput/data-wait/peak-memory telemetry,
and mark B1 complete only after a verified 20-epoch run and Notebook 03 report.
