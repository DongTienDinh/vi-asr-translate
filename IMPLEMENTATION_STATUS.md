# Implementation Status & Verification Log

**Last Updated:** 2026-10-17  
**Protocol Version:** 2.0 (Kaggle T4 ×2)

## 1. Quality Gates (G0 – G8)

| Gate | Description | Status | Verification Evidence |
|---|---|---|---|
| **G0** | Source audit, schema YAML, import CLI, vendor patch | **PASSED** | CLI routes successfully, base/variant/profile configs valid, chunkformer audited & non-reentrant checkpoint patch verified |
| **G1** | Data integrity, split audit, tokenizer, token cache | **PASSED** | Manifest schema, train-only BPE, CTC collapse & inequality, sliding-window token cache & TeacherCacheReader verified |
| **G2** | Encoder loading, frontend parity, CTC shapes | **PASSED** | Frontend Kaldi parity passed (diff < 1e-5), pure CTC encoder adapter, zero missing/unexpected backbone keys, FP32 CTC loss verified |
| **G3** | Masking, alignment loss, no-grad teacher check | **PASSED** | Mask invariance, attention rows sum=1, padding weights=0, constant values invariant, teacher zero-grad verified |
| **G4** | Tiny overfit on 16–32 samples (B1, P1, P2) | **PASSED** | Overfit convergence verified in `test_tiny_overfit.py` (loss reduced >50% within 40 epochs) |
| **G5** | DDP Sampler & gradient equivalence test | **PASSED** | Microbatch accumulation equals full-batch gradient within 1e-5 (`test_ddp_gradient_equivalence.py`), GlobalDurationSampler sync verified |
| **G6** | 2× T4 Pilot 100 updates, memory, resume check | `ready_for_kaggle` | Atomic resume verified in `test_resume.py`; training engine ready for Kaggle run in `02_train_t4x2.ipynb` |
| **G7** | Full-corpus validation aggregation parity | **PASSED** | ValidationShardSampler partitions dataset with zero drops and zero duplicates (`test_ddp_sampler.py`) |
| **G8** | Protocol freeze sign-off before test set decode | `ready_for_kaggle` | Evaluation pipelines and test manifest ready in `03_evaluate.ipynb` |

---

## 2. Experimental Variants

| Run ID | Architecture | Token Pooling | Temporal Prior | Dialect Head | Status |
|---|---|---|---|---|---|
| `B_ref` | Whisper-small zero-shot | No | No | No | `not_run` |
| `B1` | ViP-VL + CTC fine-tune | No | No | No | `not_run` |
| `P1` | ViP-VL + CTC + Token Pooling | Yes ($\lambda=0.1$) | No | No | `not_run` |
| `P2` | ViP-VL + CTC + Token Pooling + Prior | Yes ($\lambda=0.1$) | Yes ($\beta=0.1$) | No | `not_run` |
| `D1` | ViP-VL + CTC + Region Head | No | No | Yes | `optional` |
| `P2D` | P2 + Region Head | Yes | Yes | Yes | `optional` |
