# Implementation Status & Verification Log

**Last Updated:** 2026-10-17  
**Protocol Version:** 2.0 (Kaggle T4 ×2)

## 1. Quality Gates (G0 – G8)

| Gate | Description | Status | Verification Evidence |
|---|---|---|---|
| **G0** | Source audit, schema YAML, import CLI, vendor patch | **PASSED** | CLI routes successfully, base/variant/profile configs valid, chunkformer audited & non-reentrant checkpoint patch verified |
| **G1** | Data integrity, split audit, tokenizer, token cache | `not_run` | Awaiting Step 4 & Step 5 |
| **G2** | Encoder loading, frontend parity, CTC shapes | `not_run` | Awaiting Step 3 & Step 6 |
| **G3** | Masking, alignment loss, no-grad teacher check | `not_run` | Awaiting Step 7 |
| **G4** | Tiny overfit on 16–32 samples (B1, P1, P2) | `not_run` | Awaiting Step 8 |
| **G5** | DDP Sampler & gradient equivalence test | `not_run` | Awaiting Step 9 |
| **G6** | 2× T4 Pilot 100 updates, memory, resume check | `not_run` | Awaiting Kaggle run |
| **G7** | Full-corpus validation aggregation parity | `not_run` | Awaiting Step 9 |
| **G8** | Protocol freeze sign-off before test set decode | `not_run` | Awaiting Protocol sign-off |

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
