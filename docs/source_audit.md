# Source Audit Report: `khanld/chunkformer`

**Audited Repository:** [https://github.com/khanld/chunkformer](https://github.com/khanld/chunkformer)  
**Pinned Commit SHA:** `24b12583947d2338bb39f005637ee43a64c364e7`  
**Date:** 2026-10-17  
**Auditor:** vi_asr research & engineering team  

---

## 1. Executive Summary

We audited the vendored `chunkformer` acoustic encoder library to verify architectural integrity, tensor shapes, masking conventions, and gradient checkpointing behavior. 

### Critical Findings:
1. **Gradient Checkpointing Bug:** The upstream `ChunkFormerEncoder` accepts `gradient_checkpointing: bool = False` in its constructor, but its `forward_layers` implementation directly invoked `layer(...)` without calling `torch.utils.checkpoint.checkpoint`. Flagging `activation_checkpointing: true` previously had zero effect on activation memory.
2. **Resolution Applied:** We authored `third_party/patches/0001-activation-checkpointing.patch` implementing non-reentrant checkpointing (`use_reentrant=False`) with per-layer closure binding via `make_custom_forward(layer)`.
3. **Eager Import Issue:** Upstream `chunkformer/__init__.py` unconditionally imported `ChunkFormerModel`, pulling heavy evaluation dependencies (`jiwer`, `soundfile`, `torchaudio`). We patched `chunkformer/__init__.py` with PEP 562 lazy loading (`__getattr__`), allowing isolated usage of `ChunkFormerEncoder`.
4. **LayerDrop Invariant:** Set `encoder_layerdrop = 0.0` for all DDP training runs to prevent per-rank structural graph divergence.

---

## 2. Component-by-Component Audit

| File / Component | Class / Signature | Input / Output Shapes | Mask Semantics | Action & Status |
|---|---|---|---|---|
| `chunkformer/modules/encoder.py` | `ChunkFormerEncoder` | Inputs: `xs [B, F, 80]`, `xs_lens [B]`. Outputs: `H [B, T, 512]`, `masks [B, 1, T]`. | `masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)` (`True` indicates valid frame, `False` indicates padding). | **Patched**: Added non-reentrant activation checkpointing. |
| `chunkformer/modules/encoder_layer.py` | `ChunkFormerEncoderLayer` | Inputs: `x [B, T, 512]`, `mask [B, 1, T]`, `pos_emb [1, 2T-1, 512]`. Output: `x [B, T, 512]`. | Macaron FFN (scale 0.5), Chunk Rel Self-Attention, Convolution (kernel 15), FFN (scale 0.5), LayerNorm. | **Reused as-is**. |
| `chunkformer/modules/subsampling.py` | `DepthwiseConvSubsampling` | Input: `[B, F, 80]`. Output: `[B, T, 512]`, where $T \approx F / 8$. | Downsamples acoustic features by 8× through two depthwise-separable conv layers with stride 2. | **Reused as-is**. Subsampling length calculation verified. |
| `chunkformer/modules/attention.py` | `ChunkAttentionWithRelativeRightContext` | Keys, Queries, Values, Rel Pos Embeddings. Output: Attention context `[B, T, 512]`. | Chunk relative self-attention with optional right context. | **Reused as-is**. |
| `chunkformer/modules/convolution.py` | `ChunkConvolutionModule` | Input: `[B, T, 512]`, `mask_pad [B, 1, T]`. | Pointwise Conv $\to$ Gated Linear Unit $\to$ Depthwise Conv (kernel 15) $\to$ BatchNorm $\to$ Swish $\to$ Pointwise Conv. | **Reused as-is**. |
| `chunkformer/modules/cmvn.py` | `GlobalCMVN` | Waveform fbank features `[B, F, 80]`. | Normalizes features using checkpoint mean and variance buffers. Must be applied **exactly once**. | **Reused as-is**. |
| `chunkformer/__init__.py` | Top-level init | N/A | Exposes `__version__` and `ChunkFormerModel`. | **Patched**: Lazy load via `__getattr__`. |

---

## 3. Subsampling and CTC Output Length Contract

The depthwise striding frontend downsamples the input time dimension by a factor of 8:
$$T_{\text{out}} = \left\lfloor \frac{\lfloor (F - 1) / 2 \rfloor - 1}{2} \right\rfloor \dots \approx \frac{F}{8}$$
For CTC validity, target length $U$ and consecutive repeated token count $R$ must satisfy:
$$T_{\text{out}} \ge U + R$$
This contract is verified during data audit and in `tests/test_ctc_contract.py`.

---

## 4. Activation Checkpointing Test Contract

In `tests/test_checkpoint_patch.py`:
1. Model weights and inputs are held strictly identical (`torch.manual_seed(42)`).
2. With dropout disabled (`model.eval()`), the outputs of `forward()` with `gradient_checkpointing=False` and `gradient_checkpointing=True` must match with absolute maximum difference $< 10^{-6}$.
3. In training mode (`model.train()`), backward gradients computed on model parameters must match within floating-point tolerance ($< 10^{-5}$).
