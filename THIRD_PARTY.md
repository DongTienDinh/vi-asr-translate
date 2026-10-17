# Third-Party Dependencies & Provenance Tracking

## 1. Upstream Repositories

| Repository | Pinned Revision / Tag | Purpose | License |
|---|---|---|---|
| [khanld/chunkformer](https://github.com/khanld/chunkformer) | `24b12583947d2338bb39f005637ee43a64c364e7` | Acoustic encoder backbone, Kaldi frontend reference | Apache-2.0 |
| [VinAIResearch/PhoBERT](https://github.com/VinAIResearch/PhoBERT) | `01daacda68afe13d83023d16ec647239e344a1e6` (HF) | Frozen text representations for offline token cache | MIT |
| [VinAIResearch/VinAI_Translate](https://github.com/VinAIResearch/VinAI_Translate) | `ae7baa85da07dbe8e23ac26a9f5ef560c17e2138` (HF) | Post-ASR cascade Vietnamese-to-English translation | Apache-2.0 |
| [nguyen-dv/ViMD_Dataset](https://github.com/nguyen-dv/ViMD_Dataset) | `3a5b30157034e7eadd5c75fae1a820c6f9383398` (HF) | Vietnamese multi-dialect speech dataset | CC BY-NC-ND 4.0 |
| [openai/whisper](https://github.com/openai/whisper) | `small` multilingual checkpoint | External baseline ASR model | MIT |
| [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | v4.0 | Voice Activity Detection for streaming demo / long audio | MIT |

---

## 2. Pretrained Model Cards & Weights

- `khanhld/vip-vl-base-vie`: Pretrained ChunkFormer on Vietnamese speech.
- `vinai/phobert-base`: Pretrained RoBERTa-based model for Vietnamese text.
- `vinai/vinai-translate-vi2en-v2`: Sequence-to-sequence neural machine translation model.
