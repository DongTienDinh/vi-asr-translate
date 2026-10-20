"""PhoBERT offline token cache generator, sliding window stitching, and sharded reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from vi_asr.data.manifest import ManifestItem
from vi_asr.data.normalize import normalize_text_raw
from vi_asr.utils.logging import setup_logger

logger = setup_logger("vi_asr.data.teacher_cache")


class PhoBERTTeacherExtractor:
    """Extracts per-content-token hidden states from frozen PhoBERT.

    Engineering Invariants:
    1. Special tokens (<s>, </s>, <pad>) are stripped.
    2. Utterances with > 254 content tokens are processed via non-truncating sliding windows.
    3. Outputs are per-content-token [N, 768], strictly NOT single-vector mean pooled.
    4. Evaluated under torch.inference_mode() with zero gradients.
    """

    def __init__(
        self,
        model_id: str = "vinai/phobert-base",
        revision: str = "01daacda68afe13d83023d16ec647239e344a1e6",
        device: str = "cpu",
        max_content_window: int = 254,
    ):
        from transformers import AutoModel, AutoTokenizer
        self.model_id = model_id
        self.revision = revision
        if "cuda" in str(device) and not torch.cuda.is_available():
            logger.warning("CUDA requested for PhoBERT teacher but not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)
        self.max_content_window = max_content_window

        logger.info(f"Loading PhoBERT teacher ({model_id} @ {revision[:8]}) on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
        )
        self.model.eval()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    @torch.inference_mode()
    def extract_from_text(self, text: str) -> Tuple[torch.Tensor, List[int]]:
        """Extract per-content-token states for a single utterance text.

        Returns:
            states: float16 Tensor of shape [N, 768]
            content_token_ids: List[int] of length N
        """
        norm_text = normalize_text_raw(text)
        if not norm_text:
            return torch.zeros((0, 768), dtype=torch.float16), []

        # Tokenize without adding special tokens automatically
        all_token_ids = self.tokenizer.encode(norm_text, add_special_tokens=False)

        # Filter out any accidentally included special tokens
        content_token_ids = [
            tid
            for tid in all_token_ids
            if tid not in (self.bos_token_id, self.eos_token_id, self.pad_token_id)
        ]

        N = len(content_token_ids)
        if N == 0:
            return torch.zeros((0, 768), dtype=torch.float16), []

        # Chunk into windows <= max_content_window to respect PhoBERT's 256 position limit
        chunks: List[List[int]] = [
            content_token_ids[i : i + self.max_content_window]
            for i in range(0, N, self.max_content_window)
        ]

        extracted_states: List[torch.Tensor] = []

        for chunk in chunks:
            # Wrap with <s> and </s>
            window_input_ids = [self.bos_token_id] + chunk + [self.eos_token_id]
            input_tensor = torch.tensor([window_input_ids], dtype=torch.long, device=self.device)

            outputs = self.model(input_ids=input_tensor)
            # outputs.last_hidden_state is [1, window_len, 768]
            last_hidden = outputs.last_hidden_state[0]  # [window_len, 768]

            # Slice out content tokens: index 1 to len(chunk) + 1
            content_states = last_hidden[1 : 1 + len(chunk), :].cpu().to(torch.float16)
            extracted_states.append(content_states)

        stitched_states = torch.cat(extracted_states, dim=0)  # [N, 768]
        assert stitched_states.size(0) == N, f"Token count mismatch: {stitched_states.size(0)} vs {N}"
        return stitched_states, content_token_ids


def build_offline_teacher_cache(
    manifest_items: List[ManifestItem],
    output_dir: Union[str, Path],
    extractor: PhoBERTTeacherExtractor,
    shard_size: int = 50000,
) -> Path:
    """Build sharded memory-mappable NumPy token cache for train split.

    Artifacts produced:
    - meta.json: Metadata & invariant parameters
    - index.jsonl: sample_id -> offset & token mapping
    - tokens-000.npy: [sum_N, 768] float16
    - token_ids-000.npy: [sum_N] int32
    - offsets-000.npy: [num_samples + 1] int64
    """
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    all_tokens: List[np.ndarray] = []
    all_token_ids: List[np.ndarray] = []
    offsets: List[int] = [0]
    index_entries: List[Dict[str, Any]] = []

    current_offset = 0
    total_tokens = 0

    for idx, item in enumerate(manifest_items):
        # Prefer pre-segmented teacher text if available, else text_raw
        src_text = item.text_teacher or item.text_raw
        states, token_ids = extractor.extract_from_text(src_text)

        num_tokens = states.size(0)
        total_tokens += num_tokens

        states_np = states.numpy()  # float16
        token_ids_np = np.array(token_ids, dtype=np.int32)

        all_tokens.append(states_np)
        all_token_ids.append(token_ids_np)

        index_entries.append({
            "sample_id": item.sample_id,
            "shard_idx": 0,
            "start_offset": current_offset,
            "num_tokens": num_tokens,
            "text_hash": item.text_hash,
            "token_ids": token_ids,
        })

        current_offset += num_tokens
        offsets.append(current_offset)

    # Concatenate contiguous arrays
    if all_tokens:
        tokens_array = np.concatenate(all_tokens, axis=0)  # [total_tokens, 768]
        token_ids_array = np.concatenate(all_token_ids, axis=0)  # [total_tokens]
    else:
        tokens_array = np.zeros((0, 768), dtype=np.float16)
        token_ids_array = np.zeros((0,), dtype=np.int32)

    offsets_array = np.array(offsets, dtype=np.int64)

    # Save shard arrays
    np.save(out_p / "tokens-000.npy", tokens_array)
    np.save(out_p / "token_ids-000.npy", token_ids_array)
    np.save(out_p / "offsets-000.npy", offsets_array)

    # Save index.jsonl
    with open(out_p / "index.jsonl", "w", encoding="utf-8") as f:
        for entry in index_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Save meta.json
    meta = {
        "model_id": extractor.model_id,
        "revision": extractor.revision,
        "layer": "last_hidden_state",
        "hidden_size": 768,
        "dtype": "float16",
        "window_size": extractor.max_content_window,
        "total_samples": len(manifest_items),
        "total_tokens": total_tokens,
    }
    with open(out_p / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Built teacher token cache: {len(manifest_items)} samples, {total_tokens} tokens saved to {out_p}"
    )
    return out_p


class TeacherCacheReader:
    """Fast, zero-copy memory-mapped reader for PhoBERT token representations."""

    def __init__(self, cache_dir: Union[str, Path]):
        self.cache_dir = Path(cache_dir)
        meta_file = self.cache_dir / "meta.json"
        index_file = self.cache_dir / "index.jsonl"
        tokens_file = self.cache_dir / "tokens-000.npy"
        token_ids_file = self.cache_dir / "token_ids-000.npy"

        if not meta_file.exists() or not index_file.exists():
            raise FileNotFoundError(f"Invalid teacher cache directory: {self.cache_dir}")

        with open(meta_file, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.index: Dict[str, Dict[str, Any]] = {}
        with open(index_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    self.index[entry["sample_id"]] = entry

        # Memory-map the token representations for fast random access without RAM blowup
        self.tokens_mmap = np.load(tokens_file, mmap_mode="r")
        self.token_ids_mmap = np.load(token_ids_file, mmap_mode="r")

    def close(self) -> None:
        """Release memory-mapped file handles (especially important on Windows)."""
        if hasattr(self, "tokens_mmap"):
            del self.tokens_mmap
        if hasattr(self, "token_ids_mmap"):
            del self.token_ids_mmap
        import gc
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, sample_id: str) -> bool:
        return sample_id in self.index

    def get(self, sample_id: str) -> Tuple[torch.Tensor, List[int]]:
        """Retrieve token representation and token IDs for a sample.

        Returns:
            states: Tensor [N, 768] (converted to float32)
            token_ids: List of content token IDs

        Raises:
            KeyError if sample_id is not in cache (NEVER returns silent zeroes).
        """
        if sample_id not in self.index:
            raise KeyError(
                f"Sample ID '{sample_id}' not found in PhoBERT teacher cache at {self.cache_dir}. "
                "Missing teacher states are not permitted."
            )

        entry = self.index[sample_id]
        start = entry["start_offset"]
        N = entry["num_tokens"]

        if N == 0:
            return torch.zeros((0, 768), dtype=torch.float32), []

        # Zero-copy slice from memmap, then convert to PyTorch tensor
        states_slice = self.tokens_mmap[start : start + N]  # [N, 768] in float16
        states_tensor = torch.from_numpy(np.array(states_slice)).float()  # convert to float32 for loss
        token_ids = entry.get("token_ids", [])
        return states_tensor, token_ids
