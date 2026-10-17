"""ChunkFormer encoder adapter for Vietnamese ASR (ViP-VL backbone).

Wraps ChunkFormerEncoder cleanly without upstream attention decoder,
preserving exact 12-block 512-dim macaron architecture and subsampling.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

# Ensure vendored chunkformer is available
_third_party_dir = Path(__file__).resolve().parent.parent.parent.parent / "third_party" / "chunkformer"
if _third_party_dir.exists() and str(_third_party_dir) not in sys.path:
    sys.path.insert(0, str(_third_party_dir))

from chunkformer.modules.encoder import ChunkFormerEncoder


class ChunkFormerEncoderAdapter(nn.Module):
    """Adapter wrapping ChunkFormer encoder for clean CTC ASR and DDP training.

    Guarantees:
    - Zero upstream attention decoder parameters (pure acoustic encoder).
    - Preserves exact 12-block, 512-dim, 8-head ViP-VL architecture.
    - Exact mask semantics: [B, T] bool where True=valid speech, False=padding.
    - Full-utterance context support with chunk_size=0.
    """

    def __init__(
        self,
        input_size: int = 80,
        output_size: int = 512,
        attention_heads: int = 8,
        linear_units: int = 2048,
        num_blocks: int = 12,
        dropout_rate: float = 0.1,
        encoder_layerdrop: float = 0.0,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        input_layer: str = "dw_striding",
        pos_enc_layer_type: str = "chunk_rel_pos",
        normalize_before: bool = True,
        cnn_module_kernel: int = 15,
        use_cnn_module: bool = True,
        activation_type: str = "swish",
        selfattention_layer_type: str = "chunk_rel_seflattn",
        cnn_module_norm: str = "layer_norm",
        dynamic_conv: bool = True,
        gradient_checkpointing: bool = False,
        global_cmvn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gradient_checkpointing = gradient_checkpointing

        self.encoder = ChunkFormerEncoder(
            input_size=input_size,
            output_size=output_size,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            encoder_layerdrop=encoder_layerdrop,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            input_layer=input_layer,
            pos_enc_layer_type=pos_enc_layer_type,
            normalize_before=normalize_before,
            cnn_module_kernel=cnn_module_kernel,
            use_cnn_module=use_cnn_module,
            activation_type=activation_type,
            selfattention_layer_type=selfattention_layer_type,
            cnn_module_norm=cnn_module_norm,
            dynamic_conv=dynamic_conv,
            gradient_checkpointing=gradient_checkpointing,
            global_cmvn=global_cmvn,
        )

    def encode(
        self,
        feats: torch.Tensor,
        lengths: torch.Tensor,
        context_policy: str = "full_utterance",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input filterbank features into acoustic representations H.

        Args:
            feats: Tensor [B, F, 80] input filterbank features
            lengths: Tensor [B] number of frames per sample
            context_policy: "full_utterance" sets chunk_size=0 (no chunking)

        Returns:
            H: Tensor [B, T, 512] encoder hidden representations
            output_lengths: Tensor [B] int64 sequence length after 8x subsampling
            valid_mask: Tensor [B, T] bool mask (True=speech, False=padding)
        """
        if context_policy == "full_utterance":
            chunk_size = 0
            left_context_size = 0
            right_context_size = 0
        else:
            chunk_size = 0
            left_context_size = 0
            right_context_size = 0

        # forward_encoder returns (H, masks) where masks is [B, 1, T] bool
        H, masks = self.encoder.forward_encoder(
            xs=feats,
            xs_lens=lengths,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            right_context_size=right_context_size,
        )

        valid_mask = masks.squeeze(1)  # [B, T] bool
        output_lengths = valid_mask.sum(dim=-1).to(torch.int64)  # [B] int64

        return H, output_lengths, valid_mask

    def forward(
        self,
        feats: torch.Tensor,
        lengths: torch.Tensor,
        context_policy: str = "full_utterance",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass equivalent to encode()."""
        return self.encode(feats, lengths, context_policy=context_policy)
