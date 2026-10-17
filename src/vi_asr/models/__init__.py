"""Model architectures and adapters for Vietnamese ASR."""

import sys
from pathlib import Path

# Ensure third_party/chunkformer is in sys.path
_third_party_chunkformer = Path(__file__).resolve().parent.parent.parent.parent / "third_party" / "chunkformer"
if _third_party_chunkformer.exists() and str(_third_party_chunkformer) not in sys.path:
    sys.path.insert(0, str(_third_party_chunkformer))

from vi_asr.models.checkpoint import load_vip_vl_pretrained
from vi_asr.models.ctc_asr import CTCHead, ViPViCTCASR
from vi_asr.models.encoder_adapter import ChunkFormerEncoderAdapter

__all__ = [
    "ChunkFormerEncoderAdapter",
    "CTCHead",
    "ViPViCTCASR",
    "load_vip_vl_pretrained",
]
