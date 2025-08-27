# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""XCodec2 audio decoder implementation for Llasa TTS.

This module implements the decoder path of XCodec2, optimized for Llama-based TTS.
Since Llama provides speech tokens directly, only the token→audio conversion
is needed, not the full audio→token encoding pipeline.
"""

from .config import XCodec2Config
from .decoder import XCodec2Decoder
from .residual_fsq import ResidualFSQ

__all__ = [
    "ResidualFSQ",
    "XCodec2Config",
    "XCodec2Decoder",
]
