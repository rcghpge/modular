# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""XCodec2 audio decoder implementation for Llasa TTS."""

from .config import XCodec2Config
from .decoder import XCodec2Decoder
from .residual_fsq import ResidualFSQ

__all__ = [
    "ResidualFSQ",
    "XCodec2Config",
    "XCodec2Decoder",
]
