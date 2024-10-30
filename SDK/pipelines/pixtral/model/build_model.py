# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Build a Pixtral model via Graph API from #TODO weights."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from max.dtype import DType
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import GGUFWeights
from nn.conv import Conv2D
from nn.norm import RMSNorm
from pixtral.model.rotary_embedding_2d import (
    RotaryEmbedding2D,
    patch_position_ids,
)
from pixtral.model.vision_encoder import VisionEncoder
