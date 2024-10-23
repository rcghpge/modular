# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""All configurable parameters for Pixtral."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from max.driver import CPU, Device
from max.dtype import DType


class SupportedVersions(str, Enum):
    pixtral_12B_2409 = "12B-2409"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


class SupportedEncodings(str, Enum):
    float32 = "float32"
    bfloat16 = "bfloat16"

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    @property
    def dtype(self) -> DType:
        return _ENCODING_TO_DTYPE[self]

    def hf_model_name(self, version: SupportedVersions) -> str:
        if version == SupportedVersions.pixtral_12B_2409:
            return _ENCODING_TO_MODEL_NAME_PIXTRAL[self]
        else:
            raise ValueError(f"Unsupported version: {version}")


_ENCODING_TO_DTYPE = {
    SupportedEncodings.float32: DType.float32,
    SupportedEncodings.bfloat16: DType.bfloat16,
}


_ENCODING_TO_MODEL_NAME_PIXTRAL = {
    # TODO: add pixtral weights file_name
    SupportedEncodings.float32: "replit-code-v1_5-3b-f32.gguf",
    SupportedEncodings.bfloat16: "replit-code-v1_5-3b-bf16.gguf",
}


@dataclass
class InferenceConfig:
    device: Device = field(default_factory=CPU)
    """Device to run inference upon."""

    weight_path: Optional[Union[str, Path]] = None
    """Path or URL of the model weights."""

    version: SupportedVersions = SupportedVersions.pixtral_12B_2409
    """Pixtral-code version."""

    encoding: SupportedEncodings = SupportedEncodings.float32
    """Weight encoding type."""

    batch_size: int = 1
    """Batch size of inputs to the model."""

    n_heads: int = 16
    """Doc me."""

    seq_len: int = 4096
    """Doc me."""

    casual: bool = True
    """Doc me."""

    alibi: bool = True
    """Doc me."""

    alibi_bias_max: int = 8
    """Doc me."""

    num_blocks: int = 32
    """Doc me."""

    vocab_size: int = 32768
    """Doc me."""

    d_model: int = 3072
    """Doc me."""

    kv_n_heads: int = 8
    """Doc me."""

    max_length = 256

    max_new_tokens = 256

    serialized_model_path: str = None  # type:ignore

    save_to_serialized_model_path: str = None  # type:ignore

    # Pixtral Vision Encoder Configs
    hidden_size: int = 1024
    """Dimension of the hidden representations."""

    intermediate_size = 4096
    """Dimension of the MLP representations."""

    num_hidden_layers = 24
    """Number of hidden layers in the Transformer encoder."""

    num_attention_heads = 16
    """Number of attention heads in the Transformer encoder."""

    num_channels = 3
    """Number of input channels in the input images."""

    image_size = 1024
    """Max dimension of the input images."""

    patch_size = 16
    """Size of the image patches."""

    hidden_act = "gelu"
    """Activation function used in the hidden layers."""

    attention_dropout = 0.0
    """Dropout probability for the attention layers."""

    rope_theta = 10000.0
    """The base period of the RoPE embeddings."""
