# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Mask configuration for attention."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MHAMaskVariant(str, Enum):
    CAUSAL_MASK = 0
    CAUSAL_ALIBI_MASK = 1
    NULL_MASK = 2
    CHUNKED_CAUSAL_MASK = 3
    SLIDING_WINDOW_CAUSAL_MASK = 4


@dataclass
class MHAMaskConfig:
    attention_mask_variant: AttentionMaskVariant
    positional_encoding_variant: PositionalEncodingVariant


class AttentionMaskVariant(str, Enum):
    NULL_MASK = "null"
    CAUSAL_MASK = "causal"
    TENSOR_MASK = "tensor_mask"
    CHUNKED_CAUSAL_MASK = "chunked_causal"
    SLIDING_WINDOW_CAUSAL_MASK = "sliding_window_causal"


class PositionalEncodingVariant(str, Enum):
    NO_POS = "no_pos"
    ALIBI_POS = "alibi_pos"
