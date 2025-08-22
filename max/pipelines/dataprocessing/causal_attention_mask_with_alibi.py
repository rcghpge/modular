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

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .causal_attention_mask import causal_attention_mask


def _alibi_bias(
    max_seq_len: int, alibi_bias_max: int, n_heads: int
) -> npt.NDArray[np.floating[Any]]:
    # This bias has to be calculated in fp32, as numpy does not have support for bf16.
    bias = np.arange(1 - max_seq_len, 1, 1).reshape((1, 1, 1, max_seq_len))
    rounded_n_heads = int(
        np.power(
            np.asarray(2, dtype=float),
            np.ceil(np.log2(np.asarray(n_heads, dtype=float))),
        )
    )
    m = np.arange(1.0, 1.0 + rounded_n_heads) * (
        np.asarray(alibi_bias_max, dtype=float)
        / np.asarray(rounded_n_heads, dtype=float)
    )

    slopes = np.asarray(1.0, dtype=float) / np.power(
        np.asarray(2.0, dtype=float), m
    )

    if rounded_n_heads != n_heads:
        slopes = np.concatenate(
            [slopes[1:rounded_n_heads:2], slopes[0:rounded_n_heads:2]]
        )
        slopes = slopes[0:n_heads]

    slopes = slopes.reshape(1, n_heads, 1, 1)

    alibi_bias = bias * slopes

    return alibi_bias[:, :, :, :max_seq_len]


def causal_attention_mask_with_alibi(
    original_start_pos: list[int],
    original_seq_len: list[int],
    alibi_bias_max: int,
    n_heads: int,
) -> npt.NDArray[np.floating[Any]]:
    # Get original causal mask
    causal_mask = causal_attention_mask(original_start_pos, original_seq_len)

    max_seq_len = causal_mask.shape[2]

    # Broadcast causal_mask out for n_heads
    causal_mask = np.expand_dims(causal_mask, axis=1)

    # Get alibi bias
    alibi_bias = _alibi_bias(max_seq_len, alibi_bias_max, n_heads)

    return causal_mask + np.float32(alibi_bias)
