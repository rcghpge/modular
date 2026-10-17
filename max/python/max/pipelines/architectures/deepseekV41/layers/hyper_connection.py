# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

"""Collapses the hyper-connection residual streams into one hidden state.

The model has no dedicated weights for this step: the last layer's FFN
pre-mix weights combine the parallel residual streams.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import TensorValue, ops


def final_collapse(
    full_stream: TensorValue, pre_mix: TensorValue
) -> TensorValue:
    """Weighted-sums ``hc_mult`` parallel residual streams down to one.

    Args:
        full_stream: ``[..., hc_mult, hidden_size]`` residual streams after
            the last decoder layer's FFN output is written back into them
            (not the bare FFN output).
        pre_mix: ``[..., hc_mult]`` per-stream weights: the last layer's FFN
            pre-mix.

    Returns:
        ``[..., hidden_size]``, in ``full_stream``'s dtype.
    """
    streams = ops.cast(full_stream, DType.float32)
    weights = ops.cast(pre_mix, DType.float32)
    collapsed = ops.sum(ops.unsqueeze(weights, -1) * streams, axis=-2)
    collapsed = ops.squeeze(collapsed, -2)
    return ops.cast(collapsed, full_stream.dtype)
