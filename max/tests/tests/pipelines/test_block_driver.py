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
"""The block spec-decode driver's phases that every block draft shares."""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.speculative.block_driver import block_dispatch_metadata


def test_block_dispatch_metadata_is_rebuilt_at_block_width() -> None:
    """The rebuilt 4-int buffer keeps the batch size and cache bound, sets
    ``q_max_seq_len`` to the block width, and zeroes ``num_partitions`` so
    the kernel recomputes the split-K count for the draft's own heads.

    The leaf's own buffer carries the verify width, which on a prefill batch
    is the prompt length; an oversized query bound drove the draft's first
    attention layer to NaN.
    """
    with Graph(
        "block_dispatch_metadata",
        input_types=(TensorType(DType.int64, [4], DeviceRef.CPU()),),
    ) as graph:
        (meta,) = graph.inputs
        graph.output(block_dispatch_metadata(meta.tensor, 8))

    compiled = InferenceSession(devices=[CPU()]).load(graph)
    (out,) = compiled.execute(
        Buffer.from_numpy(np.array([3, 129, 5, 4321], dtype=np.int64))
    )
    assert isinstance(out, Buffer)
    np.testing.assert_array_equal(out.to_numpy(), [3, 8, 0, 4321])
