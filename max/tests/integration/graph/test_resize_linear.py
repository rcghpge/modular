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
"""Tests that resize_linear runs on GPU and agrees with the CPU result."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from numpy.typing import NDArray

# Two axes change, so the GPU kernel runs two passes through a scratch buffer.
INPUT_SHAPE = [1, 3, 5, 7]
OUTPUT_SHAPE = [1, 3, 9, 3]


def _run(
    session: InferenceSession,
    device: DeviceRef,
    data: NDArray[np.float32],
    coord_mode: int,
    antialias: bool,
) -> NDArray[np.float32]:
    input_type = TensorType(DType.float32, INPUT_SHAPE, device=device)
    with Graph("test_resize_linear", input_types=[input_type]) as graph:
        graph.output(
            ops.resize_linear(
                graph.inputs[0].tensor,
                OUTPUT_SHAPE,
                coordinate_transform_mode=coord_mode,
                antialias=antialias,
            )
        )
    model = session.load(graph)
    result = model.execute(Buffer.from_numpy(data).to(model.input_devices[0]))[
        0
    ]
    assert isinstance(result, Buffer)
    assert result.device == model.input_devices[0]
    return result.to_numpy()


@pytest.mark.skipif(accelerator_count() == 0, reason="No GPU available")
@pytest.mark.parametrize("coord_mode", range(4))
@pytest.mark.parametrize("antialias", [False, True])
def test_resize_linear_gpu_matches_cpu(
    session: InferenceSession, coord_mode: int, antialias: bool
) -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal(INPUT_SHAPE).astype(np.float32)

    cpu = _run(session, DeviceRef.CPU(), data, coord_mode, antialias)
    gpu = _run(session, DeviceRef.GPU(), data, coord_mode, antialias)

    assert gpu.shape == tuple(OUTPUT_SHAPE)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(accelerator_count() == 0, reason="No GPU available")
def test_resize_linear_gpu_same_shape_copies(
    session: InferenceSession,
) -> None:
    """An unchanged shape takes the copy path, which must still write."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal(INPUT_SHAPE).astype(np.float32)

    input_type = TensorType(DType.float32, INPUT_SHAPE, device=DeviceRef.GPU())
    with Graph("test_resize_linear_same", input_types=[input_type]) as graph:
        graph.output(ops.resize_linear(graph.inputs[0].tensor, INPUT_SHAPE))
    model = session.load(graph)
    result = model.execute(Buffer.from_numpy(data).to(model.input_devices[0]))[
        0
    ]
    assert isinstance(result, Buffer)
    np.testing.assert_array_equal(result.to_numpy(), data)
