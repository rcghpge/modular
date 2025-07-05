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

import operator

import numpy as np
import pytest
from max.driver import Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, StaticDim, TensorType

device_ref = DeviceRef.GPU() if accelerator_count() > 0 else DeviceRef.CPU()


@pytest.mark.parametrize(
    ("tensor_type", "indices"),
    [
        # x[1:]
        (
            TensorType(DType.float32, shape=["dim0"], device=device_ref),
            (slice(1, None),),
        ),
        (
            TensorType(
                DType.float32, shape=["dim0", "dim1"], device=device_ref
            ),
            (slice(1, None),),
        ),
        # x[:-1]
        (
            TensorType(DType.float32, shape=["dim0"], device=device_ref),
            (slice(None, -1)),
        ),
        # x[-1:]
        (
            TensorType(DType.float32, shape=["dim0"], device=device_ref),
            (slice(-1, None)),
        ),
        # x[::2]
        (
            TensorType(DType.float32, shape=["dim0"], device=device_ref),
            (slice(None, None, 2),),
        ),
        # x[::-1]
        # TODO(AIPIPE-109): allow negative step after improving rmo.slice.
        # (TensorType(DType.float32, shape=["dim0"]), (slice(None, None, -1),)),
        # x[:, None, :]
        (
            TensorType(
                DType.float32, shape=["dim0", "dim1"], device=device_ref
            ),
            (slice(None), None, slice(None)),
        ),
        # x[..., None]
        (
            TensorType(
                DType.float32, shape=["dim0", "dim1"], device=device_ref
            ),
            (Ellipsis, None),
        ),
        # x[..., 1]
        (
            TensorType(
                DType.float32,
                shape=["dim0", "dim1", "dim2"],
                device=device_ref,
            ),
            (Ellipsis, 1),
        ),
        # x[Ellipsis, 1:]
        (
            TensorType(
                DType.float32, shape=["dim0", "dim1"], device=device_ref
            ),
            (Ellipsis, slice(1, None)),
        ),
        # x[1, ..., ::-1]
        # TODO(AIPIPE-109): allow negative step after improving rmo.slice.
        # (
        #     TensorType(DType.float32, shape=["dim0", "dim1", "dim2"]),
        #     (1, Ellipsis, slice(None, None, -1)),
        # ),
        # x[:, -1]
        (
            TensorType(
                DType.float32,
                shape=["dim0", "dim1", "dim2"],
                device=device_ref,
            ),
            (slice(None), -1),
        ),
    ],
)
def test_slice_numpy(
    session: InferenceSession, tensor_type: TensorType, indices: tuple[slice]
) -> None:
    """Tests end-to-end slice lowering and execution."""
    graph = Graph(
        "slice",
        forward=operator.itemgetter(indices),
        input_types=[tensor_type],
    )

    # Compile and execute the slice graph.
    model = session.load(graph)

    # Compute a random input with shape compatible with tensor_type.
    input_shape = [
        idx * 3 + 7 if not isinstance(dim, StaticDim) else dim.dim
        for idx, dim in enumerate(tensor_type.shape)
    ]
    input_array = np.random.randn(*input_shape).astype(
        tensor_type.dtype.to_numpy()
    )

    # Run the slice graph.
    out = model.execute(
        Tensor.from_numpy(input_array).to(model.input_devices[0])
    )
    assert isinstance(out[0], Tensor)
    sliced = out[0].to_numpy()

    # Verify that the max.graph slicing matches NumPy.
    expected = input_array[indices]
    np.testing.assert_array_equal(sliced, expected)
