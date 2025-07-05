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

from typing import cast

import numpy as np
import pytest
import torch
from max.driver import Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

device_ref = DeviceRef.GPU() if accelerator_count() > 0 else DeviceRef.CPU()


@pytest.mark.parametrize(
    ("input_shape", "k", "axis"),
    [
        ((5,), 3, -1),
        ((4, 3), 2, 0),
        ((2, 3, 4), 1, 1),
        ((6,), 5, 0),
        ((3, 5), 4, -1),
    ],
)
def test_top_k_execution(
    session: InferenceSession, input_shape: tuple[int, ...], k: int, axis: int
) -> None:
    """Tests end-to-end top_k lowering and execution against torch.topk."""

    if accelerator_count() > 0 and axis != -1:
        pytest.skip("Only -1 axis supported on GPU")

    graph = Graph(
        "top_k_test",
        forward=lambda x: ops.top_k(x, k=k, axis=axis),
        input_types=[
            TensorType(DType.float32, shape=input_shape, device=device_ref)
        ],
    )

    # Compile and execute the graph.
    model = session.load(graph)

    # Generate random input data.
    np_input = np.random.randn(*input_shape).astype(np.float32)
    torch_input = torch.from_numpy(np_input)

    # Execute MAX model.
    max_values, max_indices = model.execute(
        Tensor.from_numpy(np_input).to(model.input_devices[0])
    )

    # Get torch reference results.
    torch_values, torch_indices = torch.topk(
        torch_input,
        k=k,
        dim=axis,
        sorted=True,
        largest=True,  # MAX currently implements largest=True semantics.
    )

    # Verify values match.
    np.testing.assert_allclose(
        cast(Tensor, max_values).to_numpy(), torch_values.numpy(), rtol=1e-5
    )

    # For indices verification, check that gathered values match.
    # This handles potential index ordering differences when sorted=False
    # (once this test supports that).
    np.testing.assert_array_equal(
        np.take_along_axis(
            np_input, cast(Tensor, max_indices).to_numpy(), axis=axis
        ),
        np.take_along_axis(np_input, torch_indices.numpy(), axis=axis),
    )


def test_top_k_invalid_k(session: InferenceSession) -> None:
    """Verifies error when k exceeds dimension size along axis."""
    input_shape = (3,)
    # Larger than dimension size 3.
    invalid_k = 4
    axis = 0

    graph = Graph(
        "top_k_invalid_test",
        forward=lambda x: ops.top_k(x, k=invalid_k, axis=axis),
        input_types=[
            # Circumvent static checks with a symbolic dim to make sure the
            # runtime checks fire.
            TensorType(DType.float32, shape=("dim",), device=device_ref)
        ],
    )

    # Compile and init the graph.
    model = session.load(graph)

    with pytest.raises(
        ValueError,
        match=r"k value exceeds dimension size along specified axis",
    ):
        model.execute(
            Tensor.from_numpy(np.ones(input_shape, dtype=np.float32)).to(
                model.input_devices[0]
            )
        )
