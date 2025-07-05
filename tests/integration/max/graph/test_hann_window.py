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

import platform

import numpy as np
import pytest
import torch
from max.driver import accelerator_count
from max.dtype import DType
from max.graph import DeviceRef, Graph, ops


@pytest.mark.parametrize("window_length", [0, 1, 2, 5, 10, 100])
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])
def test_hann_window(
    session, window_length: int, periodic: bool, dtype: DType
) -> None:
    """Test hann_window against PyTorch's implementation."""
    if dtype == DType.bfloat16 and platform.machine() in ["arm64", "aarch64"]:
        pytest.skip("BF16 is not supported on ARM CPU architecture")

    # Get PyTorch's implementation
    torch_dtype = torch.float32 if dtype == DType.float32 else torch.bfloat16
    torch_window = (
        torch.hann_window(window_length, periodic=periodic, dtype=torch_dtype)
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
    )

    # Get our implementation
    with Graph(f"hann_window_{dtype}", input_types=[]) as graph:
        out = ops.hann_window(
            window_length,
            device=(
                DeviceRef.GPU() if accelerator_count() > 0 else DeviceRef.CPU()
            ),
            periodic=periodic,
            dtype=dtype,
        )
        graph.output(out.cast(DType.float32))

    model = session.load(graph)
    max_window = model()[0].to_numpy()

    # Compare shapes
    assert max_window.shape == torch_window.shape, (
        f"Shape mismatch: got {max_window.shape}, expected {torch_window.shape}"
    )

    # Compare values with a small tolerance
    np.testing.assert_allclose(
        max_window,
        torch_window,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Values mismatch for window_length={window_length}, periodic={periodic}",
        verbose=True,
    )
