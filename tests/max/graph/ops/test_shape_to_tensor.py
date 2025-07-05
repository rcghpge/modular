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

"""Test the shape_to_tensor operation."""

import pytest
from conftest import shapes, static_dims
from hypothesis import given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import Graph, ops


def test_shape_to_tensor_invalid_type() -> None:
    """Test that shape_to_tensor fails with non-shape inputs."""
    with Graph("shape_to_tensor_invalid") as graph:
        with pytest.raises(TypeError):
            ops.shape_to_tensor(None)

        with pytest.raises(ValueError):
            ops.shape_to_tensor("not a shape")

        with pytest.raises(TypeError):
            ops.shape_to_tensor(42)


shared_static_shapes = st.shared(shapes(dims=static_dims()))


@given(shape=shared_static_shapes)
def test_shape_to_tensor_valid(shape: list[int]) -> None:
    """Test that shape_to_tensor works with valid shape inputs."""
    with Graph("shape_to_tensor_valid") as graph:
        out = ops.shape_to_tensor(shape)
        assert out.dtype == DType.int64
        assert out.rank == 1
        assert out.shape[0] == len(shape)
        graph.output(out)
