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
"""ops.reduction tests."""

import pytest
from conftest import axes, shapes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import Graph, ops

shared_shapes = st.shared(shapes(min_rank=1))

SAME_TYPE_REDUCTIONS = [ops.min, ops.max, ops.mean, ops.sum]
INDEX_REDUCTIONS = [ops.argmax, ops.argmin]
ALL_REDUCTIONS = SAME_TYPE_REDUCTIONS + INDEX_REDUCTIONS


@given(
    op=st.sampled_from(SAME_TYPE_REDUCTIONS),
    input_type=tensor_types(shapes=shared_shapes),
    axis=axes(shared_shapes),
)
def test_reduction__same_type(op, input_type, axis) -> None:
    with Graph("test_reduction", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        result = op(x, axis=axis)
        expected_shape = list(input_type.shape)
        expected_shape[axis] = 1
        assert result.shape == expected_shape
        assert result.dtype == input_type.dtype


@given(
    op=st.sampled_from(INDEX_REDUCTIONS),
    input_type=tensor_types(shapes=shared_shapes),
    axis=axes(shared_shapes),
)
def test_reduction__index(op, input_type, axis) -> None:
    with Graph("test_reduction", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        result = op(x, axis=axis)
        expected_shape = list(input_type.shape)
        expected_shape[axis] = 1
        assert result.shape == expected_shape
        assert result.dtype == DType.int64


@given(
    op=st.sampled_from(ALL_REDUCTIONS),
    input_type=tensor_types(shapes=shapes(min_rank=0, max_rank=0)),
    axis=st.integers(),
)
def test_reduction_fails__zero_rank_input(op, input_type, axis) -> None:
    with Graph("test_reduction", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        with pytest.raises(ValueError):
            op(x, axis=axis)


@given(
    op=st.sampled_from(ALL_REDUCTIONS),
    input_type=tensor_types(shapes=shared_shapes),
    axis=st.integers(),
)
def test_reduction_fails__axis_out_of_bounds(op, input_type, axis) -> None:
    assume(not -input_type.rank <= axis < input_type.rank)
    with Graph("test_reduction", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        with pytest.raises(ValueError):
            op(x, axis=axis)
