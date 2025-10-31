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
"""ops.gather tests."""

import pytest
from conftest import (
    axes,
    dims,
    shapes,
    static_dims,
    symbolic_dims,
    tensor_types,
)
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, Shape, StaticDim, TensorType, ops

# gather not meaningful for scalar inputs
input_types = tensor_types(shapes=st.lists(dims, min_size=1))


@given(
    input_type=st.shared(input_types, key="input"),
    indices_type=tensor_types(dtypes=st.just(DType.int64)),
    axis=axes(st.shared(input_types, key="input")),
)
def test_gather(
    input_type: TensorType, indices_type: TensorType, axis: int
) -> None:
    assume(indices_type.rank > 0)
    with Graph("gather", input_types=[input_type, indices_type]) as graph:
        input, indices = graph.inputs[0].tensor, graph.inputs[1].tensor
        out = ops.gather(input, indices, axis)
        if axis < 0:
            axis += input.rank
        target_shape = [
            *input.shape[:axis],
            *indices.shape,
            *input.shape[axis + 1 :],
        ]
        assert out.type == TensorType(input.dtype, target_shape, input.device)
        graph.output(out)


def test_gather_simple() -> None:
    input_type = TensorType(DType.float32, [5, 10], DeviceRef.CPU())
    indices_type = TensorType(DType.int64, [5], DeviceRef.CPU())
    with Graph("gather", input_types=[input_type, indices_type]) as graph:
        input, indices = graph.inputs[0].tensor, graph.inputs[1].tensor
        out = ops.gather(input, indices, axis=-1)
        assert out.type == TensorType(input.dtype, [5, 5], input.device)


@given(
    input_type=st.shared(input_types, key="input"),
    indices_type=tensor_types(dtypes=st.just(DType.int64)),
    axis=...,
)
def test_gather__invalid_axis(
    input_type: TensorType, indices_type: TensorType, axis: int
) -> None:
    assume(indices_type.rank > 0)
    assume(not -input_type.rank <= axis < input_type.rank)
    with Graph("gather", input_types=[input_type, indices_type]) as graph:
        input, indices = graph.inputs[0].tensor, graph.inputs[1].tensor
        with pytest.raises(IndexError):
            ops.gather(input, indices, axis)


@given(input=..., indices=tensor_types(dtypes=st.just(DType.uint64)))
def test_gather_nd(input: TensorType, indices: TensorType) -> None:
    assume(isinstance(indices.shape[-1], StaticDim))
    index_size = int(indices.shape[-1])
    assume(index_size <= input.rank)

    with Graph("gather_nd", input_types=[input, indices]) as graph:
        out = ops.gather_nd(*graph.inputs)  # type: ignore
        assert out.dtype == input.dtype
        assert out.shape == [*indices.shape[:-1], *input.shape[index_size:]]


n_batch_dims = st.shared(st.integers(min_value=0, max_value=10))
batch_dims = st.shared(
    n_batch_dims.flatmap(lambda n: shapes(min_rank=n, max_rank=n))
)
input_types = st.shared(
    batch_dims.flatmap(
        lambda dims: tensor_types().map(
            lambda t: TensorType(t.dtype, dims + t.shape, t.device)
        )
    )
)
indices_types = st.shared(
    batch_dims.flatmap(
        lambda dims: static_dims(max=10).flatmap(
            lambda index_size: tensor_types(dtypes=st.just(DType.uint64)).map(
                lambda t: TensorType(
                    t.dtype, [*dims, *t.shape, index_size], t.device
                )
            )
        )
    )
)


@given(input=input_types, indices=indices_types, batch_dims=n_batch_dims)
def test_gather_nd__batch_dims(
    input: TensorType, indices: TensorType, batch_dims: int
) -> None:
    assert isinstance(indices.shape[-1], StaticDim)
    index_size = int(indices.shape[-1])
    assert 0 <= batch_dims < min(input.rank, indices.rank - 1)
    assert input.shape[:batch_dims] == indices.shape[:batch_dims]
    assume(batch_dims + index_size <= input.rank)

    with Graph("gather_nd", input_types=[input, indices]) as graph:
        out = ops.gather_nd(*graph.inputs, batch_dims=batch_dims)  # type: ignore
        assert out.dtype == input.dtype
        assert out.shape == [
            *input.shape[:batch_dims],
            *indices.shape[batch_dims:-1],
            *input.shape[batch_dims + index_size :],
        ]


@given(input=input_types, indices=indices_types, batch_dims=...)
def test_gather_nd__invalid_batch_dims(
    input: TensorType, indices: TensorType, batch_dims: int
) -> None:
    assert isinstance(indices.shape[-1], StaticDim)
    index_size = int(indices.shape[-1])

    # valid so far, now assume bad batch dim condition
    assume(batch_dims < 0 or index_size + batch_dims > input.rank)

    with Graph("gather_nd", input_types=[input, indices]) as graph:
        with pytest.raises(ValueError):
            ops.gather_nd(*graph.inputs, batch_dims=batch_dims)  # type: ignore


# TODO: what happens if batch_dims + 1 > indices.rank?


@given(
    input=tensor_types(),
    indices=tensor_types(),
    input_batch=shapes(),
    indices_batch=shapes(),
)
def test_gather_nd__mismatching_batch_dims(
    input: TensorType,
    indices: TensorType,
    input_batch: Shape,
    indices_batch: Shape,
) -> None:
    # We require that len(input_batch) >= len(indices_batch), but using assume
    # to enforce that constraint results in a high rejection rate, so we just
    # swap the two.
    if len(input_batch) < len(indices_batch):
        input_batch, indices_batch = indices_batch, input_batch

    for i in range(len(indices_batch)):
        assume(input_batch[i] != indices_batch[i])

    input_with_batch = TensorType(
        input.dtype, [*input_batch, *input.shape], input.device
    )
    indices_with_batch = TensorType(
        indices.dtype, [*indices_batch, *indices.shape], indices.device
    )

    with Graph(
        "gather_nd", input_types=[input_with_batch, indices_with_batch]
    ) as graph:
        with pytest.raises(ValueError):
            ops.gather_nd(*graph.inputs, batch_dims=len(input_batch))  # type: ignore


@given(
    input=input_types,
    indices=tensor_types(
        shapes=shapes().flatmap(
            lambda shape: symbolic_dims.map(lambda dim: Shape([*shape, dim]))
        )
    ),
    batch_shape=shapes(min_rank=1, max_rank=3),
)
def test_gather_nd__symbolic_index(
    input: TensorType, indices: TensorType, batch_shape: Shape
) -> None:
    input = TensorType(input.dtype, [*batch_shape, *input.shape], input.device)
    indices = TensorType(
        indices.dtype, [*batch_shape, *indices.shape], indices.device
    )

    with Graph("gather_nd", input_types=[input, indices]) as graph:
        with pytest.raises(ValueError):
            ops.gather_nd(*graph.inputs, batch_dims=len(batch_shape))  # type: ignore


@given(input=input_types, indices=indices_types, batch_dims=n_batch_dims)
def test_gather_nd__index_too_long(
    input: TensorType, indices: TensorType, batch_dims: int
) -> None:
    assert isinstance(indices.shape[-1], StaticDim)
    assert 0 <= batch_dims < min(input.rank, indices.rank - 1)

    index_size = int(indices.shape[-1])
    # error condition
    assume(index_size + batch_dims > input.rank)

    with Graph("gather_nd", input_types=[input, indices]) as graph:
        with pytest.raises(ValueError):
            ops.gather_nd(*graph.inputs, batch_dims=batch_dims)  # type: ignore


@given(
    input=input_types,
    indices=indices_types,
    batch_dims=n_batch_dims,
    dtype=st.sampled_from([dtype for dtype in DType if dtype.is_float()]),
)
def test_gather_nd__non_int_indices(
    input: TensorType,
    indices: TensorType,
    batch_dims: int,
    dtype: DType,
) -> None:
    assert isinstance(indices.shape[-1], StaticDim)
    assert 0 <= batch_dims < min(input.rank, indices.rank - 1)
    index_size = int(indices.shape[-1])
    assume(index_size + batch_dims <= input.rank)

    # error condition
    assert dtype.is_float()
    indices.dtype = dtype

    with Graph("gather_nd", input_types=[input, indices]) as graph:
        with pytest.raises(ValueError):
            ops.gather_nd(*graph.inputs, batch_dims=batch_dims)  # type: ignore


@given(
    input_type=st.shared(input_types, key="input"),
    indices_type=tensor_types(dtypes=st.just(DType.int64)),
    axis=axes(st.shared(input_types, key="input")),
)
def test_gather__mismatched_devices(
    input_type: TensorType, indices_type: TensorType, axis: int
) -> None:
    assume(indices_type.rank > 0)
    device = DeviceRef.GPU(1)
    assume(input_type.device != device)
    indices_type = TensorType(indices_type.dtype, indices_type.shape, device)
    with Graph("gather_nd", input_types=[input_type, indices_type]) as graph:
        input, indices = graph.inputs
        with pytest.raises(ValueError, match="same device"):
            ops.gather(input.tensor, indices.tensor, axis)


@given(input=..., indices=tensor_types(dtypes=st.just(DType.uint64)))
def test_gather_nd__mismatched_devices(
    input: TensorType, indices: TensorType
) -> None:
    device = DeviceRef.GPU(1)
    assume(input.device != device)
    indices = TensorType(indices.dtype, indices.shape, device)
    with Graph("gather_nd", input_types=[input, indices]) as graph:
        with pytest.raises(ValueError, match="same device"):
            ops.gather_nd(*graph.inputs)  # type: ignore
