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
"""Test the max.graph Python bindings."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import mock

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from max import mlir
from max._core import graph as _graph
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.graph.graph import _location
from max.mlir.dialects import rmo

empty_graphs = st.builds(
    Graph, st.text(), input_types=st.lists(st.from_type(TensorType))
)


@given(graph=empty_graphs)
def test_simple_graphs(graph: Graph) -> None:
    assume(len(graph.inputs) > 0)
    with graph:
        graph.output(graph.inputs[0])


def test_mlir_module_create() -> None:
    """Tests whether we can import mlir and create a Module.

    This is a basic smoke test for max.graph Python packaging.
    """
    with mlir.Context(), mlir.Location.unknown():
        _ = mlir.Module.create()


def test_elementwise_add_graph() -> None:
    """Builds a simple graph with an elementwise addition and checks the IR."""
    with Graph(
        "elementwise_add",
        input_types=[
            TensorType(
                dtype=DType.float32,
                shape=["batch", "channels"],
                device=DeviceRef.CPU(),
            )
        ],
    ) as graph:
        graph.output(graph.inputs[0] + 1)


def test_elementwise_add_graph_with_device_prop() -> None:
    """Builds a simple graph with explicit device on inputs and checks for output device propagation in the IR."""
    with Graph(
        "elementwise_add",
        input_types=[
            TensorType(
                dtype=DType.float32,
                shape=["batch", "channels"],
                device=DeviceRef.GPU(0),
            ),
            TensorType(
                dtype=DType.float32,
                shape=["batch", "channels"],
                device=DeviceRef.GPU(0),
            ),
        ],
    ) as graph:
        graph.output(graph.inputs[0] + graph.inputs[1])
        # Ensure input tensor has cuda
        for input in graph.inputs:
            assert "gpu" in str(input)
        # Ensure output tensor has cuda propagated
        assert " -> !mo.tensor<[batch, channels], f32, gpu:0>" in str(
            graph._mlir_op
        )


def test_transpose_graph_with_device_prop() -> None:
    """Builds a simple graph with an transpose and checks the IR."""
    with Graph(
        "elementwise_add",
        input_types=[
            TensorType(
                dtype=DType.float32,
                shape=["batch", "channels"],
                device=DeviceRef.GPU(0),
            )
        ],
    ) as graph:
        graph.output(ops.transpose(graph.inputs[0], -1, -2))
        for input in graph.inputs:
            assert "gpu" in str(input)
        assert " -> !mo.tensor<[channels, batch], f32, gpu:0>" in str(
            graph._mlir_op
        )


def test_location() -> None:
    with Graph("location") as graph:

        def elided():
            return _location(ignore_frames=1)

        def foo():
            return elided()

        loc = foo()

        frames = _graph.get_frame(loc)
        assert "foo" == frames[-1].name
        assert "test_location" == frames[-2].name


def test_location_no_stack() -> None:
    with Graph("location") as graph:
        with mock.patch("traceback.extract_stack") as mock_stack:
            mock_stack.return_value = []

            loc = _location()
            assert loc == mlir.Location.unknown()


def test_add_op() -> None:
    """Builds a simple graph with an elementwise addition and checks the IR."""
    input_type = TensorType(
        dtype=DType.float32, shape=["batch", "channels"], device=DeviceRef.CPU()
    )
    with Graph("add", input_types=(input_type, input_type)) as graph:
        lhs, rhs = graph.inputs
        elemwise_sum = ops.add(lhs, rhs)
        graph.output(elemwise_sum)

        # Check that the arg/result name attributes were added.
        assert "argument_names = " in str(graph._mlir_op)
        assert "result_names = " in str(graph._mlir_op)


def test_add_op_closure() -> None:
    """Uses a closure to build a simple graph with an elementwise addition
    and checks the IR.
    """

    def elementwise_add(lhs: TensorValue, rhs: TensorValue) -> TensorValue:
        return ops.add(lhs, rhs)

    input_type = TensorType(
        dtype=DType.float32, shape=["batch", "channels"], device=DeviceRef.CPU()
    )
    add_graph = Graph("add", elementwise_add, (input_type, input_type))

    assert "rmo.add" in str(add_graph._mlir_op)
    assert "mo.output" in str(add_graph._mlir_op)


def test_invalid_operand() -> None:
    """Test that passing an invalid operand raises an error."""
    with Graph(
        "invalid_operand",
        input_types=[
            TensorType(DType.int64, [2], device=DeviceRef.CPU()),
        ],
    ) as graph:
        input_tensor = graph.inputs[0]
        with pytest.raises(ValueError):
            Graph.current._add_op(rmo.add, [2, 5], input_tensor)
        graph.output(input_tensor)


def test_load_from_file() -> None:
    """Tests printing to and loading from a file."""
    graph = Graph(
        "identity",
        forward=lambda x: x,
        input_types=[TensorType(DType.int64, [1], device=DeviceRef.CPU())],
    )
    with NamedTemporaryFile("w") as mlir_text_file:
        # Flush so that the subsequent read works.
        print(graph, file=mlir_text_file, flush=True)
        loaded_graph = Graph("loaded", path=Path(mlir_text_file.name))

    assert isinstance(
        loaded_graph._mlir_op, (mlir.Operation, mlir.OpView)
    ) and (str(loaded_graph) == str(graph))
