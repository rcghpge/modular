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
"""Op implementation for while loop."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable

from max import mlir
from max._core import Value as _Value
from max.dtype import DType
from max.mlir.dialects import mo

from ..graph import DeviceRef, Graph
from ..type import TensorType
from ..value import BufferValue, TensorValue, Value


def while_loop(
    initial_values: Iterable[Value] | Value,
    predicate: Callable[..., TensorValue],
    body: Callable[..., Iterable[Value]],
) -> list[TensorValue]:
    """Execute a loop until the predicate evaluates to false.

    Both the predicate and body functions must take in as arguments the same
    number and types of values as specified in the init_args. The predication
    function must return only a boolean scalar tensor of type :class:`DType.bool`.
    The body function must return a list of values matching the types of init_args.

    The following example demonstrates a basic while loop with a single argument:

    .. code-block:: python

        from max.graph import Graph, ops
        from max.dtype import DType

        with Graph("while_loop_example") as g:
            x = ops.constant(0, dtype=DType.int32, device=DeviceRef.CPU())

            def pred(x):
                return x < 10

            def body(x):
                return x + 1

            result = ops.while_loop(x, pred, body)
            print(result)

    The following example shows a while loop with multiple arguments:

    .. code-block:: python

        from max.graph import Graph, ops
        from max.dtype import DType

        with Graph("while_loop_example") as g:
            x = ops.constant(0, dtype=DType.int32, device=DeviceRef.CPU())
            y = ops.constant(5, dtype=DType.int32, device=DeviceRef.CPU())

            def pred(x, y):
                return ops.logical_and(x < 10, y < 15)

            def body(x, y):
                return [x + 1, y + 1]

            results = ops.while_loop((x, y), pred, body)
            print(results)

    Args:
        initial_values:
            Initial values for loop arguments. Must be non-empty.

        predicate:
            Callable that takes loop arguments and returns a boolean scalar tensor
            of type :class:`DType.bool` determining loop continuation.

        body:
            Callable that takes loop arguments and returns updated values matching
            the types of init_args.

    Returns:
        List of output values from the final loop iteration.

    Raises:
        ValueError: If init_args is empty.
        NotImplementedError: If any init_arg is a :class:`BufferValue`.

    Note:
        Buffer operations are currently not supported.
    """

    initial_values = (
        list(initial_values)
        if isinstance(initial_values, Iterable)
        else [initial_values]
    )
    if not initial_values:
        raise ValueError("While loops must have at least one iteration value.")

    # Add execution chain to initial values to track side effects across iterations
    initial_values = [*initial_values, Graph.current._current_chain]

    # Temporary restriction until buffer support is implemented
    if any(isinstance(arg, BufferValue) for arg in initial_values):
        raise NotImplementedError(
            "Buffer operations are currently not supported in while loops"
        )

    # Determine output types including the execution chain as last element
    # Chain is required for sequencing side-effecting operations like buffer accesses
    out_types = [arg.type for arg in initial_values]
    out_mlir_types = [t.to_mlir() for t in out_types]

    def wrap_while_block_function(
        user_func,
        block_args: Iterable[mlir.BlockArgument],
        is_cond_block: bool = False,
    ):
        """Adapts a user-provided loop function to handle execution chain state.

        This wrapper handles the implicit execution chain that tracks operation
        ordering in while loops. The chain is always passed as the last argument
        to loop blocks but should not be exposed to users writing loop logic.

        Args:
            user_func: The user's predicate or body function that operates on
                     loop variables only.
            block_args: The block arguments from the while loop operation, which
                      include both loop variables and the execution chain as
                      the last element.

        Returns:
            A function that properly manages the execution chain state before
            invoking the user's function with just the loop variables.

        .. note::
            The execution chain is tracked differently in while loops vs conds:
            - While loops are re-entrant and must pass the chain through iterations.
            - Cond blocks create new chain SSAs from the parent scope.
            This chain management ensures proper ordering of side-effecting ops
            across loop iterations.
        """

        def chain_aware_wrapper():
            # Separate loop variables from the execution chain
            loop_vars: Sequence[Value]
            execution_chain: Value
            *loop_vars, execution_chain = (
                Value.from_mlir(_Value._from_cmlir(arg)) for arg in block_args
            )

            # Update the graph's chain state before running user code
            Graph.current._update_chain(execution_chain)

            # Invoke user function with only the loop variables
            result = user_func(*loop_vars)

            # The cond block function returns a boolean tensor, but mo.while
            # expects the cond block's yield operation to yield all loop
            # carried values, so add loop_vars to the result list when building
            # the cond block.
            if is_cond_block:
                # Condition is expected to be on CPU
                result = result.to(DeviceRef.CPU())
                return [result] + loop_vars
            else:
                return result

        return chain_aware_wrapper

    # Create while loop operation with chain-aware signature
    # The chain is passed as implicit final operand/result for state management
    with Graph.current._pause_verification():
        results, while_op = Graph.current._add_op_get_op_with_results(
            mo.while_, out_mlir_types, initial_values
        )

    # Separate actual loop results from the execution chain
    *results, out_chain = results

    def while_condition_op(args) -> mlir.OpView:
        """Adaptor for mo.WhileConditionOp, whose constructor takes the
        condition value and the list of yielded values."""
        condition, *results = args
        return mo.WhileConditionOp(condition, results)

    try:
        pred_block = while_op.condRegion.blocks[0]
        pred_wrapped_fn = wrap_while_block_function(
            predicate, pred_block.arguments, is_cond_block=True
        )

        Graph.current._build_block(
            pred_block,
            pred_wrapped_fn,
            while_condition_op,
            "pred_block",
            [TensorType(DType.bool, [], DeviceRef.CPU())] + out_types[:-1],
        )

        body_block = while_op.bodyRegion.blocks[0]
        body_wrapped_fn = wrap_while_block_function(body, body_block.arguments)
        Graph.current._build_block(
            body_block,
            body_wrapped_fn,
            mo.YieldOp,
            "body_block",
            out_types[:-1],
        )

        Graph.current._update_chain(out_chain)
        Graph.current._verify_op(while_op)
        return results
    except Exception as e:
        while_op.erase()
        raise e
