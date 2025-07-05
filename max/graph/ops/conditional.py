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
"""Op implementation for conditional."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

from max.mlir.dialects import mo

from ..graph import Graph
from ..type import DeviceRef, Type, _ChainType
from ..value import TensorValue, TensorValueLike


def cond(
    pred: TensorValueLike,
    out_types: Iterable[Type] | None,
    then_fn: Callable,
    else_fn: Callable,
) -> list[TensorValue]:
    """Conditionally execute one of two branches based on a boolean predicate.

    Both branches must return the same number and types of values as specified
    in ``out_types``. Buffer mutations in branches are tracked automatically
    through the chain mechanism.

    Examples:

    1. Basic conditional with return values:

        .. code-block:: python

            def then_fn():
                return ops.constant(1, DType.int32, device=DeviceRef.CPU())
            def else_fn():
                return ops.constant(0, DType.int32, device=DeviceRef.CPU())

            result = ops.cond(
                pred,
                [TensorType(DType.int32, [], device=device)],
                then_fn,
                else_fn
            )

    2. Conditional with buffer mutations:

        .. code-block:: python

            def then_fn():
                ops.inplace_custom("increment", device=buffer.device, values=[buffer])
            def else_fn():
                ops.inplace_custom("decrement", device=buffer.device, values=[buffer])

            ops.cond(pred, None, then_fn, else_fn)

    ::
    Args:
        pred:
            Boolean scalar tensor of type :obj:`DType.bool` determining branch execution

        out_types:
            Expected output types for both branches. Use :obj:`None` for branches that don't return values

        then_fn:
            Callable executed when ``pred`` is True. Must return values matching ``out_types`` if ``out_types`` is not :obj:`None`

        else_fn:
            Callable executed when ``pred`` is False. Must return values matching ``out_types`` if ``out_types`` is not :obj:`None`

    Returns:
        List of output values from executed branch. Returns empty list when ``out_types``
        is :obj:`None`

    Raises:
        ValueError: If branches return different numbers of results or result types
                  don't match ``out_types``

    Note:
        Buffer operations in branches automatically update the global chain state to
        maintain mutation ordering constraints
    """
    pred = TensorValue(pred)
    pred = pred.to(DeviceRef.CPU())
    out_types_actual = [
        *(t.to_mlir() for t in out_types or []),
        _ChainType().to_mlir(),
    ]

    # Pause verification until the operation is fully constructed
    with Graph.current._pause_verification():
        results, if_op = Graph.current._add_op_get_op_with_results(
            mo.if_, pred, out_types_actual
        )

    results_len = len(results)
    out_chain = results[results_len - 1]
    results = results[: results_len - 1]

    try:
        Graph.current._build_block(
            if_op.thenRegion.blocks[0],
            then_fn,
            mo.YieldOp,
            "then_block",
            out_types,
        )

        Graph.current._build_block(
            if_op.elseRegion.blocks[0],
            else_fn,
            mo.YieldOp,
            "else_block",
            out_types,
        )

        Graph.current._update_chain(out_chain)
        Graph.current._verify_op(if_op)
        return results
    except Exception as e:
        if_op.erase()
        raise e
