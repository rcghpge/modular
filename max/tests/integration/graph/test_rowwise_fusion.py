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
"""Runs reductions the graph compiler generates as rowwise kernels.

Each graph reduces through a registered monoid with ``ops.custom``, fused
with the ops around it, and is checked against numpy on random inputs, so a
kernel that reads or writes the wrong elements fails.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops

_CPU = DeviceRef.CPU()


def _f32(shape: Sequence[int]) -> TensorType:
    return TensorType(DType.float32, shape, device=_CPU)


def _reduce(symbol: str, x: TensorValue, shape: Sequence[int]) -> TensorValue:
    return ops.custom(symbol, device=_CPU, values=[x], out_types=[_f32(shape)])[
        0
    ].tensor


def _sum_inner(x: TensorValue) -> TensorValue:
    return _reduce("mo.reduce.add.tmp", x, [4, 1])


def _sum_outer(x: TensorValue) -> TensorValue:
    return _reduce("mo.reduce.add.tmp", x, [1, 8])


def _max_inner(x: TensorValue) -> TensorValue:
    return _reduce("mo.reduce.max.tmp", x, [4, 1])


def _sum_of_squares_plus_bias(x: TensorValue, b: TensorValue) -> TensorValue:
    return _reduce("mo.reduce.add.tmp", x * x, [4, 1]) + b


_CASES: list[
    tuple[
        str,
        Callable[..., TensorValue],
        list[list[int]],
        Callable[..., np.ndarray],
    ]
] = [
    ("sum_inner", _sum_inner, [[4, 8]], lambda x: x.sum(1, keepdims=True)),
    ("sum_outer", _sum_outer, [[4, 8]], lambda x: x.sum(0, keepdims=True)),
    ("max_inner", _max_inner, [[4, 8]], lambda x: x.max(1, keepdims=True)),
    (
        "sum_of_squares_plus_bias",
        _sum_of_squares_plus_bias,
        [[4, 8], [4, 1]],
        lambda x, b: (x * x).sum(1, keepdims=True) + b,
    ),
]


@pytest.mark.parametrize(
    ("build", "shapes", "reference"),
    [case[1:] for case in _CASES],
    ids=[case[0] for case in _CASES],
)
def test_rowwise_fusion(
    session: InferenceSession,
    build: Callable[..., TensorValue],
    shapes: list[list[int]],
    reference: Callable[..., np.ndarray],
) -> None:
    with Graph("rowwise", input_types=[_f32(s) for s in shapes]) as graph:
        graph.output(build(*(i.tensor for i in graph.inputs)))
    model = session.load(graph)

    rng = np.random.default_rng(0)
    inputs = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    result = model(*(Buffer.from_numpy(a) for a in inputs))[0]
    assert isinstance(result, Buffer)
    np.testing.assert_allclose(
        result.to_numpy(), reference(*inputs), rtol=1e-5, atol=1e-5
    )
