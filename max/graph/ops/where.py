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
"""Op implementation for where."""

from max.dtype import DType
from max.mlir.dialects import rmo

from .. import dtype_promotion
from ..graph import Graph
from ..value import TensorValue, TensorValueLike


def where(
    condition: TensorValueLike, x: TensorValueLike, y: TensorValueLike
) -> TensorValue:
    """
    Returns ``condition ? x : y`` (element-wise), where ``cond``, ``x`` and ``y``
    are input tensors.

    Args:
        condition: The condition tensor to use for selecting elementwise
                   values. This tensor must have a boolean dtype.
        x: If the condition is true at a position, the value from the same
           position in this tensor will be selected.
        y: If the condition is false at a position, the value from the same
           position in this tensor will be selected.

    Returns:
        A new symbolic tensor holding either values from either ``x`` or ``y``,
        based on the elements in `condition`.
    """
    condition = TensorValue(condition)
    if condition.dtype != DType.bool:
        raise ValueError(
            f"Expected condition to be a boolean tensor, but got a tensor with dtype {condition.dtype}"
        )

    # If the inputs are tensors, check that all tensors are on the same device
    if isinstance(x, TensorValue) and isinstance(y, TensorValue):
        devices = [t.type.device for t in [condition, x, y]]
        if not all(d == devices[0] for d in devices):
            raise ValueError(
                f"All tensors must be on the same device, but got devices: {', '.join(str(d) for d in devices)}"
            )

    x, y = dtype_promotion._promote_weak_dtypes(x, y)
    return Graph.current._add_op(rmo.select, condition, x, y)[0].tensor
