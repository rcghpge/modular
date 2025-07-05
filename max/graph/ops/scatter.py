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
"""Op implementation for scatter."""

from max.dtype import DType
from max.mlir.dialects import rmo

from .. import dtype_promotion
from ..dim import DimLike
from ..graph import Graph
from ..type import DeviceRef, TensorType
from ..value import TensorValue, TensorValueLike
from .constant import constant
from .nonzero import nonzero


def scatter(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
    axis: int = -1,
) -> TensorValue:
    """
    Creates a new symbolic tensor where the updates are written to input according to indices.

    Args:
        input: The input symbolic tensor to write elements to.
        updates: A symbolic tensor of elements to write to input.
        indices: The positions in input to update.
        axis: The axis along which indices indexes into.

    Returns:
        A new symbolic tensor representing the result of the scatter operation.
    """

    input = TensorValue(input)

    if not (-input.rank <= axis < input.rank):
        raise ValueError(
            f"Invalid axis value {axis}. Axis must be in range [-{input.rank}, {input.rank - 1}]"
        )

    updates = TensorValue(updates)
    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype '{input.dtype}' and updates dtype '{updates.dtype}' must match."
        )

    indices = TensorValue(indices)
    if indices.dtype not in [DType.int32, DType.int64]:
        raise ValueError(
            f"Invalid indices dtype: '{indices.dtype}'. Indices must be of type int32 or int64."
        )

    axis_constant = constant(axis, DType.int64, DeviceRef.CPU())

    # TODO(GEX-2197): Support scatter on GPU
    old_device = input.device
    input = input.to(DeviceRef.CPU())
    updates = updates.to(DeviceRef.CPU())
    indices = indices.to(DeviceRef.CPU())

    return Graph.current._add_op(
        rmo.mo_scatter,
        input.type.to_mlir(),
        input,
        updates,
        indices,
        axis_constant,
    )[0].tensor.to(old_device)


def scatter_nd(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
) -> TensorValue:
    """
    Creates a new symbolic tensor where the updates are scattered into input at specified indices.

    Args:
        input: The input symbolic tensor to write elements to.
        updates: A symbolic tensor of elements to write to input.
        indices: A tensor of indices specifying where to write updates.
            Shape should be [num_updates, rank] for full indexing or
            [num_updates, k] for partial indexing where k < rank.

    Returns:
        A new symbolic tensor representing the result of the scatter_nd operation.
    """
    input = TensorValue(input)
    updates = TensorValue(updates)
    indices = TensorValue(indices)

    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype ({input.dtype}) and updates dtype"
            f" ({updates.dtype}) must match"
        )

    if indices.dtype not in (DType.int32, DType.int64):
        raise ValueError(
            f"Invalid indices dtype: '{indices.dtype}'. Indices must be of type int32 or int64."
        )

    # Check that all tensors are on the same device
    if not (input.device == updates.device == indices.device):
        raise ValueError(
            f"All tensors must be on the same device. Got input.device={input.device}, "
            f"updates.device={updates.device}, indices.device={indices.device}"
        )

    return Graph.current._add_op(
        rmo.mo_scatter_nd,
        TensorType(input.dtype, input.shape, input.device).to_mlir(),
        input,
        updates,
        indices,
    )[0].tensor


def masked_scatter(
    input: TensorValueLike,
    mask: TensorValueLike,
    updates: TensorValueLike,
    out_dim: DimLike,
) -> TensorValue:
    """
    Creates a new symbolic tensor where the updates are written to input where mask is true.

    Args:
        input: The input symbolic tensor to write elements to.
        mask: A symbolic tensor of boolean values to update.
        updates: A symbolic tensor of elements to write to input.
        out_dim: The new data-dependent dimension.

    Returns:
        A new symbolic tensor representing the result of the masked_scatter operation.
    """
    input, updates = TensorValue(input), TensorValue(updates)
    mask = dtype_promotion._promote_to_strong(
        mask, DType.bool, input.type.device or DeviceRef.CPU()
    )

    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype ({input.dtype}) and updates dtype"
            f" ({updates.dtype}) must match"
        )

    # input_size = reduce(mul, input.shape, 1)
    # updates_size = reduce(mul, updates.shape, 1)
    # TODO: This is a bug. They don't have to match.
    # Assuming it will throw a run-time error if updates_size != non-zeros in mask
    # if input_size != updates_size and updates_size != 1:
    #    raise ValueError(
    #        f"The number of elements in the input ({input_size}) and the number"
    #        f" of elements in updates ({updates_size}) must match"
    #    )

    mask = mask.broadcast_to(input.shape)
    indices = nonzero(mask, out_dim)

    updates = updates.flatten()

    return scatter_nd(input, updates, indices)
