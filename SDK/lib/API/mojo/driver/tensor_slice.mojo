# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from .tensor import Tensor, StaticTensorSpec
from utils import StaticTuple
from math import ceil


@value
@register_passable
struct TensorSlice[
    is_mutable: Bool, //,
    type: DType,
    rank: Int,
    lifetime: AnyLifetime[is_mutable].type,
]:
    alias _ref_type = Reference[Tensor[type, rank], lifetime]
    var _ref: Self._ref_type
    var _slices: StaticTuple[Slice, rank]
    var _spec: StaticTensorSpec[type, rank]

    fn __init__(
        inout self, tensor: Self._ref_type, slices: StaticTuple[Slice, rank]
    ):
        var adjusted_shape = StaticIntTuple[rank]()
        for i in range(rank):
            adjusted_shape[i] = int(
                ceil((slices[i].end - slices[i].start) / slices[i].step)
            )
        var slice_spec = StaticTensorSpec[type](adjusted_shape)
        self = Self(tensor, slices, slice_spec)

    @always_inline
    fn _get_adjusted_indices(
        self, indices: StaticIntTuple[rank]
    ) -> StaticIntTuple[rank]:
        var adjusted = StaticIntTuple[rank]()
        for i in range(len(indices)):
            debug_assert(indices[i] < self._spec[i], "index out of bound")
            adjusted[i] = (
                indices[i] * self._slices[i].step + self._slices[i].start
            )
        return adjusted

    fn get_static_spec(self) -> StaticTensorSpec[type, rank]:
        """Gets the static spec of the slice.

        Returns:
            Static tensor spec of slice.
        """
        return self._spec

    @always_inline
    fn __getitem__(inout self, *indices: Int) -> ref [lifetime] Scalar[type]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        debug_assert(
            len(indices) == rank, "mismatch between requested index and rank"
        )
        var adjusted_indices = self._get_adjusted_indices(indices)
        var offset = self._ref[]._compute_linear_offset(adjusted_indices)
        return self._ref[]._ptr[offset]
