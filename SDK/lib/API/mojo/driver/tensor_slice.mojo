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

    # FIXME: [MOCO-696] Use variadic length keys
    @always_inline
    fn __setitem__(
        inout self, indices: StaticIntTuple[rank], value: Scalar[type]
    ):
        """Sets the value at given index.

        Args:
            indices: Index to set value at.
            value: Value to be set.
        """
        constrained[self.is_mutable == True, "tensor slice is immutable"]()
        # TODO: Don't cast to ptr. We have to do this because we don't
        # have conditional conformance.
        var tensor_ptr = UnsafePointer.address_of(self._ref[])
        var adjusted_indices = self._get_adjusted_indices(indices)
        tensor_ptr[].store(adjusted_indices, value)

    @always_inline
    fn __getitem__(self, *indices: Int) -> Scalar[type]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        return self[indices]

    @always_inline
    fn __getitem__(self, indices: StaticIntTuple[rank]) -> Scalar[type]:
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
        return self._ref[][adjusted_indices]
