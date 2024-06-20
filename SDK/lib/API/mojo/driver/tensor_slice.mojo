# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from .tensor import Tensor, StaticTensorSpec, _compute_linear_offset
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
    var _unsafe_slice: UnsafeTensorSlice[type, rank]

    fn __init__(
        inout self, tensor: Self._ref_type, slices: StaticTuple[Slice, rank]
    ):
        self = Self(
            tensor,
            UnsafeTensorSlice[type, rank](
                tensor[]._ptr, slices, tensor[]._spec
            ),
        )

    fn get_static_spec(self) -> StaticTensorSpec[type, rank]:
        """Gets the static spec of the slice.

        Returns:
            Static tensor spec of slice.
        """
        return self._unsafe_slice.get_static_spec()

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
        var adjusted_indices = self._unsafe_slice._get_adjusted_indices(indices)
        var offset = _compute_linear_offset(
            self._unsafe_slice._slicer_spec, adjusted_indices
        )
        return self._unsafe_slice._ptr[offset]


@value
@register_passable
struct UnsafeTensorSlice[
    type: DType,
    rank: Int,
]:
    """UnsafeTensorSlice is like TensorSlice but it contains no reference to
    the original Tensor. Therefore, the user must take care to keep the ptr alive
    until UnsafeTensorSlice's last use.
    """

    var _ptr: DTypePointer[type]
    var _slices: StaticTuple[Slice, rank]
    var _slicer_spec: StaticTensorSpec[type, rank]
    var _spec: StaticTensorSpec[type, rank]

    fn __init__(
        inout self,
        ptr: DTypePointer[type],
        slices: StaticTuple[Slice, rank],
        slicer_spec: StaticTensorSpec[type, rank],
    ):
        var adjusted_shape = StaticIntTuple[rank]()
        for i in range(rank):
            adjusted_shape[i] = int(
                ceil(
                    (slices[i].end.value() - slices[i].start.value())
                    / slices[i].step
                )
            )
        var slice_spec = StaticTensorSpec[type](adjusted_shape)
        self = Self(ptr, slices, slicer_spec, slice_spec)

    @always_inline
    fn _get_adjusted_indices(
        self, indices: StaticIntTuple[rank]
    ) -> StaticIntTuple[rank]:
        var adjusted = StaticIntTuple[rank]()
        for i in range(len(indices)):
            debug_assert(indices[i] < self._spec[i], "index out of bound")
            adjusted[i] = (
                indices[i] * self._slices[i].step
                + self._slices[i].start.value()
            )
        return adjusted

    fn get_static_spec(self) -> StaticTensorSpec[type, rank]:
        """Gets the static spec of the slice.

        Returns:
            Static tensor spec of slice.
        """
        return self._spec

    @always_inline
    fn __getitem__(inout self, *indices: Int) -> Scalar[type]:
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
        var offset = _compute_linear_offset(self._slicer_spec, adjusted_indices)
        return self._ptr[offset]
