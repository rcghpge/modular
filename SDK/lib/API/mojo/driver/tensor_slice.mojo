# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from .tensor import Tensor, _StaticTensorSpec
from utils import StaticTuple


@value
@register_passable
struct TensorSlice[
    type: DType,
    rank: Int,
    is_mutable: Bool,
    lifetime: AnyLifetime[is_mutable].type,
]:
    alias _ref_type = Reference[Tensor[type, rank], is_mutable, lifetime]
    var _ref: Self._ref_type
    var _slices: StaticTuple[Slice, rank]
    var _spec: _StaticTensorSpec[type, rank]

    fn __init__(
        inout self, tensor: Self._ref_type, slices: StaticTuple[Slice, rank]
    ):
        var adjusted_shape = StaticIntTuple[rank]()
        for i in range(rank):
            debug_assert(
                slices[i].step == 1,
                "FIXME: step other than one is not supported",
            )
            adjusted_shape[i] = slices[i].end - slices[i].start
        var slice_spec = _StaticTensorSpec[type](adjusted_shape)
        self = Self(tensor, slices, slice_spec)

    fn _get_adjusted_offset(self, indices: VariadicList[Int]) -> Int:
        var adjusted = StaticIntTuple[rank]()
        for i in range(len(indices)):
            debug_assert(indices[i] < self._spec[i], "index out of bound")
            adjusted[i] = indices[i] + self._slices[i].start
        return self._ref[]._compute_linear_offset(adjusted)

    @always_inline
    fn __getitem__(self, *indices: Int) -> Scalar[type]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        debug_assert(
            len(indices) == rank, "mismatch between requested index and rank"
        )
        var adjusted_offset = self._get_adjusted_offset(indices)
        return self._ref[].load(adjusted_offset)
