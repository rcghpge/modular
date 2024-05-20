# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from .device import Device
from .tensor_slice import TensorSlice
from utils import StaticTuple


@value
@register_passable
struct _StaticTensorSpec[type: DType, rank: Int]:
    var shape: StaticIntTuple[rank]

    fn get_rank(self) -> Int:
        return rank

    fn dtype(self) -> DType:
        return type

    fn __getitem__(self, idx: Int) -> Int:
        return self.shape[idx]


struct Tensor[type: DType, rank: Int]:
    var _ptr: DTypePointer[type]
    var _spec: _StaticTensorSpec[type, rank]
    var _device: Device

    fn __init__(
        inout self,
        ptr: DTypePointer[type],
        shape: StaticIntTuple[rank],
        device: Device,
    ):
        self._ptr = ptr
        self._spec = _StaticTensorSpec[type](shape)
        self._device = device

    fn __moveinit__(inout self, owned existing: Self):
        self._ptr = existing._ptr
        self._spec = existing._spec
        self._device = existing._device

    fn get_rank(self) -> Int:
        return self._spec.get_rank()

    fn get_dtype(self) -> DType:
        return self._spec.dtype()

    @always_inline
    fn __getitem__(self, *indices: Int) -> Scalar[type]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        return self.load[width=1](indices)

    @always_inline
    fn __getitem__(self, indices: StaticIntTuple[rank]) -> Scalar[type]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        return self.load[width=1](indices)

    @always_inline
    fn __getitem__(
        self: Reference[Self, _, _], *slices: Slice
    ) -> TensorSlice[type, rank, self.is_mutable, self.lifetime]:
        debug_assert(
            len(slices) == rank, "mismatch between slice count and rank"
        )
        var slice_tuple = StaticTuple[Slice, rank]()
        for i in range(len(slices)):
            slice_tuple[i] = slices[i]
        return TensorSlice(self, slice_tuple)

    @always_inline
    fn load[*, width: Int = 1](self, *indices: Int) -> SIMD[type, width]:
        """Gets the SIMD value at the specified indices.

        Parameters:
          width: The SIMD width of the vector.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The SIMD value at the specified indices.
        """
        return self._ptr.load[width=width](self._compute_linear_offset(indices))

    @always_inline
    fn load[
        *, width: Int = 1
    ](self, indices: StaticIntTuple[rank]) -> SIMD[type, width]:
        """Gets the SIMD value at the specified indices.

        Parameters:
          width: The SIMD width of the vector.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The SIMD value at the specified indices.
        """
        return self._ptr.load[width=width](self._compute_linear_offset(indices))

    @always_inline
    fn __setitem__(
        inout self, indices: StaticIntTuple[rank], val: Scalar[type]
    ):
        """Sets the value at the specified indices.

        Args:
          indices: The indices of the value to set.
          val: The value to store.
        """
        self.store[width=1](indices, val)

    @always_inline
    fn store[
        *, width: Int = 1
    ](inout self, indices: StaticIntTuple[rank], val: SIMD[type, width]):
        """Sets the SIMD value at the specified indices.

        Parameters:
          width: The SIMD width of the vector.

        Args:
          indices: The indices of the value to set.
          val: The SIMD value to store.
        """
        self._ptr.store[width=width](self._compute_linear_offset(indices), val)

    @always_inline
    fn _compute_linear_offset(self, indices: StaticIntTuple[rank]) -> Int:
        """Computes the linear offset into the tensor from the indices provided.

        Args:
          indices: The indices to index against.

        Returns:
          The linearized index into the tensor data.
        """
        var result = indices[0]

        @parameter
        for i in range(rank - 1):
            result = self._spec[i + 1] * result + indices[i + 1]
        return result

    @always_inline
    fn _compute_linear_offset(self, indices: VariadicList[Int]) -> Int:
        """Computes the linear offset into the tensor from the indices provided.

        Args:
          indices: The indices to index against.

        Returns:
          The linearized index into the tensor data.
        """
        var rank = len(indices)
        var result = indices[0]
        for i in range(rank - 1):
            result = self._spec[i + 1] * result + indices[i + 1]
        return result

    fn __del__(owned self):
        self._device._free(self._ptr.bitcast[DType.invalid]())
