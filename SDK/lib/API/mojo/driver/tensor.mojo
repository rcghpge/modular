# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from .device import Device, DeviceMemory
from .tensor_slice import TensorSlice
from utils import StaticTuple
from tensor import TensorSpec


@value
@register_passable
struct StaticTensorSpec[type: DType, rank: Int]:
    var shape: StaticIntTuple[rank]

    fn get_rank(self) -> Int:
        return rank

    fn dtype(self) -> DType:
        return type

    fn get_tensor_spec(self) -> TensorSpec:
        var shapes = List[Int]()
        shapes.reserve(rank)
        for i in range(rank):
            shapes.append(self[i])
        return TensorSpec(type, shapes)

    fn __getitem__(self, idx: Int) -> Int:
        return self.shape[idx]


struct Tensor[type: DType, rank: Int](CollectionElement):
    var _ptr: DTypePointer[type]
    var _spec: StaticTensorSpec[type, rank]
    var _device: Device
    var name: Optional[String]

    fn __init__(
        inout self,
        ptr: DTypePointer[type],
        shape: StaticIntTuple[rank],
        device: Device,
        name: Optional[String] = None,
    ):
        self._ptr = ptr
        self._spec = StaticTensorSpec[type](shape)
        self._device = device
        self.name = name

    fn __moveinit__(inout self, owned existing: Self):
        self._ptr = existing._ptr
        self._spec = existing._spec
        self._device = existing._device
        self.name = existing.name^

    fn __copyinit__(inout self, existing: Self):
        # This temporarily exists so that we can store Tensor in a List
        # TODO(MSTDL-467): Once Copyable requirement on List is removed, this
        # can be removed
        constrained[False, "__copyinit__ not supported on Tensor, MSTDL-467"]()
        self._ptr = existing._ptr
        self._spec = existing._spec
        self._device = existing._device
        self.name = existing.name

    fn get_rank(self) -> Int:
        return self._spec.get_rank()

    fn get_dtype(self) -> DType:
        return self._spec.dtype()

    fn get_tensor_spec(self) -> TensorSpec:
        return self._spec.get_tensor_spec()

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
            slice_tuple[i].end = min(slice_tuple[i].end, self[]._spec[i])
        return TensorSlice(self, slice_tuple)

    fn _get_mut_ref(
        self: Reference[Self, True, _]
    ) -> Reference[Self, True, self.lifetime]:
        return self

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
        debug_assert(
            len(indices) == rank, "mismatch between requested index and rank"
        )
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

    # FIXME: [MOCO-696] Use variadic length keys
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

    fn _steal_ptr(owned self) -> DTypePointer[type]:
        var tmp = self._ptr
        self._ptr = DTypePointer[type]()
        return tmp

    fn _get_device(self) -> Device:
        return self._device

    fn get_device_memory(owned self) raises -> DeviceMemory:
        return DeviceMemory(self^)

    fn __del__(owned self):
        self._device._free(self._ptr.bitcast[DType.invalid]())

    fn unsafe_ptr(self) -> DTypePointer[type]:
        """Returns a pointer to the underlying memory.

        Note: The caller is responsible for ensuring that the returned pointer
        is not used after it's owner is last used.
        """
        return self._ptr
