# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from .device import Device, DeviceMemory, DeviceTensor
from .tensor_slice import TensorSlice
from utils import StaticTuple
from tensor import TensorSpec


@always_inline
fn _compute_linear_offset[
    type: DType, rank: Int
](spec: StaticTensorSpec[type, rank], indices: StaticIntTuple[rank]) -> Int:
    """Computes the linear offset into the tensor from the indices provided.

    Args:
        spec: The spec of the tensor being indexed into.
        indices: The indices to index against.

    Returns:
        The linearized index into the tensor data.
    """
    var result = indices[0]

    @parameter
    for i in range(rank - 1):
        result = spec[i + 1] * result + indices[i + 1]
    return result


@value
@register_passable
struct StaticTensorSpec[type: DType, rank: Int]:
    var shape: StaticIntTuple[rank]

    fn __init__(inout self, spec: TensorSpec):
        """Construct from TensorSpec.

        Args:
            spec: TensorSpec of given rank and type.
        """
        debug_assert(spec.rank() == rank, "rank mismatch")
        debug_assert(spec.dtype() == type, "dtype mismatch")
        self.shape = StaticIntTuple[rank]()

        for i in range(rank):
            self.shape[i] = spec[i]

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

    # TODO: We should be able to hold DeviceMemory here. Revisit
    # after DeviceMemory/DeviceTensor work.
    # this is needed because DeviceMemory may have a custom free
    # function set on the cpp side.
    var _device_memory_impl_ptr: DTypePointer[DType.invalid]

    fn __init__(inout self, owned device_tensor: DeviceTensor) raises:
        self._device = device_tensor.device()
        self.name = device_tensor.name()
        self._spec = device_tensor.spec
        self._ptr = device_tensor.unsafe_ptr().bitcast[type]()
        var tmp = device_tensor._storage^
        device_tensor._storage = DeviceMemory()
        self._device_memory_impl_ptr = tmp^._steal_impl_ptr()

    fn __moveinit__(inout self, owned existing: Self):
        self._ptr = existing._ptr
        self._spec = existing._spec^
        self._device = existing._device^
        self.name = existing.name^
        self._device_memory_impl_ptr = existing._device_memory_impl_ptr

    fn __copyinit__(inout self, existing: Self):
        # This temporarily exists so that we can store Tensor in a List
        # TODO(MSTDL-467): Once Copyable requirement on List is removed, this
        # can be removed
        constrained[False, "__copyinit__ not supported on Tensor, MSTDL-467"]()
        self._ptr = existing._ptr
        self._spec = existing._spec
        self._device = existing._device
        self.name = existing.name
        self._device_memory_impl_ptr = existing._device_memory_impl_ptr

    fn get_rank(self) -> Int:
        return self._spec.get_rank()

    fn get_dtype(self) -> DType:
        return self._spec.dtype()

    fn get_tensor_spec(self) -> TensorSpec:
        return self._spec.get_tensor_spec()

    @always_inline
    fn __getitem__(
        inout self, *indices: Int
    ) -> ref [__lifetime_of(self)] Scalar[type]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        debug_assert(
            len(indices) == rank, "mismatch between requested index and rank"
        )
        var offset = _compute_linear_offset(self._spec, indices)
        return self._ptr[offset]

    @always_inline
    fn __getitem__(
        ref [_]self: Self, *slices: Slice
    ) raises -> TensorSlice[type, rank, __lifetime_of(self)]:
        if len(slices) != rank:
            raise "mismatch between slice count and rank"
        var slice_tuple = StaticTuple[Slice, rank]()
        for i in range(len(slices)):
            slice_tuple[i] = slices[i]
            slice_tuple[i].start = (slice_tuple[i].start or 0).value()
            slice_tuple[i].end = (slice_tuple[i].end or self._spec[i]).value()
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
        debug_assert(
            len(indices) == rank, "mismatch between requested index and rank"
        )
        return SIMD[size=width].load(
            self._ptr, _compute_linear_offset(self._spec, indices)
        )

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
        return SIMD[size=width].load(
            self._ptr, _compute_linear_offset(self._spec, indices)
        )

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
        SIMD[size=width].store(
            self._ptr, _compute_linear_offset(self._spec, indices), val
        )

    fn _steal_ptr(owned self) -> DTypePointer[type]:
        var tmp = self._ptr
        self._ptr = DTypePointer[type]()
        return tmp

    fn _get_device(self) -> Device:
        return self._device

    fn get_device_memory(owned self) raises -> DeviceMemory:
        return DeviceMemory(self^)

    fn device_tensor(owned self) raises -> DeviceTensor:
        var spec = self._spec.get_tensor_spec()
        return DeviceTensor(DeviceMemory(self^), spec)

    fn __del__(owned self):
        _ = DeviceMemory(
            self._device_memory_impl_ptr,
            self._spec.get_tensor_spec().bytecount(),
            self._device,
        )

    fn unsafe_ptr(self) -> DTypePointer[type]:
        """Returns a pointer to the underlying memory.

        Note: The caller is responsible for ensuring that the returned pointer
        is not used after it's owner is last used.
        """
        return self._ptr
