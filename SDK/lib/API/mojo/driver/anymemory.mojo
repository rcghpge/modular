# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from .device_memory import DeviceMemory
from .tensor import Tensor
from .device import Device
from tensor import TensorSpec


struct AnyMemory:
    """A type erased tensor representation that is useful
    for situations where we need variadics of tensors."""

    var _data: DTypePointer[DType.invalid]
    var _spec: TensorSpec
    var _device: Device
    var _name: Optional[String]
    var _device_memory_impl_ptr: DTypePointer[DType.invalid]

    fn __init__(inout self) raises:
        self._device = Device()
        self._spec = TensorSpec()
        self._name = None
        self._data = DTypePointer[DType.invalid]()
        self._device_memory_impl_ptr = DTypePointer[DType.invalid]()

    fn __init__(inout self, owned device_memory: DeviceMemory) raises:
        self._device = device_memory.get_device()
        self._spec = device_memory.get_tensor_spec()
        self._name = device_memory.name
        self._data = device_memory.unsafe_ptr()
        self._device_memory_impl_ptr = device_memory^._steal_impl_ptr()

    fn __copyinit__(inout self, existing: Self):
        constrained[False, "AnyMemory is non-copyable"]()
        self._device = existing._device
        self._spec = existing._spec
        self._name = existing._name
        self._data = existing._data
        self._device_memory_impl_ptr = existing._device_memory_impl_ptr

    fn __moveinit__(inout self, owned existing: Self):
        self._device = existing._device^
        self._spec = existing._spec^
        self._name = existing._name^
        self._data = existing._data
        self._device_memory_impl_ptr = existing._device_memory_impl_ptr

    fn __init__[
        type: DType, rank: Int
    ](inout self, owned tensor: Tensor[type, rank]) raises:
        self = Self(tensor^.get_device_memory())

    fn get_rank(self) -> Int:
        """Gets rank of the tensor.

        Returns:
            Rank of the tensor.
        """
        return self._spec.rank()

    fn _steal_ptr(owned self) -> DTypePointer[DType.invalid]:
        var ptr = self._data
        self._data = DTypePointer[DType.invalid]()
        return ptr

    fn device_memory(owned self) raises -> DeviceMemory:
        return DeviceMemory(self^)

    fn __del__(owned self):
        _ = DeviceMemory(self._device_memory_impl_ptr, self._spec, self._device)
