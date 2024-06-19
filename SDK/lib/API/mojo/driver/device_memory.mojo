# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor import TensorSpec, TensorShape
from ._driver_library import DriverLibrary
from max.engine._utils import call_dylib_func
from builtin._startup import _get_current_or_global_runtime
from .device import Device, _CDevice
from .tensor import Tensor
from .anymemory import AnyMemory


trait DeviceBuffer:
    fn copy_to(self, dev: Device, name: Optional[String]) raises -> Self:
        """Copies the contents of self into DeviceBuffer allocated on dev.
        Note: this function allocates memory on dev.

        Args:
            dev: The Device on which to allocate the new DeviceBuffer.
            name: Optional name of the new DeviceBuffer.

        Returns:
            Newly allocated DeviceBuffer containing a copy of self's contents.

        Raises:
            If the DeviceBuffer is backed by the same Device object as dev.
        """
        ...

    fn copy_into(self, inout dst_memory: Self) raises:
        """Copies the contents of self into a preallocated DeviceBuffer.

        Args:
            dst_memory: The destination DeviceBuffer of the copy.
        """
        ...

    fn unsafe_ptr(self) -> DTypePointer[DType.invalid]:
        ...

    fn device(self) -> Device:
        ...

    fn bytecount(self) -> Int:
        ...


struct DeviceMemory(DeviceBuffer, StringableRaising, CollectionElement):
    """DeviceMemory is an owning buffer allocated on a (possibly non-CPU) Device.
    """

    var _ptr: DTypePointer[DType.invalid]
    var _device: Device
    var lib: DriverLibrary
    var name: Optional[String]
    var num_bytes: Int

    fn __init__(inout self):
        # Workaround for MOCO-840
        # Removing this will cause mojo to do a double free!
        try:
            self = Self(
                DTypePointer[DType.invalid](),
                0,
                Device(),
            )
        except e:
            self = abort[DeviceMemory](e)

    fn __init__(
        inout self,
        num_bytes: Int,
        device: Device,
        name: Optional[String] = None,
    ) raises:
        """Allocates a DeviceMemory from the Device's address space."""
        self._device = device
        self.lib = self._device.lib
        alias func_name_create_tensor = "M_createDeviceMemory"
        var tmp_spec = TensorSpec(DType.uint8, num_bytes)
        # CAUTION: this assumes that TensorSpec is bitwise identical in mojo and cpp
        self._ptr = call_dylib_func[DTypePointer[DType.invalid]](
            self.lib.get_handle(),
            func_name_create_tensor,
            UnsafePointer[TensorSpec].address_of(tmp_spec),
            self._device._cdev,
        )
        self.name = name
        self.num_bytes = num_bytes
        _ = tmp_spec^

    fn __init__(
        inout self,
        owned_ptr: DTypePointer[DType.invalid],
        num_bytes: Int,
        device: Device,
        name: Optional[String] = None,
    ):
        """Creates DeviceMemory from a C Ptr."""
        self._device = device
        self.lib = self._device.lib
        self._ptr = owned_ptr
        self.name = name
        self.num_bytes = num_bytes

    fn __copyinit__(inout self, existing: Self):
        # This temporarily exists so that we can store DeviceMemory in a List
        # TODO(MSTDL-467): Once Copyable requirement on List is removed, this
        # can be removed
        constrained[
            False, "__copyinit__ not supported on DeviceMemory, MSTDL-467"
        ]()
        self._ptr = existing._ptr
        self._device = existing._device
        self.lib = existing.lib
        self.name = existing.name
        self.num_bytes = existing.num_bytes

    fn __init__[
        type: DType, rank: Int
    ](inout self, owned tensor: Tensor[type, rank]) raises:
        """Creates a DeviceMemory from given tensor."""

        self._device = tensor._get_device()
        self.lib = self._device.lib
        self.name = tensor.name
        self.num_bytes = tensor.get_tensor_spec().bytecount()
        self._ptr = tensor._device_memory_impl_ptr
        tensor._device_memory_impl_ptr = DTypePointer[DType.invalid]()

    fn __init__(inout self, owned anymemory: AnyMemory) raises:
        """Creates a device tensor from given anymemory."""

        self._device = anymemory._device
        self.lib = self._device.lib
        self.name = anymemory._name
        self._ptr = anymemory._device_memory_impl_ptr
        self.num_bytes = anymemory._spec.bytecount()
        anymemory._device_memory_impl_ptr = DTypePointer[DType.invalid]()

    fn __del__(owned self):
        alias func_name_destroy = "M_destroyDeviceMemory"
        call_dylib_func[NoneType](
            self.lib.get_handle(), func_name_destroy, self._ptr
        )
        # Extend lifetime of library until C function returns.
        _ = self.lib^

    fn bytecount(self) -> Int:
        return self.num_bytes

    fn get_device(self) -> Device:
        """Returns the device on which the DeviceMemory was allocated."""

        return self._device

    fn device(self) -> Device:
        """Returns the device on which the DeviceMemory was allocated."""

        return self._device

    fn __moveinit__(inout self, owned existing: Self):
        self._ptr = existing._ptr
        self._device = existing._device^
        self.lib = existing.lib^
        self.name = existing.name^
        self.num_bytes = existing.num_bytes

    fn __str__(self) raises -> String:
        """Returns a description of the DeviceMemory."""

        return (
            "DeviceMemory("
            + (self.name.value() + "," if self.name else "")
            + str(self.get_device())
            + ",Bytecount("
            + str(self.bytecount())
            + "))"
        )

    fn _steal_impl_ptr(owned self) -> DTypePointer[DType.invalid]:
        var tmp = self._ptr
        self._ptr = DTypePointer[DType.invalid]()
        return tmp

    fn _steal_ptr(owned self) -> DTypePointer[DType.invalid]:
        alias func_name_take_data = "M_takeDataFromDeviceMemory"
        var take_data_func = self.lib.get_handle().get_function[
            fn (DTypePointer[DType.invalid]) -> DTypePointer[DType.invalid]
        ](func_name_take_data)
        var data = take_data_func(self._ptr)
        # Extend lifetime of self to avoid the C funtion working on invalid pointer.
        _ = self^
        return data

    fn copy_into(self, inout dst_memory: DeviceMemory) raises:
        if self.bytecount() != dst_memory.bytecount():
            raise "src and dst bytcount mismatch in copy_into()"

        alias func_name_steal_data = "M_copyDeviceMemory"
        call_dylib_func(
            self.lib.get_handle(),
            func_name_steal_data,
            dst_memory._ptr,
            self._ptr,
        )

    fn copy_to(
        self, dev: Device, name: Optional[String] = None
    ) raises -> DeviceMemory:
        if dev == self._device:
            raise str(self) + "is already allocated on " + str(dev)

        var dst = dev.allocate(self.bytecount(), name)
        self.copy_into(dst)
        return dst^

    fn unsafe_ptr(self) -> DTypePointer[DType.invalid]:
        """Returns a pointer to the underlying device memory.

        Note: The caller is responsible for ensuring that the returned pointer
        is not used after it's owner is last used.
        """

        alias func_name_get_data = "M_getData"
        return call_dylib_func[
            DTypePointer[DType.invalid], DTypePointer[DType.invalid]
        ](
            self.lib.get_handle(),
            func_name_get_data,
            self._ptr,
        )


struct DeviceTensor(DeviceBuffer, StringableRaising, CollectionElement):
    var _storage: DeviceMemory
    var spec: TensorSpec

    fn __init__(
        inout self, spec: TensorSpec, device: Device, name: Optional[String]
    ) raises:
        self.spec = spec
        self._storage = DeviceMemory(
            spec.bytecount(),
            device,
            name,
        )

    fn __init__(
        inout self, owned storage: DeviceMemory, spec: TensorSpec
    ) raises:
        self._storage = storage^
        self.spec = spec

        if self.bytecount() != self.bytecount():
            raise "DeviceMemory size does not match DeviceTensor requirements"

    fn copy_to(self, dev: Device, name: Optional[String] = None) raises -> Self:
        var t = Self(self._storage.copy_to(dev, name), self.spec)
        return t

    fn copy_into(self, inout dst_tensor: Self) raises:
        if dst_tensor.spec != self.spec:
            raise "src and dst tensor specs do not match"
        self._storage.copy_into(dst_tensor._storage)

    fn unsafe_ptr(self) -> DTypePointer[DType.invalid]:
        return self._storage.unsafe_ptr()

    fn get_tensor[
        type: DType, rank: Int
    ](owned self) raises -> Tensor[type, rank]:
        if rank != self.spec.rank():
            raise "requested rank does not match existing rank"

        if type != self.spec.dtype():
            raise "requested dtype does not match existing type."

        return Tensor[type, rank](self^)

    fn device(self) -> Device:
        return self._storage.device()

    fn name(self) -> Optional[String]:
        return self._storage.name

    fn __copyinit__(inout self, existing: Self):
        # This temporarily exists so that we can store DeviceMemory in a List
        # TODO(MSTDL-467): Once Copyable requirement on List is removed, this
        # can be removed
        constrained[
            False, "__copyinit__ not supported on DeviceTensor, MSTDL-467"
        ]()
        self._storage = existing._storage
        self.spec = existing.spec

    fn __moveinit__(inout self, owned existing: Self):
        self._storage = existing._storage^
        self.spec = existing.spec^

    fn __str__(self) raises -> String:
        """Returns a description of the DeviceMemory."""

        return (
            "DeviceTensor("
            + (self._storage.name.value() + "," if self._storage.name else "")
            + str(self.device())
            + ",Spec("
            + str(self.spec)
            + "))"
        )

    fn bytecount(self) -> Int:
        return self.spec.bytecount()
