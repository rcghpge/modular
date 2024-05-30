# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor import TensorSpec
from ._driver_library import DriverLibrary
from max.engine._utils import call_dylib_func
from builtin._startup import _get_current_or_global_runtime
from .device import Device, _CDevice
from .tensor import Tensor
from .anymemory import AnyMemory


struct DeviceMemory(StringableRaising, CollectionElement):
    var _ptr: DTypePointer[DType.invalid]
    var _device: Device
    var lib: DriverLibrary
    # TODO: de-duplicate spec on mojo and cpp sides
    var spec: TensorSpec
    var name: Optional[String]

    fn __init__(
        inout self,
        spec: TensorSpec,
        device: Device,
        name: Optional[String] = None,
    ) raises:
        """Allocates a DeviceMemory from the Device's address space."""
        self._device = device
        self.lib = self._device.lib
        alias func_name_create_tensor = "M_createDeviceMemory"
        # CAUTION: this assumes that TensorSpec is bitwise identical in mojo and cpp
        self._ptr = call_dylib_func[DTypePointer[DType.invalid]](
            self.lib.get_handle(),
            func_name_create_tensor,
            UnsafePointer[TensorSpec].address_of(spec),
            self._device._cdev,
        )
        self.spec = spec
        self.name = name

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
        self.spec = existing.spec
        self.name = existing.name

    fn __init__[
        type: DType, rank: Int
    ](inout self, owned tensor: Tensor[type, rank]) raises:
        """Creates a DeviceMemory from given tensor."""

        self._device = tensor._get_device()
        self.lib = self._device.lib
        self.spec = tensor.get_tensor_spec()
        self.name = tensor.name
        alias func_name_create_tensor = "M_createDeviceMemoryFrom"
        var create_func = self.lib.get_handle().get_function[
            fn (
                DTypePointer[DType.invalid], UnsafePointer[TensorSpec], _CDevice
            ) -> DTypePointer[DType.invalid]
        ](func_name_create_tensor)
        var data = tensor^._steal_ptr().bitcast[DType.invalid]()
        # Need to initialize self._ptr before calling address_of(self.spec)
        self._ptr = DTypePointer[DType.invalid]()
        # CAUTION: this assumes that TensorSpec is bitwise identical in mojo and cpp
        self._ptr = create_func(
            data,
            UnsafePointer[TensorSpec].address_of(self.spec),
            self._device._cdev,
        )

    fn __init__(inout self, owned anymemory: AnyMemory) raises:
        """Creates a device tensor from given anymemory."""

        self._device = anymemory._device
        self.lib = self._device.lib
        self.spec = anymemory._spec
        self.name = anymemory._name
        alias func_name_create_tensor = "M_createDeviceMemoryFrom"
        var create_func = self.lib.get_handle().get_function[
            fn (
                DTypePointer[DType.invalid], UnsafePointer[TensorSpec], _CDevice
            ) -> DTypePointer[DType.invalid]
        ](func_name_create_tensor)
        # Need to initialize self._ptr before calling address_of(self.spec)
        self._ptr = DTypePointer[DType.invalid]()
        # CAUTION: this assumes that TensorSpec is bitwise identical in mojo and cpp
        self._ptr = create_func(
            anymemory^._steal_ptr(),
            UnsafePointer[TensorSpec].address_of(self.spec),
            self._device._cdev,
        )

    fn __del__(owned self):
        alias func_name_destroy = "M_destroyDeviceMemory"
        call_dylib_func[NoneType](
            self.lib.get_handle(), func_name_destroy, self._ptr
        )
        # Extend lifetime of library until C function returns.
        _ = self.lib^

    fn get_tensor_spec(self) -> TensorSpec:
        return self.spec

    fn get_device(self) raises -> Device:
        """Returns the device on which the DeviceMemory was allocated."""

        alias func_name_get_dev = "M_getDeviceFromDeviceMemory"
        var _cdev = _CDevice(
            call_dylib_func[
                DTypePointer[DType.invalid], DTypePointer[DType.invalid]
            ](self.lib.get_handle(), func_name_get_dev, self._ptr)
        )
        # copy() is needed to bump the refcount on the device object
        return Device(self.lib, owned_ptr=_cdev.copy(self.lib))

    fn __moveinit__(inout self, owned existing: Self):
        self._ptr = existing._ptr
        self._device = existing._device^
        self.lib = existing.lib^
        self.spec = existing.spec^
        self.name = existing.name^

    fn get_rank(self) -> Int:
        return self.spec.rank()

    fn __str__(self) raises -> String:
        """Returns a description of the DeviceMemory."""

        return (
            "DeviceMemory("
            + (self.name.value()[] + "," if self.name else "")
            + str(self.get_device())
            + ",Spec("
            + str(self.spec)
            + "))"
        )

    fn _steal_ptr(owned self) -> DTypePointer[DType.invalid]:
        alias func_name_take_data = "M_takeDataFromDeviceMemory"
        var take_data_func = self.lib.get_handle().get_function[
            fn (DTypePointer[DType.invalid]) -> DTypePointer[DType.invalid]
        ](func_name_take_data)
        var data = take_data_func(self._ptr)
        # Extend lifetime of self to avoid the C funtion working on invalid pointer.
        _ = self^
        return data

    fn get_tensor[
        dtype: DType, rank: Int
    ](owned self) raises -> Tensor[dtype, rank]:
        if rank != self.spec.rank():
            raise "requested rank does not match existing rank"

        if dtype != self.spec.dtype():
            raise "requested dtype does not match existing type."

        var shape = StaticIntTuple[rank]()

        for i in range(rank):
            shape[i] = self.spec[i]

        alias func_name_steal_data = "M_takeDataFromDeviceMemory"
        var data = call_dylib_func[DTypePointer[dtype]](
            self.lib.get_handle(), func_name_steal_data, self._ptr
        )
        return Tensor(data, shape, self._device, self.name)

    fn copy_to(self, inout dst_memory: DeviceMemory) raises:
        """Copies the contents of self into a preallocated buffer.

        Args:
            dst_memory: The destination buffer of the copy.
        """
        if self.get_tensor_spec() != dst_memory.get_tensor_spec():
            raise "Tensor specs do not match in copy_to(), src=" + str(
                self.get_tensor_spec()
            ) + ", dst=" + str(dst_memory.get_tensor_spec())

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
        """Copies the contents of self into DeviceMemory allocated on dev.
        Note: this function allocates memory on dev.

        Args:
            dev: The Device on which to allocate the new DeviceMemory.
            name: Optional name of the new DeviceMemory.

        Returns:
            Newly allocated DeviceMemory containing a copy of the self's contents.

        Raises:
            If the DeviceMemory is backed by the same Device object as dev.
        """
        if dev == self._device:
            raise str(self) + "is already allocated on " + str(dev)

        var dst = dev.allocate(self.spec, name)
        self.copy_to(dst)
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
