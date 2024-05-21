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


struct DeviceTensor(StringableRaising):
    var _ptr: DTypePointer[DType.invalid]
    var _device: Device
    var lib: DriverLibrary
    var spec: TensorSpec

    fn __init__(inout self, spec: TensorSpec, device: Device) raises:
        """Allocates a DeviceTensor from the Device's address space."""

        self._device = device
        self.lib = self._device.lib
        alias func_name_create_tensor = "M_createDeviceTensor"
        self._ptr = call_dylib_func[DTypePointer[DType.invalid]](
            self.lib.get_handle(),
            func_name_create_tensor,
            spec.bytecount(),
            self._device._cdev,
        )
        self.spec = spec

    fn __init__[
        type: DType, rank: Int
    ](inout self, owned tensor: Tensor[type, rank]) raises:
        """Creates a DeviceTensor from given tensor."""

        self._device = tensor._get_device()
        self.lib = self._device.lib
        self.spec = tensor.get_tensor_spec()
        alias func_name_create_tensor = "M_createDeviceTensorFrom"
        var create_func = self.lib.get_handle().get_function[
            fn (
                DTypePointer[DType.invalid], Int, _CDevice
            ) -> DTypePointer[DType.invalid]
        ](func_name_create_tensor)
        var data = tensor^._steal_ptr().bitcast[DType.invalid]()
        self._ptr = create_func(data, self.spec.bytecount(), self._device._cdev)

    fn __del__(owned self):
        alias func_name_destroy = "M_destroyDeviceTensor"
        call_dylib_func[NoneType](
            self.lib.get_handle(), func_name_destroy, self._ptr
        )

    fn get_tensor_spec(self) -> TensorSpec:
        return self.spec

    fn get_device(self) raises -> Device:
        """Returns the device on which the DeviceTensor was allocated."""

        alias func_name_get_dev = "M_getDeviceFromDeviceTensor"
        return Device(
            _CDevice(
                call_dylib_func[
                    DTypePointer[DType.invalid], DTypePointer[DType.invalid]
                ](self.lib.get_handle(), func_name_get_dev, self._ptr)
            )
        )

    fn get_rank(self) -> Int:
        return self.spec.rank()

    fn __str__(self) raises -> String:
        """Returns a description of the DeviceTensor."""

        return (
            "DeviceTensor("
            + str(self.get_device())
            + ",Spec("
            + str(self.spec)
            + "))"
        )

    fn _steal_ptr(owned self) -> DTypePointer[DType.invalid]:
        alias func_name_take_data = "M_takeDataFromDeviceTensor"
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
        debug_assert(
            rank == self.spec.rank(),
            "requested rank does not match existing rank",
        )
        var shape = StaticIntTuple[rank]()

        for i in range(rank):
            shape[i] = self.spec[i]

        alias func_name_steal_data = "M_takeDataFromDeviceTensor"
        var data = call_dylib_func[DTypePointer[dtype]](
            self.lib.get_handle(), func_name_steal_data, self._ptr
        )
        return Tensor(data, shape, self._device)
