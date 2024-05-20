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
            device._cdev,
            _get_current_or_global_runtime(),
        )
        self.spec = spec

    fn __del__(owned self):
        alias func_name_destroy = "M_destroyDeviceTensor"
        call_dylib_func[NoneType](
            self.lib.get_handle(), func_name_destroy, self._ptr
        )

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

    fn __str__(self) raises -> String:
        """Returns a description of the DeviceTensor."""

        return (
            "DeviceTensor("
            + str(self.get_device())
            + ",Spec("
            + str(self.spec)
            + "))"
        )

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
