# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .device import Device, _get_driver_path, _CDevice
from max.engine._utils import call_dylib_func
from sys.ffi import DLHandle
from ._driver_library import DriverLibrary, ManagedDLHandle


fn get_cuda_device(gpu_id: Int = 0) raises -> Device:
    var lib = ManagedDLHandle(DLHandle(_get_driver_path()))
    alias func_name_create = "M_createCUDADevice"
    var _cdev = _CDevice(
        call_dylib_func[DTypePointer[DType.invalid]](
            lib.get_handle(), func_name_create, gpu_id
        )
    )
    return Device(lib^, owned_ptr=_cdev)
