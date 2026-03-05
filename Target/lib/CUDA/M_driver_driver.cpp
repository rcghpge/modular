//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#include "Driver.h"
#include "Error.h"
#include "HAL/M_driver_api.h"
#include "llvm/ADT/StringRef.h"

namespace M::Target::CUDA {

Driver::~Driver() {
  rawCuda.unload();
  rawNvml.unload();
}

M_driver_status_t Driver::destroy(Driver *driver) {
  delete driver;
  return M_DRIVER_STATUS_SUCCESS;
}

Device *Driver::findDevice(CUdevice id) {
  for (auto &dev : devices) {
    if (dev.deviceId == id)
      return &dev;
  }
  return nullptr;
}

} // namespace M::Target::CUDA

using namespace M::Target::CUDA;

M_driver_status_t M_driver_create(M_driver_driver_t *outDriver) {
  auto nvml = NVML::load();
  if (!nvml)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;

  if (nvml->nvmlInit_v2() != NVML_SUCCESS) {
    nvml->unload();
    return M_DRIVER_STATUS_UNKNOWN_ERROR;
  }

  auto cuda = CUDALoader::load();
  if (!cuda) {
    nvml->unload();
    return M_DRIVER_STATUS_UNKNOWN_ERROR;
  }

  if (cuda->cuInit(0) != CUDA_SUCCESS) {
    cuda->unload();
    nvml->unload();
    return M_DRIVER_STATUS_UNKNOWN_ERROR;
  }

  auto *driver = new Driver(std::move(*cuda), std::move(*nvml));

  // Reserve upfront so Device pointers (used as handles) stay stable.
  unsigned int deviceCount = 0;
  if (driver->nvml().nvmlDeviceGetCount_v2(&deviceCount) != NVML_SUCCESS) {
    delete driver;
    return M_DRIVER_STATUS_NO_DEVICES;
  }
  driver->devices.reserve(deviceCount);

  *outDriver = driver->cast();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_destroy(M_driver_driver_t handle) {
  M_driver_status_t status = Driver::destroy(Driver::from(handle));
  return status;
}

M_driver_status_t M_driver_property([[maybe_unused]] M_driver_driver_t handle,
                                    const char *property_name, void *value) {
  llvm::StringRef name(property_name);
  if (name == M_DRIVER_PROPERTY_API)
    *static_cast<const char **>(value) = "CUDA";
  else
    return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
  return M_DRIVER_STATUS_SUCCESS;
}
