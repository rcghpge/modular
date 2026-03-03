//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#include "Driver.h"
#include "Error.h"
#include "HAL/M_driver_api.h"

namespace M::Target::CUDA {

Driver::~Driver() {
  rawCuda.unload();
  rawNvml.unload();
}

M_driver_status_t Driver::destroy(Driver *driver) {
  M_driver_status_t result = M_DRIVER_STATUS_SUCCESS;
  M_driver_driver_t handle = driver->cast();
  for (size_t i = 0; i < driver->cudaContexts.size(); i++) {
    if (driver->cudaContextsValid[i]) {
      M_driver_status_t status =
          M_driver_context_destroy(handle, static_cast<M_driver_context_t>(i));
      if (status != M_DRIVER_STATUS_SUCCESS)
        result = status;
    }
  }
  delete driver;
  return result;
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
  *outDriver = driver->cast();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_destroy(M_driver_driver_t handle) {
  M_driver_status_t status = Driver::destroy(Driver::from(handle));
  return status;
}

M_driver_status_t M_driver_property(const char *property_name, void *value) {
  (void)property_name;
  (void)value;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
