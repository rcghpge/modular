//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#include "Driver.h"

M_driver_driver::~M_driver_driver() {
  rawCuda.unload();
  rawNvml.unload();
}

M_driver_status_t M_driver_driver::destroy(M_driver_driver_t driver) {
  delete driver;
  return M_DRIVER_STATUS_SUCCESS;
}

DriverResult<M_driver_driver_t> M_driver_driver::create() {
  auto nvml = NVML::load();
  if (!nvml)
    return tl::unexpected(M_DRIVER_STATUS_UNKNOWN_ERROR);

  if (nvml->nvmlInit_v2() != NVML_SUCCESS) {
    nvml->unload();
    return tl::unexpected(M_DRIVER_STATUS_UNKNOWN_ERROR);
  }

  auto cuda = CUDALoader::load();
  if (!cuda) {
    nvml->unload();
    return tl::unexpected(M_DRIVER_STATUS_UNKNOWN_ERROR);
  }

  if (cuda->cuInit(0) != CUDA_SUCCESS) {
    cuda->unload();
    nvml->unload();
    return tl::unexpected(M_DRIVER_STATUS_UNKNOWN_ERROR);
  }

  return new M_driver_driver(std::move(*cuda), std::move(*nvml));
}

DriverResult<int64_t> M_driver_driver::deviceCount() {
  unsigned int nvmlCount = 0;
  if (rawNvml.nvmlDeviceGetCount_v2(&nvmlCount) != NVML_SUCCESS)
    return tl::unexpected(M_DRIVER_STATUS_NO_DEVICES);
  return static_cast<int64_t>(nvmlCount);
}
