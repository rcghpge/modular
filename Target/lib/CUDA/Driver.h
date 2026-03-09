//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CUDA_DRIVER_H
#define TARGET_LIB_CUDA_DRIVER_H

#include "Error.h"
#include "Loader/CUDALoader.h"
#include "Loader/NVML.h"

using M::Target::CUDA::CUDALoader;
using M::Target::CUDA::NVML;

struct M_driver_driver {
private:
  CUDALoader rawCuda;
  NVML rawNvml;

public:
  [[nodiscard]] static DriverResult<M_driver_driver_t> create();
  static M_driver_status_t destroy(M_driver_driver_t driver);

  M_driver_driver(CUDALoader &&cuda, NVML &&nvml)
      : rawCuda(std::move(cuda)), rawNvml(std::move(nvml)) {}
  ~M_driver_driver();
  M_driver_driver(const M_driver_driver &) = delete;
  M_driver_driver &operator=(const M_driver_driver &) = delete;

  CUDALoader &cuda() { return rawCuda; }
  NVML &nvml() { return rawNvml; }
  DriverResult<int64_t> deviceCount();
};

#endif // TARGET_LIB_CUDA_DRIVER_H
