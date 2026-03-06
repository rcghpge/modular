//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CUDA_DRIVER_H
#define TARGET_LIB_CUDA_DRIVER_H

#include "HAL/M_driver_common.h"

#include "CUDALoader.h"
#include "Context.h"
#include "Device.h"
#include "NVML.h"

#include "llvm/ADT/SmallVector.h"

using M::Target::CUDA::CUDALoader;
using M::Target::CUDA::NVML;

struct M_driver_driver {
  CUDALoader rawCuda;
  NVML rawNvml;

  M_driver_driver(CUDALoader &&cuda, NVML &&nvml)
      : rawCuda(std::move(cuda)), rawNvml(std::move(nvml)) {}
  ~M_driver_driver();

  M_driver_driver(const M_driver_driver &) = delete;
  M_driver_driver &operator=(const M_driver_driver &) = delete;

  static M_driver_status_t destroy(M_driver_driver_t driver);

  CUDALoader &cuda() { return rawCuda; }
  NVML &nvml() { return rawNvml; }

  llvm::SmallVector<M_driver_device> devices;

  M_driver_device *findDevice(CUdevice id);
};

#endif // TARGET_LIB_CUDA_DRIVER_H
