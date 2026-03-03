//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CUDA_DRIVER_H
#define TARGET_LIB_CUDA_DRIVER_H

#include "HAL/M_driver_common.h"

#include "CUDALoader.h"
#include "NVML.h"

#include "llvm/ADT/SmallVector.h"

namespace M::Target::CUDA {

class Driver {
  CUDALoader rawCuda;
  NVML rawNvml;

public:
  Driver(CUDALoader &&cuda, NVML &&nvml)
      : rawCuda(std::move(cuda)), rawNvml(std::move(nvml)) {}
  ~Driver();

  Driver(const Driver &) = delete;
  Driver &operator=(const Driver &) = delete;

  static M_driver_status_t destroy(Driver *driver);

  M_driver_driver_t cast() { return reinterpret_cast<M_driver_driver_t>(this); }

  static Driver *from(M_driver_driver_t handle) {
    return reinterpret_cast<Driver *>(handle);
  }

  CUDALoader &cuda() { return rawCuda; }
  NVML &nvml() { return rawNvml; }

  llvm::SmallVector<CUcontext> cudaContexts{};
  llvm::SmallVector<bool> cudaContextsValid{};
};

} // namespace M::Target::CUDA

#endif // TARGET_LIB_CUDA_DRIVER_H
