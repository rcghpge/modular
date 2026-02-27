//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CUDA_DRIVER_H
#define TARGET_LIB_CUDA_DRIVER_H

#include "HAL/M_driver_common.h"

#include "cuda.h"

#include "llvm/ADT/SmallVector.h"

struct Driver {
  [[nodiscard]] static Driver *create();
  static M_driver_status_t destroy(Driver *driver);

  M_driver_driver_t cast() { return reinterpret_cast<M_driver_driver_t>(this); }
  static Driver *from(M_driver_driver_t handle) {
    return reinterpret_cast<Driver *>(handle);
  }

  llvm::SmallVector<CUcontext> cudaContexts{};
  llvm::SmallVector<bool> cudaContextsValid{};
};

#endif // TARGET_LIB_CUDA_DRIVER_H
