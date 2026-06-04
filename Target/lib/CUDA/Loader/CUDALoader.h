//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the CUDA driver interface.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CUDA_CUDALOADER_H
#define TARGET_LIB_CUDA_CUDALOADER_H

#include "Target/lib/CUDA/Error.h"

#include <cuda.h>

namespace M::Target::CUDA {

struct CUDALoader {
  void *lib = nullptr;

#define CUDA_FN(name) decltype(&::name) name = nullptr;
#include "CUDALoader.inc"
#undef CUDA_FN

  CUDALoader(const CUDALoader &) = delete;
  CUDALoader &operator=(const CUDALoader &) = delete;
  CUDALoader(CUDALoader &&) = default;
  CUDALoader &operator=(CUDALoader &&) = default;
  CUDALoader() = default;

  static DriverResult<CUDALoader> load();
  void unload();
};

} // namespace M::Target::CUDA

#endif // TARGET_LIB_CUDA_CUDALOADER_H
