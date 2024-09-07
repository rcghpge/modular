//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CUDA driver.
//
//===----------------------------------------------------------------------===//

#include "CUDADriver.h"

#include <dlfcn.h>
#include <iostream>
#include <stdlib.h>

namespace M::AsyncRT {

#if defined(__APPLE__) || defined(__MACOSX)
const char CUDADriver::lib_name[] = "/usr/local/cuda/lib/libcuda.dylib";
#elif defined(__unix__)
const char CUDADriver::lib_name[] = "libcuda.so.1";
#else
#error Unsupported platform
#endif

#define GET_FUNC(name)                                                         \
  driver->name##_ = reinterpret_cast<t##name *>(dlsym(dl_lib, #name));         \
  if (driver->name##_ == nullptr) {                                            \
    std::cerr << "Failed to find symbol \"" << #name << "\" in \"" << lib_name \
              << "\"\n";                                                       \
    return nullptr;                                                            \
  }

const CUDADriver *CUDADriver::allocAndInitDriver() {
  // Attempt to open the shared library of the CUDA driver.
  void *dl_lib = dlopen(lib_name, RTLD_NOW);
  if (dl_lib == nullptr) {
    std::cerr << "Failed to open library \"" << lib_name << "\"\n";
    return nullptr;
  }

  // Allocate a driver and register all of the used shared library entry points.
  CUDADriver *driver = new CUDADriver();
  GET_FUNC(cuInit);
  GET_FUNC(cuDeviceGetName);
  GET_FUNC(cuCtxCreate_v2);
  GET_FUNC(cuCtxDestroy);
  GET_FUNC(cuStreamCreate);
  GET_FUNC(cuStreamDestroy);
  GET_FUNC(cuStreamSynchronize);
  GET_FUNC(cuMemAllocHost_v2);
  GET_FUNC(cuMemFreeHost);
  GET_FUNC(cuMemAllocAsync);
  GET_FUNC(cuMemFreeAsync);
  GET_FUNC(cuMemcpyHtoDAsync_v2);
  GET_FUNC(cuMemcpyDtoHAsync_v2);
  GET_FUNC(cuMemcpyDtoDAsync_v2);
  GET_FUNC(cuModuleLoadData);
  GET_FUNC(cuModuleUnload);
  GET_FUNC(cuModuleGetFunction);
  GET_FUNC(cuLaunchKernel);

  // Initialize the driver and return the newly allocated driver if successful.
  Result err = driver->Init(0);
  if (err != SUCCESS) {
    std::cerr << "Initializing CUDA Driver failed: " << err << "\n";
  }

  return driver;
}

const CUDADriver *CUDADriver::get() {
  static const CUDADriver *driver = allocAndInitDriver();

  return driver;
}

} // namespace M::AsyncRT
