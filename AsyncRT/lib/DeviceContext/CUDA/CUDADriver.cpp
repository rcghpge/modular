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

#include "Support/ErrorOr.h"
#include "llvm/ADT/Twine.h"

#include <dlfcn.h>
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
    return Error(llvm::Twine("Failed to find symbol \"") + #name +             \
                 "\" in \"" + lib_name + "\"");                                \
  }

ErrorOr<const CUDADriver &> CUDADriver::allocAndInitDriver() {
  // Attempt to open the shared library of the CUDA driver.
  void *dl_lib = dlopen(lib_name, RTLD_NOW);
  if (dl_lib == nullptr) {
    return Error(llvm::Twine("Failed to open library \"") + lib_name + "\"");
  }

  // Allocate a driver and register all of the used shared library entry points.
  CUDADriver *driver = new CUDADriver();
  GET_FUNC(cuInit);
  GET_FUNC(cuDeviceGetCount);
  GET_FUNC(cuDeviceGetName);
  GET_FUNC(cuCtxCreate_v2);
  GET_FUNC(cuCtxDestroy);
  GET_FUNC(cuCtxSetCurrent);
  GET_FUNC(cuStreamCreate);
  GET_FUNC(cuStreamDestroy);
  GET_FUNC(cuStreamSynchronize);
  GET_FUNC(cuMemAllocHost_v2);
  GET_FUNC(cuMemFreeHost);
  GET_FUNC(cuMemAllocAsync);
  GET_FUNC(cuMemFreeAsync);
  GET_FUNC(cuMemAlloc);
  GET_FUNC(cuMemFree);
  GET_FUNC(cuMemcpyHtoDAsync_v2);
  GET_FUNC(cuMemcpyDtoHAsync_v2);
  GET_FUNC(cuMemcpyDtoDAsync_v2);
  GET_FUNC(cuMemcpyPeerAsync);
  GET_FUNC(cuMemcpyHtoD_v2);
  GET_FUNC(cuMemcpyDtoH_v2);
  GET_FUNC(cuMemcpyDtoD_v2);
  GET_FUNC(cuMemcpyPeer);
  GET_FUNC(cuModuleLoadData);
  GET_FUNC(cuModuleUnload);
  GET_FUNC(cuModuleGetFunction);
  GET_FUNC(cuLaunchKernel);
  GET_FUNC(cuEventCreate);
  GET_FUNC(cuEventDestroy);
  GET_FUNC(cuEventRecord);
  GET_FUNC(cuEventQuery);
  GET_FUNC(cuEventSynchronize);
  GET_FUNC(cuEventElapsedTime);
  GET_FUNC(cuMemGetInfo);

  // Initialize the driver and return the newly allocated driver if successful.
  Result err = driver->Init(0);
  if (err != SUCCESS) {
    delete driver;
    return Error(llvm::Twine("Call to cuInit failed with ") + llvm::Twine(err));
  }

  return *driver;
}

ErrorOr<const CUDADriver &> CUDADriver::get() {
  static ErrorOr<const CUDADriver &> err_or_driver = allocAndInitDriver();

  if (err_or_driver) {
    return Error(err_or_driver.getError());
  }
  return err_or_driver.get();
}

} // namespace M::AsyncRT
