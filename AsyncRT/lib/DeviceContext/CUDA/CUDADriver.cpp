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
#include "Support/PlatformUtils.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#include <dlfcn.h>
#include <stdlib.h>

namespace M::AsyncRT {

#ifdef MODULAR_LINUX
const char CUDADriver::lib_name[] = "libcuda.so.1";
#else
const char CUDADriver::lib_name[] = "";
#endif

#define GET_FUNC(name)                                                         \
  driver->name##_ = reinterpret_cast<t##name *>(dlsym(dlLib, #name));          \
  if (driver->name##_ == nullptr) {                                            \
    return Error(llvm::Twine("Failed to find symbol \"") + #name +             \
                 "\" in \"" + lib_name + "\"");                                \
  }

ErrorOr<std::unique_ptr<CUDADriver>> CUDADriver::allocAndInitDriver() {
  // Attempt to open the shared library of the CUDA driver.
  // We use RTLD_GLOBAL to allow UCX CUDA backend to also use these loaded CUDA
  // symbols.
  void *dlLib = dlopen(lib_name, RTLD_NOW | RTLD_GLOBAL);
  if (dlLib == nullptr) {
    return Error(llvm::Twine("Failed to open library \"") + lib_name + "\"");
  }

  // Allocate a driver and register all of the used shared library entry points.
  auto driver = std::make_unique<CUDADriver>();
  GET_FUNC(cuInit);

  GET_FUNC(cuGetErrorName);
  GET_FUNC(cuGetErrorString);

  GET_FUNC(cuDriverGetVersion);

  GET_FUNC(cuDeviceGet);
  GET_FUNC(cuDeviceGetMemPool);
  GET_FUNC(cuDeviceGetAttribute);
  GET_FUNC(cuDeviceGetCount);
  GET_FUNC(cuDeviceGetName);
  GET_FUNC(cuDeviceCanAccessPeer);
  GET_FUNC(cuDeviceGetP2PAttribute);

  GET_FUNC(cuDevicePrimaryCtxRetain);
  GET_FUNC(cuDevicePrimaryCtxRelease_v2);

  GET_FUNC(cuCtxCreate_v2);
  GET_FUNC(cuCtxDestroy);
  GET_FUNC(cuCtxSetCurrent);
  GET_FUNC(cuCtxGetCurrent);
  GET_FUNC(cuCtxPushCurrent);
  GET_FUNC(cuCtxPopCurrent);
  GET_FUNC(cuCtxEnablePeerAccess);
  GET_FUNC(cuCtxGetStreamPriorityRange);

  GET_FUNC(cuStreamCreate);
  GET_FUNC(cuStreamCreateWithPriority);
  GET_FUNC(cuStreamDestroy);
  GET_FUNC(cuStreamSynchronize);
  GET_FUNC(cuStreamWaitEvent);

  GET_FUNC(cuMemAllocHost_v2);
  GET_FUNC(cuMemFreeHost);
  GET_FUNC(cuMemAllocAsync);
  GET_FUNC(cuMemFreeAsync);
  GET_FUNC(cuMemAlloc_v2);
  GET_FUNC(cuMemFree_v2);

  GET_FUNC(cuMemPoolSetAccess);
  GET_FUNC(cuMemPoolTrimTo);

  GET_FUNC(cuMemcpyHtoDAsync_v2);
  GET_FUNC(cuMemcpyDtoHAsync_v2);
  GET_FUNC(cuMemcpyDtoDAsync_v2);
  GET_FUNC(cuMemcpyPeerAsync);
  GET_FUNC(cuMemcpyHtoD_v2);
  GET_FUNC(cuMemcpyDtoH_v2);
  GET_FUNC(cuMemcpyDtoD_v2);
  GET_FUNC(cuMemcpyPeer);

  GET_FUNC(cuMemsetD8Async);
  GET_FUNC(cuMemsetD16Async);
  GET_FUNC(cuMemsetD32Async);
  GET_FUNC(cuMemsetD2D32Async);
  GET_FUNC(cuMemsetD8_v2);
  GET_FUNC(cuMemsetD16_v2);
  GET_FUNC(cuMemsetD32_v2);
  GET_FUNC(cuMemsetD2D32_v2)

  GET_FUNC(cuModuleLoadDataEx);
  GET_FUNC(cuModuleGetGlobal_v2);
  GET_FUNC(cuModuleUnload);
  GET_FUNC(cuModuleGetFunction);
  GET_FUNC(cuFuncGetAttribute);
  GET_FUNC(cuFuncSetAttribute);
  GET_FUNC(cuLaunchHostFunc);
  GET_FUNC(cuLaunchKernel);
  GET_FUNC(cuLaunchKernelEx);

  GET_FUNC(cuEventCreate);
  GET_FUNC(cuEventDestroy);
  GET_FUNC(cuEventRecord);
  GET_FUNC(cuEventQuery);
  GET_FUNC(cuEventSynchronize);
  GET_FUNC(cuEventElapsedTime);

  GET_FUNC(cuMemGetInfo_v2);

  GET_FUNC(cuMemGetAllocationGranularity);
  GET_FUNC(cuMemCreate);
  GET_FUNC(cuMemRelease);
  GET_FUNC(cuMemAddressReserve);
  GET_FUNC(cuMemAddressFree);
  GET_FUNC(cuMemMap);
  GET_FUNC(cuMemUnmap);
  GET_FUNC(cuMemSetAccess);

  GET_FUNC(cuMulticastCreate);
  GET_FUNC(cuMulticastGetGranularity);
  GET_FUNC(cuMulticastAddDevice);
  GET_FUNC(cuMulticastBindMem);
  GET_FUNC(cuMulticastUnbind);

  GET_FUNC(cuTensorMapEncodeTiled);
  GET_FUNC(cuPointerGetAttributes);

  GET_FUNC(cuOccupancyMaxActiveBlocksPerMultiprocessor);

  // Initialize the driver and return the newly allocated driver if successful.
  Result err = driver->Init(0);
  if (err != SUCCESS) {
    return Error(llvm::Twine("Call to cuInit failed with ") + llvm::Twine(err));
  }

  return driver;
}

ErrorOr<const CUDADriver &> CUDADriver::get() {
  static ErrorOr<std::unique_ptr<CUDADriver>> errOrDriver =
      allocAndInitDriver();

  if (errOrDriver) {
    return Error(errOrDriver.getError());
  }

  return *errOrDriver.get();
}

} // namespace M::AsyncRT
