//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the HIP driver.
//
//===----------------------------------------------------------------------===//

#include "HIPDriver.h"

#include "Support/ErrorOr.h"
#include "Support/PlatformUtils.h"
#include "llvm/ADT/Twine.h"

#include <dlfcn.h>
#include <stdlib.h>

namespace M::AsyncRT {

#ifdef MODULAR_LINUX
const char HIPDriver::lib_name[] = "libamdhip64.so";
#else
const char HIPDriver::lib_name[] = "";
#endif

#define GET_FUNC(name)                                                         \
  driver->name##_ = reinterpret_cast<t##name *>(dlsym(dl_lib, #name));         \
  if (driver->name##_ == nullptr) {                                            \
    return Error(llvm::Twine("Failed to find symbol \"") + #name +             \
                 "\" in \"" + lib_name + "\"");                                \
  }

ErrorOr<const HIPDriver &> HIPDriver::allocAndInitDriver() {
  // Attempt to open the shared library of the HIP driver.
  void *dl_lib = dlopen(lib_name, RTLD_NOW);
  if (dl_lib == nullptr) {
    return Error(llvm::Twine("Failed to open library \"") + lib_name +
                 "\": " + dlerror());
  }

  // Allocate a driver and register all of the used shared library entry points.
  HIPDriver *driver = new HIPDriver();
  GET_FUNC(hipInit);

  GET_FUNC(hipDrvGetErrorName);
  GET_FUNC(hipDrvGetErrorString);

  GET_FUNC(hipDriverGetVersion);

  GET_FUNC(hipDeviceGetAttribute);
  GET_FUNC(hipGetDeviceCount);
  GET_FUNC(hipSetDevice);
  GET_FUNC(hipDeviceGetName);

  GET_FUNC(hipCtxCreate);
  GET_FUNC(hipCtxDestroy);
  GET_FUNC(hipCtxSetCurrent);

  GET_FUNC(hipStreamCreate);
  GET_FUNC(hipStreamDestroy);
  GET_FUNC(hipStreamSynchronize);
  GET_FUNC(hipStreamWaitEvent);

  GET_FUNC(hipHostMalloc);
  GET_FUNC(hipHostFree);
  GET_FUNC(hipMallocAsync);
  GET_FUNC(hipFreeAsync);
  GET_FUNC(hipMalloc);
  GET_FUNC(hipFree);

  GET_FUNC(hipMemcpyHtoDAsync);
  GET_FUNC(hipMemcpyDtoHAsync);
  GET_FUNC(hipMemcpyDtoDAsync);
  GET_FUNC(hipMemcpyHtoD);
  GET_FUNC(hipMemcpyDtoH);
  GET_FUNC(hipMemcpyDtoD);

  GET_FUNC(hipMemsetD8Async);
  GET_FUNC(hipMemsetD16Async);
  GET_FUNC(hipMemsetD32Async);
  GET_FUNC(hipMemset2DAsync);
  GET_FUNC(hipMemsetD8);
  GET_FUNC(hipMemsetD16);
  GET_FUNC(hipMemsetD32);
  GET_FUNC(hipMemset2D);

  GET_FUNC(hipModuleLoadData);
  GET_FUNC(hipModuleLoadDataEx);
  GET_FUNC(hipModuleGetGlobal);
  GET_FUNC(hipModuleUnload);
  GET_FUNC(hipModuleGetFunction);
  GET_FUNC(hipFuncGetAttribute);
  GET_FUNC(hipModuleLaunchKernel);

  GET_FUNC(hipEventCreateWithFlags);
  GET_FUNC(hipEventDestroy);
  GET_FUNC(hipEventRecord);
  GET_FUNC(hipEventQuery);
  GET_FUNC(hipEventSynchronize);
  GET_FUNC(hipEventElapsedTime);

  GET_FUNC(hipMemGetInfo);

  // Initialize the driver and return the newly allocated driver if successful.
  Result err = driver->Init(0);
  if (err != SUCCESS) {
    const char *errName = nullptr;
    const char *errStr = nullptr;
    driver->GetErrorName(err, &errName);
    driver->GetErrorString(err, &errStr);

    delete driver;
    return Error(llvm::Twine("Call to hipInit failed with ") +
                 llvm::Twine(err) + llvm::Twine(errName) + llvm::Twine(errStr));
  }

  return *driver;
}

ErrorOr<const HIPDriver &> HIPDriver::get() {
  static ErrorOr<const HIPDriver &> errOrDriver = allocAndInitDriver();

  if (errOrDriver) {
    return Error(errOrDriver.getError());
  }
  return errOrDriver.get();
}

} // namespace M::AsyncRT
