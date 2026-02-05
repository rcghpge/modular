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
#include <string.h>

namespace M::Driver {

#if MODULAR_LINUX
const char CUDADriver::lib_name[] = "libcuda.so.1";
#else
const char CUDADriver::lib_name[] = "";
#endif

ErrorOr<std::unique_ptr<CUDADriver>> CUDADriver::allocAndInitDriver() {
  if (strcmp("", lib_name) == 0) {
    return Error(
        llvm::Twine("Failed to open CUDA library: "
                    "cuda is not supported on this operating system."));
  }

  // We use both NVML and CUDA Driver to access devices. NVML discovers devices
  // in the order of PCI bus ID, so we want to force CUDA to enumerate in the
  // same order.
  // See:
  // [1] Documentation for nvmlDeviceGetHandleByIndex_v2 (non-deterministic
  //     device-ordering between reboots)
  // [2]
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-environment-variables
  setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", /*overwrite=*/1);

  // Attempt to open the shared library of the CUDA driver.
  // We use RTLD_GLOBAL to allow UCX CUDA backend to also use these loaded CUDA
  // symbols.
  void *dlLib = dlopen(lib_name, RTLD_NOW | RTLD_GLOBAL);
  if (dlLib == nullptr) {
    return Error(llvm::Twine("Failed to open library \"") + lib_name + "\"");
  }

  // Allocate a driver and register all of the used shared library entry points.
  auto driver = std::make_unique<CUDADriver>();

#define CUDA_FUNC(name, ...)                                                   \
  driver->name##_ = reinterpret_cast<t##name *>(dlsym(dlLib, #name));          \
  if (driver->name##_ == nullptr) {                                            \
    return Error(llvm::Twine("Failed to find symbol \"") + #name +             \
                 "\" in \"" + lib_name + "\"");                                \
  }

#include "CUDADriver.inc"
#undef CUDA_FUNC

  // Hand-written dlsym for cuTensorMapEncodeIm2col (exceeds macro limit).
  driver->cuTensorMapEncodeIm2col_ =
      reinterpret_cast<tcuTensorMapEncodeIm2col *>(
          dlsym(dlLib, "cuTensorMapEncodeIm2col"));
  if (driver->cuTensorMapEncodeIm2col_ == nullptr) {
    return Error(llvm::Twine("Failed to find symbol "
                             "\"cuTensorMapEncodeIm2col\" in \"") +
                 lib_name + "\"");
  }

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

} // namespace M::Driver
