//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides access to the NVML API.
//
//===----------------------------------------------------------------------===//

#include "NVML.h"
#include "Support/PlatformUtils.h"
#include "llvm/Support/raw_ostream.h"
#include <dlfcn.h>
#include <sstream>
#include <stddef.h>
#include <thread>

namespace M::Driver {

#define CUDA_NVML_CALL_LOG()                                                   \
  if (doCallLog) {                                                             \
    /* Make sure we emit a single entry per log call. */                       \
    std::ostringstream buffer;                                                 \
    buffer << "THR-" << std::this_thread::get_id()                             \
           << ": NVML::" << __FUNCTION__ << "\n";                              \
    llvm::errs() << buffer.str();                                              \
  }

#define GET_FUNC(name)                                                         \
  nvml->name##_ = reinterpret_cast<t##name *>(dlsym(nvmlLib, #name));          \
  if (nvml->name##_ == nullptr) {                                              \
    return Error(llvm::Twine("Failed to find symbol \"") + #name +             \
                 "\" in \"" + nvmlLibName + "\"");                             \
  }

#if MODULAR_LINUX
const char nvmlLibName[] = "libnvidia-ml.so.1";
#else
const char nvmlLibName[] = "";
#endif

ErrorOr<const NVML &> NVML::allocAndInitNVML() {
  // Attempt to open the NVML shared library.
  void *nvmlLib = dlopen(nvmlLibName, RTLD_NOW);
  if (nvmlLib == nullptr) {
    if (strcmp(nvmlLibName, "") == 0) {
      return Error("NVML library name is not defined for this platform");
    }
    return Error(llvm::Twine("Failed to open library \"") + nvmlLibName + "\"");
  }

  NVML *nvml = new NVML();

  GET_FUNC(nvmlInit_v2);
  GET_FUNC(nvmlShutdown);
  GET_FUNC(nvmlErrorString);
  GET_FUNC(nvmlDeviceGetHandleByIndex);
  GET_FUNC(nvmlDeviceGetCudaComputeCapability);

  GET_FUNC(nvmlSystemGetDriverVersion);

  auto result = nvml->nvmlInit_v2_();
  if (result != SUCCESS) {
    return Error(llvm::Twine("Failed to initialize NVML: ") +
                 std::to_string(result));
  }

  return *nvml;
}

NVML::~NVML() {
  CUDA_NVML_CALL_LOG();
  nvmlShutdown_();
}

ErrorOr<std::string> NVML::GetDriverVersion() const {
  CUDA_NVML_CALL_LOG();

  // The maximum version buffer size for the NVML system driver version.
  constexpr int nvmlSystemDriverVersionBufferSize = 80;
  std::string driverVersion(nvmlSystemDriverVersionBufferSize, '\0');

  auto result = nvmlSystemGetDriverVersion_(&driverVersion[0],
                                            nvmlSystemDriverVersionBufferSize);
  if (result != SUCCESS) {
    return Error(llvm::Twine("Failed to get driver version: ") +
                 nvmlErrorString_(result));
  }

  return driverVersion;
}

ErrorOr<const NVML &> NVML::get() {
  static const ErrorOr<const NVML &> errOrNVML = allocAndInitNVML();

  if (errOrNVML) {
    return Error(errOrNVML.getError());
  }
  return errOrNVML.get();
}

} // namespace M::Driver
