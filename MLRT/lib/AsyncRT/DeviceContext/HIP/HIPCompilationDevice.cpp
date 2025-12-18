//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the HIP compilation device for compile-only mode.
//
//===----------------------------------------------------------------------===//

#include "HIPCompilationDevice.h"

#include "MLRT/AsyncRT/DeviceContext/DeviceContext.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

using namespace llvm;

namespace M::AsyncRT {

HIPCompilationDevice::HIPCompilationDevice(std::string targetArch)
    : CompilationDevice(Device::hipAPI, std::move(targetArch)) {}

HIPCompilationDevice::~HIPCompilationDevice() = default;

ErrorOr<std::unique_ptr<HIPCompilationDevice>>
HIPCompilationDevice::create(StringRef targetArch) {
  return std::unique_ptr<HIPCompilationDevice>(
      new HIPCompilationDevice(targetArch.str()));
}

ErrorOr<int> HIPCompilationDevice::getApiVersion() const {
  // In compile-only mode, we don't have access to the HIP runtime.
  // Return an error if someone tries to query the API version.
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "HIPCompilationDevice does not have access to HIP runtime in "
      "compile-only mode");
}

ErrorOr<DriverVersion &> HIPCompilationDevice::getDriverVersion() const {
  // In compile-only mode, we don't have access to the HIP driver.
  // Return an error if someone tries to query the driver version.
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "HIPCompilationDevice does not have access to HIP driver in "
      "compile-only mode");
}

ErrorOr<const char *> HIPCompilationDevice::getArchName() const {
  // Return the target architecture that was provided during creation
  return targetArch().c_str();
}

ErrorOr<M::BufferRef>
HIPCompilationDevice::compileFunction(StringRef ir, StringRef debugLevel,
                                      int32_t optimizationLevel,
                                      StringRef targetAccelerator) const {
  // HIP does not require runtime compilation like CUDA.
  // All compilation from LLVM IR to device code is handled by KGEN
  // using the LLVM AMDGPU backend and ld.lld linker.
  // This method should not be called for HIP.
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "HIPCompilationDevice does not support runtime compilation. "
      "All HIP compilation is handled by KGEN through the LLVM AMDGPU "
      "backend.");
}

} // namespace M::AsyncRT
