//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the HIP compilation device for compile-only mode.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_LIB_DEVICECONTEXT_HIP_HIPCOMPILATIONDEVICE_H
#define ASYNCRT_LIB_DEVICECONTEXT_HIP_HIPCOMPILATIONDEVICE_H

#include "MLRT/AsyncRT/DeviceContext/CompilationDevice.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

#include <memory>
#include <string>

namespace M::AsyncRT {

class DriverVersion;

/// HIP-specific compilation device for compile-only mode.
///
/// Provides HIP device information for compilation without requiring GPU
/// hardware or hipInit(). Unlike CUDA, HIP does not require a runtime
/// compilation step - KGEN handles all compilation through the LLVM AMDGPU
/// backend. This device primarily exists to store target architecture
/// information for the compiler.
///
/// Example usage:
///   auto device = HIPCompilationDevice::create("gfx942");
///   auto arch = device->getArchName();  // Returns "gfx942"
class HIPCompilationDevice : public CompilationDevice {
public:
  /// Create a HIP compilation device for the specified target architecture.
  ///
  /// \param targetArch Target architecture (e.g., "gfx942", "gfx90a")
  /// \return Compilation device or error
  static ErrorOr<std::unique_ptr<HIPCompilationDevice>>
  create(StringRef targetArch);

  ~HIPCompilationDevice() override;

  ErrorOr<int> getApiVersion() const override;
  ErrorOr<DriverVersion &> getDriverVersion() const override;
  ErrorOr<const char *> getArchName() const override;

  /// HIP does not require runtime compilation like CUDA.
  ///
  /// For HIP, all compilation from LLVM IR to device code (.so with HSACO)
  /// is handled by KGEN using the LLVM AMDGPU backend and ld.lld linker.
  /// This method should not be called for HIP.
  ErrorOr<M::BufferRef>
  compileFunction(StringRef ir, StringRef debugLevel, int32_t optimizationLevel,
                  StringRef targetAccelerator) const override;

private:
  HIPCompilationDevice(std::string targetArch);
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_HIP_HIPCOMPILATIONDEVICE_H
