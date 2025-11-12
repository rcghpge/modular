//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the CompilationDevice interface for compile-only mode.
//
// CompilationDevice provides device functionality needed for compilation
// without requiring actual GPU hardware or driver initialization (cuInit()).
// This enables compiling GPU code on machines without GPUs.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_DEVICECONTEXT_COMPILATIONDEVICE_H
#define ASYNCRT_DEVICECONTEXT_COMPILATIONDEVICE_H

#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

#include <cstdint>
#include <string>

namespace M::AsyncRT {

// Forward declarations
class DriverVersion;

/// CompilationDevice provides compilation functionality without GPU hardware.
///
/// This abstract base class defines the interface for device compilation
/// operations that do not require physical GPU access or driver initialization.
/// Implementations (CUDACompilationDevice, HIPCompilationDevice) provide
/// compilation for specific GPU architectures.
///
/// Key properties:
/// - Does NOT require cuInit() or GPU hardware
/// - Provides target architecture information for compilation
/// - Implements standalone compilation (e.g., NVPTX compiler for CUDA)
/// - Can be used by both real devices and virtual devices
///
/// Usage:
/// - CUDADevice uses CUDACompilationDevice for compilation operations
/// - VirtualDevice uses CUDACompilationDevice to compile without real GPU
class CompilationDevice {
public:
  using API = StringLiteral;

  virtual ~CompilationDevice() = default;

  /// Get the API type (cuda, hip, etc.)
  API api() const { return api_; }

  /// Get the target architecture name (e.g., "sm_80", "gfx90a")
  const std::string &targetArch() const { return targetArch_; }

  /// Get the API version supported by this compilation device
  virtual ErrorOr<int> getApiVersion() const = 0;

  /// Get the driver version (may be simulated for compile-only devices)
  virtual ErrorOr<DriverVersion &> getDriverVersion() const = 0;

  /// Get the architecture name string (e.g., "sm_80")
  virtual ErrorOr<std::string> getArchName() const = 0;

  /// Compile intermediate representation (IR) to device binary.
  ///
  /// For CUDA: Compiles PTX to CUBIN using libnvptxcompiler
  /// For HIP: Compiles to device code using HIP compiler
  ///
  /// This method does NOT require GPU hardware or driver initialization.
  ///
  /// \param ir The intermediate representation code (e.g., PTX for CUDA)
  /// \param debugLevel Debug level for compilation
  /// \param optimizationLevel Optimization level (-1 for default)
  /// \param targetAccelerator Target accelerator override (e.g., "sm_80")
  /// \return Compiled binary code or error
  virtual ErrorOr<M::BufferRef>
  compileFunction(StringRef ir, StringRef debugLevel, int32_t optimizationLevel,
                  StringRef targetAccelerator) const = 0;

protected:
  CompilationDevice(API api, std::string targetArch)
      : api_(api), targetArch_(std::move(targetArch)) {}

private:
  API api_;
  std::string targetArch_;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_DEVICECONTEXT_COMPILATIONDEVICE_H
