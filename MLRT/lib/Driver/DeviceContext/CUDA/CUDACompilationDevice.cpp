//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CUDA compilation device for compile-only mode.
//
//===----------------------------------------------------------------------===//

#include "CUDACompilationDevice.h"

#include "CUDACompilationDriver.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "NVML.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

using namespace llvm;

namespace M::AsyncRT {

CUDACompilationDevice::CUDACompilationDevice(std::string targetArch)
    : CompilationDevice(Device::cudaAPI, std::move(targetArch)) {}

CUDACompilationDevice::~CUDACompilationDevice() = default;

SmallVector<std::string>
CUDACompilationDevice::setupCompilationOptionsNVPTXCompiler(
    StringRef debugLevel, int32_t optimizationLevel, StringRef gpuName) {
  // TODO: add more options needed.
  // https://docs.nvidia.com/cuda/ptx-compiler-api/index.html#compilation-options

  static constexpr int kDefaultOptimizationLevel = 4;

  // The default optimization level for NVIDIA GPUs is 4.
  if (optimizationLevel < 0)
    optimizationLevel = kDefaultOptimizationLevel;

  // The Mojo compiler uses optimization level 3 as the highest optimization
  // level, but 4 is the highest for CUDA.
  if (optimizationLevel == 3)
    optimizationLevel = kDefaultOptimizationLevel;

  SmallVector<std::string, 3> options;
  options.push_back("--gpu-name=" + gpuName.str());

  if (debugLevel == "full") {
    options.push_back("-g");
    optimizationLevel = 0;
  } else if (debugLevel == "line-tables") {
    options.push_back("--generate-line-info");
  }

  options.push_back("--opt-level=" + std::to_string(optimizationLevel));

  return options;
}

ErrorOr<std::unique_ptr<CUDACompilationDevice>>
CUDACompilationDevice::create(StringRef targetArch) {
  return std::unique_ptr<CUDACompilationDevice>(
      new CUDACompilationDevice(targetArch.str()));
}

ErrorOr<int> CUDACompilationDevice::getApiVersion() const {
#ifdef MODULAR_LINUX
  // Get the NVPTX compiler version
  unsigned int majorVer, minorVer;
  auto versionResult =
      CUDACompilationDriver::NVPTXCompilerGetVersion(&majorVer, &minorVer);
  if (versionResult != CUDACompilationDriver::SUCCESS) {
    return Error("Failed to get NVPTX compiler version");
  }

  // Convert NVPTX compiler version to CUDA API version format
  // CUDA version format: major * 1000 + minor * 10
  // Example: NVPTX 12.4 -> 12040
  int apiVersion = majorVer * 1000 + minorVer * 10;
  return apiVersion;
#else
  return Error("NVPTX compiler not supported on this platform");
#endif
}

ErrorOr<DriverVersion &> CUDACompilationDevice::getDriverVersion() const {
  // Initialize the driver version once on first call of this method
  // or remember the error.
  static ErrorOr<std::string> errOrVersion = []() -> ErrorOr<std::string> {
    ErrorOr<const NVML &> errOrNVML = NVML::get();
    if (errOrNVML.isError()) {
      return errOrNVML.takeError();
    }
    ErrorOr<std::string> errOrVersion = errOrNVML->GetDriverVersion();
    if (errOrVersion.isError()) {
      return errOrVersion.takeError();
    }
    return *errOrVersion;
  }();

  // Propagate the error if getting the version produced an error.
  if (errOrVersion.isError()) {
    return Error(errOrVersion.getError());
  }
  // Return the version string.
  static DriverVersion version(*errOrVersion);
  return version;
}

ErrorOr<const char *> CUDACompilationDevice::getArchName() const {
  return targetArch().c_str();
}

ErrorOr<M::BufferRef>
CUDACompilationDevice::compileFunction(StringRef ptx, StringRef debugLevel,
                                       int32_t optimizationLevel,
                                       StringRef targetAccelerator) const {
#ifdef MODULAR_LINUX
  // Use targetAccelerator if provided, otherwise use this device's target arch
  StringRef gpuName =
      targetAccelerator.empty() ? targetArch() : targetAccelerator;

  auto options = CUDACompilationDevice::setupCompilationOptionsNVPTXCompiler(
      debugLevel, optimizationLevel, gpuName);

  // NOTE: NVPTX compiler is a standalone library that doesn't require a CUDA
  // context or GPU to be present. This allows compilation in environments
  // without GPUs (e.g., compile-only mode with virtual devices).

  nvPTXCompilerHandle compiler = nullptr;
  size_t elfSize, infoSize, errorSize;
  unsigned int minorVer, majorVer;

  // Get NVPTX compiler version
  auto versionResult =
      CUDACompilationDriver::NVPTXCompilerGetVersion(&majorVer, &minorVer);
  if (versionResult != CUDACompilationDriver::SUCCESS) {
    return Error("Failed to get NVPTX compiler version");
  }

  // Create NVPTX compiler handle
  auto createResult = CUDACompilationDriver::NVPTXCompilerCreate(
      &compiler, (size_t)ptx.size(), (const char *)ptx.data());
  if (createResult != CUDACompilationDriver::SUCCESS) {
    return Error("Failed to create NVPTX compiler");
  }

  // Setup compilation options
  constexpr int maxNumOptions = 16;
  const char *compileOptions[maxNumOptions];
  for (auto [idx, option] : llvm::enumerate(options)) {
    compileOptions[idx] = option.c_str();
  }

  // Compile PTX to CUBIN
  CUDACompilationDriver::Result result =
      CUDACompilationDriver::NVPTXCompilerCompile(compiler, options.size(),
                                                  compileOptions);

  if (result != CUDACompilationDriver::SUCCESS) {
    // Get error log
    CUDACompilationDriver::NVPTXCompilerGetErrorLogSize(compiler, &errorSize);

    std::string errorLog;
    if (errorSize != 0) {
      errorLog.resize(errorSize + 1);
      CUDACompilationDriver::Result getErrorResult =
          CUDACompilationDriver::NVPTXCompilerGetErrorLog(compiler,
                                                          errorLog.data());

      if (getErrorResult != CUDACompilationDriver::SUCCESS) {
        CUDACompilationDriver::NVPTXCompilerDestroy(&compiler);
        return Error("Compilation failed and could not retrieve error log");
      }
    }

    CUDACompilationDriver::NVPTXCompilerDestroy(&compiler);
    return Error(errorLog + "\nCompilation to CUBIN failed");
  }

  // Get compiled program size
  auto sizeResult = CUDACompilationDriver::NVPTXCompilerGetCompiledProgramSize(
      compiler, &elfSize);
  if (sizeResult != CUDACompilationDriver::SUCCESS) {
    CUDACompilationDriver::NVPTXCompilerDestroy(&compiler);
    return Error("Failed to get compiled program size");
  }

  // Allocate buffer for compiled program
  auto resultBuf = M::WriteableBuffer::get(elfSize);

  // Get compiled program
  auto programResult = CUDACompilationDriver::NVPTXCompilerGetCompiledProgram(
      compiler, (void *)resultBuf->getBufferStart());
  if (programResult != CUDACompilationDriver::SUCCESS) {
    CUDACompilationDriver::NVPTXCompilerDestroy(&compiler);
    return Error("Failed to get compiled program");
  }

  // Check for warnings
  CUDACompilationDriver::NVPTXCompilerGetInfoLogSize(compiler, &infoSize);

  std::optional<std::string> printWarning =
      llvm::sys::Process::GetEnv("MODULAR_NVPTX_COMPILER_WARN");
  std::optional<std::string> werror =
      llvm::sys::Process::GetEnv("MODULAR_NVPTX_COMPILER_WERROR");

  if (infoSize != 0 && (printWarning || werror)) {
    std::string infoLog;
    infoLog.resize(infoSize + 1);
    CUDACompilationDriver::Result getLogResult =
        CUDACompilationDriver::NVPTXCompilerGetInfoLog(compiler,
                                                       infoLog.data());

    if (getLogResult != CUDACompilationDriver::SUCCESS) {
      CUDACompilationDriver::NVPTXCompilerDestroy(&compiler);
      return Error("Compilation has warnings but could not retrieve info log");
    }

    if (werror) {
      CUDACompilationDriver::NVPTXCompilerDestroy(&compiler);
      return Error(infoLog + "\nCompilation failed due to warnings");
    } else {
      llvm::errs() << infoLog << "\n";
    }
  }

  // Clean up compiler
  CUDACompilationDriver::NVPTXCompilerDestroy(&compiler);

  return resultBuf;
#else  // MODULAR_LINUX
  return Error("NVPTX compiler not supported on this platform");
#endif // MODULAR_LINUX
}

} // namespace M::AsyncRT
