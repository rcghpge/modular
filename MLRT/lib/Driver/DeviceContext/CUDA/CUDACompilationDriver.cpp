//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CUDA compilation driver.
//
//===----------------------------------------------------------------------===//

#include "CUDACompilationDriver.h"

#include "Support/ErrorOr.h"
#include "llvm/ADT/Twine.h"

namespace M::Driver {

#ifdef MODULAR_LINUX

CUDACompilationDriver::Result
CUDACompilationDriver::NVPTXCompilerGetVersion(unsigned int *major,
                                               unsigned int *minor) {
  auto result = ::nvPTXCompilerGetVersion(major, minor);
  return static_cast<Result>(result);
}

CUDACompilationDriver::Result CUDACompilationDriver::NVPTXCompilerCreate(
    ::nvPTXCompilerHandle *compiler, size_t ptxCodeLen, const char *ptxCode) {
  auto result = ::nvPTXCompilerCreate(compiler, ptxCodeLen, ptxCode);
  return static_cast<Result>(result);
}

CUDACompilationDriver::Result
CUDACompilationDriver::NVPTXCompilerDestroy(::nvPTXCompilerHandle *compiler) {
  auto result = ::nvPTXCompilerDestroy(compiler);
  return static_cast<Result>(result);
}

CUDACompilationDriver::Result
CUDACompilationDriver::NVPTXCompilerCompile(::nvPTXCompilerHandle compiler,
                                            int numCompileOptions,
                                            const char *const *compileOptions) {
  auto result =
      ::nvPTXCompilerCompile(compiler, numCompileOptions, compileOptions);
  return static_cast<Result>(result);
}

CUDACompilationDriver::Result
CUDACompilationDriver::NVPTXCompilerGetCompiledProgramSize(
    ::nvPTXCompilerHandle compiler, size_t *binaryImageSize) {
  auto result =
      ::nvPTXCompilerGetCompiledProgramSize(compiler, binaryImageSize);
  return static_cast<Result>(result);
}

CUDACompilationDriver::Result
CUDACompilationDriver::NVPTXCompilerGetCompiledProgram(
    ::nvPTXCompilerHandle compiler, void *binaryImage) {
  auto result = ::nvPTXCompilerGetCompiledProgram(compiler, binaryImage);
  return static_cast<Result>(result);
}

CUDACompilationDriver::Result
CUDACompilationDriver::NVPTXCompilerGetErrorLogSize(
    ::nvPTXCompilerHandle compiler, size_t *errorLogSize) {
  auto result = ::nvPTXCompilerGetErrorLogSize(compiler, errorLogSize);
  return static_cast<Result>(result);
}

CUDACompilationDriver::Result
CUDACompilationDriver::NVPTXCompilerGetErrorLog(::nvPTXCompilerHandle compiler,
                                                char *errorLog) {
  auto result = ::nvPTXCompilerGetErrorLog(compiler, errorLog);
  return static_cast<Result>(result);
}

CUDACompilationDriver::Result
CUDACompilationDriver::NVPTXCompilerGetInfoLogSize(
    ::nvPTXCompilerHandle compiler, size_t *infoLogSize) {
  auto result = ::nvPTXCompilerGetInfoLogSize(compiler, infoLogSize);
  return static_cast<Result>(result);
}

CUDACompilationDriver::Result
CUDACompilationDriver::NVPTXCompilerGetInfoLog(::nvPTXCompilerHandle compiler,
                                               char *infoLog) {
  auto result = ::nvPTXCompilerGetInfoLog(compiler, infoLog);
  return static_cast<Result>(result);
}

#else // !MODULAR_LINUX

// Stub implementations for non-Linux platforms
CUDACompilationDriver::Result
CUDACompilationDriver::NVPTXCompilerGetVersion(unsigned int *major,
                                               unsigned int *minor) {
  return ERROR;
}

CUDACompilationDriver::Result CUDACompilationDriver::NVPTXCompilerCreate(
    ::nvPTXCompilerHandle *compiler, size_t ptxCodeLen, const char *ptxCode) {
  return ERROR;
}

CUDACompilationDriver::Result
CUDACompilationDriver::NVPTXCompilerDestroy(::nvPTXCompilerHandle *compiler) {
  return ERROR;
}

CUDACompilationDriver::Result
CUDACompilationDriver::NVPTXCompilerCompile(::nvPTXCompilerHandle compiler,
                                            int numCompileOptions,
                                            const char *const *compileOptions) {
  return ERROR;
}

CUDACompilationDriver::Result
CUDACompilationDriver::NVPTXCompilerGetCompiledProgramSize(
    ::nvPTXCompilerHandle compiler, size_t *binaryImageSize) {
  return ERROR;
}

CUDACompilationDriver::Result
CUDACompilationDriver::NVPTXCompilerGetCompiledProgram(
    ::nvPTXCompilerHandle compiler, void *binaryImage) {
  return ERROR;
}

CUDACompilationDriver::Result
CUDACompilationDriver::NVPTXCompilerGetErrorLogSize(
    ::nvPTXCompilerHandle compiler, size_t *errorLogSize) {
  return ERROR;
}

CUDACompilationDriver::Result
CUDACompilationDriver::NVPTXCompilerGetErrorLog(::nvPTXCompilerHandle compiler,
                                                char *errorLog) {
  return ERROR;
}

CUDACompilationDriver::Result
CUDACompilationDriver::NVPTXCompilerGetInfoLogSize(
    ::nvPTXCompilerHandle compiler, size_t *infoLogSize) {
  return ERROR;
}

CUDACompilationDriver::Result
CUDACompilationDriver::NVPTXCompilerGetInfoLog(::nvPTXCompilerHandle compiler,
                                               char *infoLog) {
  return ERROR;
}

#endif // MODULAR_LINUX

} // namespace M::Driver
