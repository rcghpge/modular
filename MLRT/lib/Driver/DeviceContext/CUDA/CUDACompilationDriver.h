//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the CUDA compilation driver interface.
//
// CUDACompilationDriver provides access to NVPTX compiler functions
// (libnvptxcompiler.so) without requiring GPU hardware or cuInit().
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDACOMPILATIONDRIVER_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDACOMPILATIONDRIVER_H

#include "Support/ErrorOr.h"
#include "Support/PlatformUtils.h"

#if MODULAR_LINUX
#include "nvPTXCompiler.h"
#else
// Forward declare for non-Linux platforms (in global namespace)
using nvPTXCompilerHandle = struct nvPTXCompiler *;
#endif

namespace M::Driver {

/// CUDACompilationDriver provides access to NVPTX compiler functions.
///
/// This class wraps the statically-linked nvPTXCompiler library and provides
/// static methods to compile PTX to CUBIN without requiring GPU hardware or
/// cuInit().
///
/// The NVPTX compiler library is standalone and does not depend on
/// libcuda.so or any GPU driver initialization.
class CUDACompilationDriver {
public:
  enum Result : unsigned int {
    SUCCESS = 0,
    ERROR = 1,
  };

  static Result NVPTXCompilerGetVersion(unsigned int *major,
                                        unsigned int *minor);

  static Result NVPTXCompilerCreate(::nvPTXCompilerHandle *compiler,
                                    size_t ptxCodeLen, const char *ptxCode);

  static Result NVPTXCompilerDestroy(::nvPTXCompilerHandle *compiler);

  static Result NVPTXCompilerCompile(::nvPTXCompilerHandle compiler,
                                     int numCompileOptions,
                                     const char *const *compileOptions);

  static Result
  NVPTXCompilerGetCompiledProgramSize(::nvPTXCompilerHandle compiler,
                                      size_t *binaryImageSize);

  static Result NVPTXCompilerGetCompiledProgram(::nvPTXCompilerHandle compiler,
                                                void *binaryImage);

  static Result NVPTXCompilerGetErrorLogSize(::nvPTXCompilerHandle compiler,
                                             size_t *errorLogSize);

  static Result NVPTXCompilerGetErrorLog(::nvPTXCompilerHandle compiler,
                                         char *errorLog);

  static Result NVPTXCompilerGetInfoLogSize(::nvPTXCompilerHandle compiler,
                                            size_t *infoLogSize);

  static Result NVPTXCompilerGetInfoLog(::nvPTXCompilerHandle compiler,
                                        char *infoLog);

private:
  // Enable in-depth logging at build time.
  static constexpr bool do_call_log = false;
};

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDACOMPILATIONDRIVER_H
