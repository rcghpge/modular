//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the NVML API interface.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_LIB_DEVICECONTEXT_CUDA_NVML_H
#define ASYNCRT_LIB_DEVICECONTEXT_CUDA_NVML_H

#include "Support/ErrorOr.h"

namespace M::AsyncRT {

class NVML {
public:
  ~NVML();

  ErrorOr<std::string> GetDriverVersion() const;

  static ErrorOr<const NVML &> get();

private:
  // Enable in-depth logging at build time.
  static constexpr bool doCallLog = false;

  enum Result : unsigned int {
    SUCCESS = 0,
  };

  static ErrorOr<const NVML &> allocAndInitNVML();
  static const char lib_name[];

  using tnvmlInit_v2 = Result();
  tnvmlInit_v2 *nvmlInit_v2_;

  using tnvmlShutdown = Result();
  tnvmlShutdown *nvmlShutdown_;

  using tnvmlErrorString = const char *(Result);
  tnvmlErrorString *nvmlErrorString_;

  using tnvmlSystemGetDriverVersion = Result(char *, int);
  tnvmlSystemGetDriverVersion *nvmlSystemGetDriverVersion_;
};

} // namespace M::AsyncRT
#endif // ASYNCRT_LIB_DEVICECONTEXT_CUDA_NVML_H
