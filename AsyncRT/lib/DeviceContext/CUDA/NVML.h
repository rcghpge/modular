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

using nvmlDevice_t = struct nvmlDevice_st *;

class NVML {
public:
  enum Result : unsigned int {
    SUCCESS = 0,
  };

  ~NVML();

  ErrorOr<std::string> GetDriverVersion() const;
  Result DeviceGetHandleByIndex(int idx, nvmlDevice_t *dev) const {
    return nvmlDeviceGetHandleByIndex_(idx, dev);
  }
  Result DeviceGetCudaComputeCapability(nvmlDevice_t dev, int *major,
                                        int *minor) const {
    return nvmlDeviceGetCudaComputeCapability_(dev, major, minor);
  }

  static ErrorOr<const NVML &> get();

private:
  // Enable in-depth logging at build time.
  static constexpr bool doCallLog = false;

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

  using tnvmlDeviceGetHandleByIndex = Result(int, nvmlDevice_t *);
  tnvmlDeviceGetHandleByIndex *nvmlDeviceGetHandleByIndex_;

  using tnvmlDeviceGetCudaComputeCapability = Result(nvmlDevice_t, int *,
                                                     int *);
  tnvmlDeviceGetCudaComputeCapability *nvmlDeviceGetCudaComputeCapability_;
};

} // namespace M::AsyncRT
#endif // ASYNCRT_LIB_DEVICECONTEXT_CUDA_NVML_H
