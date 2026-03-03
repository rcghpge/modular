//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the NVML API interface.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CUDA_NVML_H
#define TARGET_LIB_CUDA_NVML_H

#include <nvml.h>
#include <optional>

namespace M::Target::CUDA {

struct NVML {
  void *lib = nullptr;

#define NVML_FN(name) decltype(&::name) name = nullptr;
#include "NVML.inc"
#undef NVML_FN

  NVML(const NVML &) = delete;
  NVML &operator=(const NVML &) = delete;
  NVML(NVML &&) = default;
  NVML &operator=(NVML &&) = default;
  NVML() = default;

  static std::optional<NVML> load();
  void unload();

  /// Returns the NVML driver version string, or nullptr on failure.
  const char *getDriverVersion() const;
};

} // namespace M::Target::CUDA

#endif // TARGET_LIB_CUDA_NVML_H
