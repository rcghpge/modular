//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Runtime loading of the NVML library via dlopen+dlsym.
//
//===----------------------------------------------------------------------===//

#include "NVML.h"

#include <dlfcn.h>

namespace M::Target::CUDA {

std::optional<NVML> NVML::load() {
  void *lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!lib)
    return std::nullopt;

  NVML nvml{};
  nvml.lib = lib;

#define NVML_FN(name)                                                          \
  nvml.name = reinterpret_cast<decltype(&::name)>(dlsym(lib, #name));          \
  if (!nvml.name) {                                                            \
    nvml.unload();                                                             \
    return std::nullopt;                                                       \
  }

#include "NVML.inc"
#undef NVML_FN

  return nvml;
}

void NVML::unload() {
  if (lib) {
    nvmlShutdown();
    dlclose(lib);
    lib = nullptr;
  }
}

const char *NVML::getDriverVersion() const {
  static char buf[80];
  if (nvmlSystemGetDriverVersion(buf, sizeof(buf)) != NVML_SUCCESS)
    return nullptr;
  return buf;
}

} // namespace M::Target::CUDA
