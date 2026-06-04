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

#include <cstdio>
#include <dlfcn.h>

namespace M::Target::CUDA {

DriverResult<NVML> NVML::load() {
  void *lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!lib) {
    fprintf(stderr, "CUDAPlugin: dlopen(libnvidia-ml.so.1) failed: %s\n",
            dlerror());
    return tl::unexpected(M_DRIVER_STATUS_NO_DEVICES);
  }

  NVML nvml{};
  nvml.lib = lib;

#define NVML_FN(name)                                                          \
  nvml.name = reinterpret_cast<decltype(&::name)>(dlsym(lib, #name));          \
  if (!nvml.name) {                                                            \
    fprintf(stderr, "CUDAPlugin: dlsym(%s) failed: %s\n", #name, dlerror());   \
    nvml.unload();                                                             \
    return tl::unexpected(M_DRIVER_STATUS_NO_DEVICES);                         \
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
