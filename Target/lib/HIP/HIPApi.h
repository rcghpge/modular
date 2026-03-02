//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Wraps the HIP C API as loaded through `dlopen`.
// Types come from HIP itself, but main HIP libraries only exist at runtime.
// Using dlopen eliminates link-time dependency on libamdhip64.so.
//
// To add a new HIP API call, add a single entry to HIP_FORALL_FUNCTIONS.

#ifndef HIPAPI_H
#define HIPAPI_H

#include "hip/hip_runtime_api.h"

#include <dlfcn.h>
#include <optional>
#include <stdio.h>

// Add a `HIP_FN(<fn_name>)` for each fn to dynamically link and
// hold a member for. `HIP_FN` derives type information from decltype,
// so no additional work needs to be done to keep types in sync.
#define HIP_FORALL_FUNCTIONS(HIP_FN)                                           \
  HIP_FN(hipInit)                                                              \
  HIP_FN(hipGetDeviceCount)

struct HIPApi {
  // dlopen handle to loaded libamdhip64.so.
  void *lib = nullptr;

  // Declare member fn ptrs, derive types from corresponding decls in HIP
  // headers
#define DECLARE_FN_PTR(name) decltype(&::name) name = nullptr;
  HIP_FORALL_FUNCTIONS(DECLARE_FN_PTR)
#undef DECLARE_FN_PTR

  // HIPApi owns the dlopen handle; copying would produce two owners that both
  // call dlclose on destruction.
  HIPApi(const HIPApi &) = delete;
  HIPApi &operator=(const HIPApi &) = delete;
  HIPApi(HIPApi &&) = default;
  HIPApi &operator=(HIPApi &&) = default;
  HIPApi() = default;

  /// Try to dlopen libamdhip64.so and resolve all symbols.
  /// Returns a HIPApi on success, or std::nullopt on failure.
  static std::optional<HIPApi> load() {
    void *lib = dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!lib) {
      fprintf(stderr, "HIPPlugin: dlopen(libamdhip64.so) failed: %s\n",
              dlerror());
      return std::nullopt;
    }

    auto api = HIPApi();
    api.lib = lib;

#define LOAD_SYM(name)                                                         \
  api.name = reinterpret_cast<decltype(api.name)>(dlsym(lib, #name));          \
  if (!api.name) {                                                             \
    fprintf(stderr, "HIPPlugin: dlsym(%s) failed: %s\n", #name, dlerror());    \
    api.unload();                                                              \
    return std::nullopt;                                                       \
  }

    HIP_FORALL_FUNCTIONS(LOAD_SYM)

#undef LOAD_SYM

    return api;
  }

  /// Close the library handle. Call only after all HIP usage is done.
  void unload() {
    if (lib) {
      dlclose(lib);
      lib = nullptr;
    }
  }
};

#endif /* HIPAPI_H */
