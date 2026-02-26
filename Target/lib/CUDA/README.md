# CUDA hardware support

This folder hosts the implementation of the Driver-HAL for CUDA.
We implement the C API functions in `SDK/.../HAL/M_*.h` in terms of
calls to CUDA driver API.

## Developer Notes

1. This plugin translates HAL API calls to CUDA driver calls.
2. It is a shared object that the Driver-HAL will `dlopen`/`dlsym`/`dlclose`.
3. It `#include`-s `"cuda.h"` and links to `libcuda.so` like any other regular
   library being built against CUDA. Note how this is different from our
   previous approach where `libcuda.so` was directly being `dlopen`-ed.
4. The plugin shared object name is not given the `lib` prefix:
   plugins are not libraries in the traditional sense. To this end, our bazel
   target is a `binary` and not a `library` and the nomenclature includes an
   explicit `.so` as suffix.
