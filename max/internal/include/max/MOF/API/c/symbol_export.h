//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#if (defined(_WIN32) || defined(__CYGWIN__))
#define MODULAR_API_VISIBILITY_EXPORT __declspec(dllexport)
#else
#define MODULAR_API_VISIBILITY_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#define MODULAR_API_EXPORT extern "C" MODULAR_API_VISIBILITY_EXPORT
#else
#define MODULAR_API_EXPORT MODULAR_API_VISIBILITY_EXPORT
#endif
