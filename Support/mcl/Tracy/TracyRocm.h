//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides a singleton manager for the Tracy ROCm profiling context.
// The Tracy ROCm context is initialized once when the first HIP device context
// is created.
//
// Unlike the CUDA backend (which uses explicit TracyCUDAContext() calls), the
// ROCm backend uses rocprofiler-sdk's tool registration model: Tracy exports
// rocprofiler_configure() as an extern "C" symbol, and the rocprofiler runtime
// discovers and calls it automatically when librocprofiler-sdk.so is loaded.
// This singleton ensures the library loading chain is triggered at the right
// time.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROFILING_TRACYROCM_H
#define SUPPORT_PROFILING_TRACYROCM_H

#if TRACY_ROCM_ENABLE

namespace M {

/// Singleton manager for Tracy ROCm profiling context.
/// Exists for API symmetry with TracyCudaContext; actual initialization is
/// driven by the __attribute__((constructor)) in TracyRocm.cpp which promotes
/// libTracy.so to RTLD_GLOBAL for rocprofiler tool discovery.  Unlike the CUDA
/// backend, the ROCm backend does not require explicit context creation —
/// the rocprofiler runtime handles that via the tool registration callback.
class TracyRocmContext {
public:
  /// Get the singleton instance, triggering initialization if needed.
  /// This is thread-safe (Meyers singleton).
  static TracyRocmContext &getInstance();

  // Delete copy/move constructors and assignment operators.
  TracyRocmContext(const TracyRocmContext &) = delete;
  TracyRocmContext &operator=(const TracyRocmContext &) = delete;
  TracyRocmContext(TracyRocmContext &&) = delete;
  TracyRocmContext &operator=(TracyRocmContext &&) = delete;

private:
  TracyRocmContext();
  ~TracyRocmContext();
};

/// Initialize the Tracy ROCm context singleton.
/// Safe to call multiple times; initialization only happens once.
/// This should be called when the first HIP device context is created.
inline void initTracyRocm() { TracyRocmContext::getInstance(); }

} // namespace M

#else // !TRACY_ROCM_ENABLE

namespace M {

// No-op implementation when Tracy ROCm support is disabled (e.g. on macOS or
// when building without an AMD GPU target).
inline void initTracyRocm() {}

} // namespace M

#endif // TRACY_ROCM_ENABLE

#endif // SUPPORT_PROFILING_TRACYROCM_H
