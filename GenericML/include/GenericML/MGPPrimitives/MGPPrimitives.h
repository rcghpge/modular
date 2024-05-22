//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPPRIMITIVES_MGPPRIMITIVES_H
#define GENERICML_MGPPRIMITIVES_MGPPRIMITIVES_H

#include "GenericML/Support/CompiledKernelsContext.h"
#include "KGEN/ExecutionEngine/ExecutionEngine.h"
#include "Support/ReferenceCounted.h"

namespace M {
class PrimitiveRegistry;

namespace MGP {

/// Represents the 'runtime' for a CPU device (which is generally also the host
/// device). This is currently distinct from the LLCL::Runtime.
///
/// If we start supporting a device-per-numa-node architecture then this may
/// grow to also hold an allocator and work queue.
struct CPURuntime : public ReferenceCounted<CPURuntime> {
  // The (virtual) device id for this runtime. Ie the id of the #M.device_ref
  // attribute corresponding to this runtime.
  size_t virtualDeviceId;

  // Execution engine held by the runtime and shared by all loaded modules.
  KGEN::ExecutionEngine *engine;

  // The kernel context for each loaded module. Modules are referenced by
  // index into this vector.
  SmallVector<RCRef<GML::CompiledKernelsContext>> modules;

  CPURuntime(size_t virtualDeviceId, KGEN::ExecutionEngine *engine)
      : virtualDeviceId(virtualDeviceId), engine(engine) {}

  ~CPURuntime() {
    if constexpr (ProfilerEntry<Trace::EnableTrace(Trace::kMojo, 1),
                                Trace::kMojo>::isEnabled()) {
      // Mojo static strings are part of the data section of their compiled
      // object files, which are jitted by the `KGEN::ExecutionEngine`.
      // Then the `mgp.load.module.cpu` op loads the object files into this
      // `CPURuntime` (`!mgp.runtime`).
      // However, the profiler borrows pointers to those static string data on
      // Mojo kernel `Trace.__enter__()` and `Trace.__exit__()` calls.
      // Since this dtor is about to unload the object files rendering the
      // string pointers invalid, intern the strings into the profiler so that
      // the strings are accessible upon `TimeTraceProfiler::write()` call.
      if (auto runtime = LLCL::Runtime::getCurrentRuntimeOrNull()) {
        // Make sure any borrowed strings accumulated during profiling are
        // converted to owned strings.
        if (std::optional<TimeTraceProfiler> &profiler = runtime->getProfiler())
          profiler->intern();
      }
    }
  }

  /// Returns the compiled kernel at the given module and kernel indexes.
  GML::CompiledKernelFn *getKernel(size_t moduleIndex,
                                   size_t kernelIndex) const {
    GENERICML_ASSERT(moduleIndex < modules.size())
        << "invalid module index " << moduleIndex;
    return modules[moduleIndex]->getKernel(kernelIndex);
  }
  /// Returns the compiled kernelAsyncABI at the given index.
  GML::CompiledKernelAsyncABIFn *getKernelAsyncABI(size_t moduleIndex,
                                                   size_t kernelIndex) const {
    GENERICML_ASSERT(moduleIndex < modules.size())
        << "invalid module index " << moduleIndex;
    return modules[moduleIndex]->getKernelAsyncABI(kernelIndex);
  }
};

} // namespace MGP

/// Registers the device-generic subset of the 'mgp' dialect primitives for
/// "cpu" devices.
void registerMGPCPUPrimitives(PrimitiveRegistry &registry);

/// Registers all the 'mgp' dialect primitives.
void registerMGPPrimitives(PrimitiveRegistry &registry);

} // namespace M

#endif // GENERICML_MGPPRIMITIVES_MGPPRIMITIVES_H
