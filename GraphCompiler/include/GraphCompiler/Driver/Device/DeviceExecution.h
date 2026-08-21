//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHCOMPILER_DRIVER_DEVICE_DEVICEEXECUTION_H
#define GRAPHCOMPILER_DRIVER_DEVICE_DEVICEEXECUTION_H

#include "MLRT/Core/Tensor/Tensor.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/Context.h"
#include "Support/ML/TensorSpec.h"
#include "Support/SymbolExport.h"

namespace M::GraphCompiler::Driver {

/// Allocates a fresh device buffer for `spec` directly against `device`,
/// applying the debug allocation poisoning requested by
/// `MODULAR_DEBUG_DEVICE_ALLOCATOR=uninitialized-poison`. Replaces
/// `Device::createDeviceMemory(spec, hostAlloc)`.
MODULAR_VISIBILITY_EXPORT
ErrorOr<M::Driver::DeviceBufferRef>
allocateDeviceBuffer(const M::Driver::DeviceContextRef &device,
                     const TensorSpec &spec, bool hostAlloc = false);

/// Wraps `byteSize` bytes at `ptr` in a buffer that does not own the memory:
/// ownership stays with the caller, who must keep it alive for at least as
/// long as the returned buffer. `~DeviceBuffer` returns immediately for a
/// non-owning buffer, so nothing is freed.
///
/// Prefer this over `createBufferOwning(ptr, n, /*owning=*/true)` paired with
/// a no-op `setDestructorFn`: that reaches the same "do not free" outcome by a
/// later branch of the destructor, while reading as the opposite of what it
/// means.
MODULAR_VISIBILITY_EXPORT M::Driver::DeviceBufferRef
borrowBuffer(const M::Driver::DeviceContextRef &device, void *ptr,
             size_t byteSize);

// Construct a device buffer to hold the provided tensor, allocated directly
// against `device` (an MLRT device context) rather than through the
// GenericML `Device`/`DeviceMemory` wrapper.
MODULAR_VISIBILITY_EXPORT
ErrorOr<M::Driver::DeviceBufferRef>
constructDeviceMemoryFromTensor(M::ContextRef context,
                                M::Driver::DeviceContextRef device,
                                M::MLRT::Tensor tensor);

/// Transforms a DeviceTensor representation into an AnyAsyncValueRef executable
/// returned by an ExecutableModel.
MODULAR_VISIBILITY_EXPORT AnyAsyncValueRef
prepareInput(CPUDevice &cpuDevice, M::Driver::DeviceBufferRef inputBuffer,
             TensorSpec inputSpec);

} // namespace M::GraphCompiler::Driver

#endif
