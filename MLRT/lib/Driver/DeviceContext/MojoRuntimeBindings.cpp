//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains runtime bindings for non-DeviceContext components.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "AsyncRT/Runtime/AsyncValue.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "MLRT/Core/Tensor/TensorBufferRef.h"
#include "MLRT/Driver/DeviceContext/DeviceAffinity.h"
#include "Support/Profiling/internal/Tracy.h"
#include "Support/RCRef.h"
#include "Support/SymbolExport.h"

using namespace M;

//===----------------------------------------------------------------------===//
// AsyncValue
//===----------------------------------------------------------------------===//

// Reference counting for a bare `AsyncValue *` handle, backing the limited Mojo
// `AnyAsyncValueRef` wrapper. A null handle is the empty/non-tracked reference,
// so both are no-ops on null.
MODULAR_EXPORT void AsyncRT_AsyncValue_retain(AsyncRT::AsyncValue *value) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_AsyncValue_retain", TRACY_COLOR_GRAY);
  if (value)
    RCRef<AsyncRT::AsyncValue>::lowLevelAddRef(value);
}

MODULAR_EXPORT void AsyncRT_AsyncValue_release(AsyncRT::AsyncValue *value) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_AsyncValue_release", TRACY_COLOR_GRAY);
  if (value)
    RCRef<AsyncRT::AsyncValue>::lowLevelDropRef(value);
}

// Retains the `AsyncValue` behind an existing `AnyAsyncValueRef` storage handle
// (e.g. a cached buffer's memory handle) and hands the owning `AsyncValue*`
// (refcount +1) to a Mojo `AnyAsyncValueRef`. A null handle yields null.
MODULAR_EXPORT AsyncRT::AsyncValue *
AsyncRT_AsyncValue_retainHandle(AsyncRT::AnyAsyncValueRef *handle) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_AsyncValue_retainHandle", TRACY_COLOR_GRAY);
  if (!handle)
    return nullptr;
  return handle->copy().releasePointer();
}

// Retains the backing storage handle of a packed buffer/tensor async slot and
// hands the owning `AsyncValue*` (refcount +1) to a Mojo `AnyAsyncValueRef`.
// Used by unpack so an unpacked value (or a derivative) keeps its backing
// memory alive when it is re-packed as an output. Non-tracked storage yields
// null.
MODULAR_EXPORT AsyncRT::AsyncValue *
AsyncRT_AsyncValue_retainBufferStorage(AsyncRT::AnyAsyncValueRef *async) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_AsyncValue_retainBufferStorage",
                       TRACY_COLOR_GRAY);
  AsyncRT::AnyAsyncValueRef &value = *async;
  AsyncRT::AnyAsyncValueRef handle;
  if (value.isType<MLRT::TensorBufferRef>())
    handle = value.get<MLRT::TensorBufferRef>().getMemStorageHandle();
  else if (value.isType<MLRT::Tensor>())
    handle = value.get<MLRT::Tensor>().getBufferRef().getMemStorageHandle();
  return handle.releasePointer();
}

//===----------------------------------------------------------------------===//
// Device affinity
//===----------------------------------------------------------------------===//

// Compute the worker task ID for a GPU device. Called from the Mojo standard
// library via external_call (std.runtime.asyncrt.task_id_for_device). Obtains
// the thread pool size internally from the current AsyncRT cpuDevice.
MODULAR_EXPORT int32_t MLRT_TaskIdForDevice(int32_t deviceId) {
  auto *rt = AsyncRT::CPUDevice::getCurrentCPUDeviceOrNull();
  size_t numWorkers = rt->getWorkQueue()->getParallelismLevel();
  return MLRT::taskIdForDevice(deviceId, numWorkers);
}
