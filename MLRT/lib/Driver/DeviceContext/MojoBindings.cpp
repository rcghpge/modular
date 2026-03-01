//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains bindings for the C++ DeviceContext
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/BinaryID.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/CUDA/CUDADeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/CUDA/CUDADriver.h"
#include "MLRT/lib/Driver/DeviceContext/HIP/HIPDeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/HIP/HIPDriver.h"
#include "MLRT/lib/Driver/DeviceContext/Metal/MetalDeviceContext.h"
#include "Support/BinaryID.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringRef.h"

#include <chrono>

using namespace M;
using namespace M::Driver;

//===----------------------------------------------------------------------===//
// DeviceContext
//===----------------------------------------------------------------------===//

// Make a malloc'd copy of the string, so that we can safely free the
// pointer from Mojo.
static const char *raiseError(const char *msg) { return strdup(msg); }

static const char *checkSuccess(ErrorOrSuccess errOr) {
  if (errOr) {
    return raiseError(errOr.getError());
  }
  return nullptr;
}

template <typename T, typename R>
const char *checkRef(T **result, ErrorOr<R> errOr) {
  if (errOr) {
    *result = nullptr;
    return raiseError(errOr.getError());
  }
  // Return the pointer, transferring the ownership to Mojo.
  *result = errOr.get().release();
  return nullptr;
}

template <typename T, typename V>
const char *checkVal(T *result, V errVal, ErrorOr<T> errOr) {
  if (errOr) {
    *result = errVal;
    return raiseError(errOr.getError());
  }
  *result = errOr.get();
  return nullptr;
}

MODULAR_EXPORT void AsyncRT_DeviceContext_strfree(const char *ptr) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_release", TRACY_COLOR_GRAY);
  free(const_cast<char *>(ptr));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_create(DeviceContext **result, const char *kind, int id) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_release", TRACY_COLOR_GRAY);
  return checkRef(result, DeviceContext::create(kind, id));
}

MODULAR_EXPORT void AsyncRT_DeviceContext_retain(DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_retain", TRACY_COLOR_GRAY);
  // Increment the low-level reference count.
  DeviceContextRef::lowLevelAddRef(ctx);
}

MODULAR_EXPORT void AsyncRT_DeviceContext_release(DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_release", TRACY_COLOR_GRAY);
  // Give the event handlers opportunity to run, which can decrement the
  // reference count.
  [[maybe_unused]] ErrorOrSuccess ignored = ctx->pollPendingEvents();
  // Decrement the low-level reference count.
  DeviceContextRef::lowLevelDropRef(ctx);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_deviceName(DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_deviceName", TRACY_COLOR_GRAY);
  return raiseError(ctx->device().name().c_str());
}

MODULAR_EXPORT void AsyncRT_DeviceContext_deviceApi(llvm::StringRef *result,
                                                    DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_deviceApi", TRACY_COLOR_GRAY);
  *result = ctx->device().api();
}

MODULAR_EXPORT void AsyncRT_DeviceContext_archName(llvm::StringRef *result,
                                                   DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_archName", TRACY_COLOR_GRAY);
  ErrorOr<const char *> archName = ctx->device().getArchName();
  if (archName.isError())
    *result = "";
  else
    *result = archName.get();
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContextScope_create(const DeviceContextScope **result,
                                  DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContextScope_create", TRACY_COLOR_GRAY);
  DeviceContextScope *scope = new DeviceContextScope(*ctx);
  const char *errMsg = checkSuccess(scope->err());
  if (errMsg == nullptr) {
    *result = scope;
  }
  return errMsg;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_setAsCurrent(DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_setAsCurrent", TRACY_COLOR_GRAY);
  ErrorOrSuccess err = ctx->setAsCurrent();
  return checkSuccess(std::move(err));
}

MODULAR_EXPORT void
AsyncRT_DeviceContextScope_release(const DeviceContextScope *scope) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContextScope_release", TRACY_COLOR_GRAY);
  delete scope;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_cuda_current_context(CUcontext *result) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_cuda_current_context",
                       TRACY_COLOR_GRAY);
  ErrorOr<const CUDADriver &> errDriver = CUDADriver::get();
  if (errDriver.isError()) {
    return raiseError(errDriver.getError());
  }
  if ((*errDriver).cuCtxGetCurrent(result) != CUDADriver::SUCCESS) {
    return raiseError("cuCtxGetCurrent failed.");
  }
  return nullptr;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_createBuffer_async(DeviceBuffer **result,
                                         void **devicePtr, DeviceContext *ctx,
                                         size_t len, size_t elemSize) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_createBuffer_async",
                       TRACY_COLOR_GRAY);
  const char *errMsg =
      checkRef(result, ctx->enqueueCreateBuffer(len * elemSize));
  if (errMsg == nullptr) {
    *devicePtr = (*result)->ptr();
  }
  return errMsg;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_createHostBuffer(DeviceBuffer **result, void **devicePtr,
                                       DeviceContext *ctx, size_t len,
                                       size_t elemSize) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_createHostBuffer",
                       TRACY_COLOR_GRAY);
  const char *errMsg = checkRef(result, ctx->createHostBuffer(len * elemSize));
  if (errMsg == nullptr) {
    *devicePtr = (*result)->ptr();
  }
  return errMsg;
}

// TODO: Remove the V1 compatibility.
MODULAR_EXPORT void AsyncRT_DeviceContext_createBuffer_owning(
    DeviceBuffer **result, DeviceContext *ctx, void *devicePtr, size_t len,
    size_t elemSize, bool owning) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_createBuffer_owning",
                       TRACY_COLOR_GRAY);
  *result =
      ctx->createBufferOwning(devicePtr, len * elemSize, owning).release();
}
MODULAR_EXPORT const char *
AsyncRT_DeviceBuffer_createSubBuffer(DeviceBuffer **result, void **devicePtr,
                                     DeviceBuffer *buf, size_t offset,
                                     size_t len, size_t elemSize) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceBuffer_createSubBuffer",
                       TRACY_COLOR_GRAY);
  const char *errMsg =
      checkRef(result, buf->createSubBuffer(offset * elemSize, len * elemSize));
  if (errMsg == nullptr) {
    *devicePtr = (*result)->ptr();
  }
  return errMsg;
}

MODULAR_EXPORT void AsyncRT_DeviceBuffer_retain(DeviceBuffer *buffer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceBuffer_retain", TRACY_COLOR_GRAY);
  // Increment the low-level reference count.
  DeviceBufferRef::lowLevelAddRef(buffer);
}

MODULAR_EXPORT void AsyncRT_DeviceBuffer_release(DeviceBuffer *buffer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceBuffer_release", TRACY_COLOR_GRAY);
  // Decrement the low-level reference count.
  DeviceBufferRef::lowLevelDropRef(buffer);
}

MODULAR_EXPORT const DeviceContext *
AsyncRT_DeviceBuffer_context(DeviceBuffer *buffer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceBuffer_context", TRACY_COLOR_GRAY);
  return &buffer->context();
}

MODULAR_EXPORT const char *
AsyncRT_DeviceBuffer_reassignOwnershipTo(DeviceBuffer *buf,
                                         DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceBuffer_reassignOwnershipTo",
                       TRACY_COLOR_GRAY);
  return checkSuccess(buf->reassignOwnershipTo(*ctx));
}

// TODO: Remove the V1 compatibility.
MODULAR_EXPORT void AsyncRT_DeviceBuffer_release_ptr(DeviceBuffer *buffer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceBuffer_release_ptr", TRACY_COLOR_GRAY);
  buffer->releasePtr();
}

MODULAR_EXPORT int64_t AsyncRT_DeviceBuffer_bytesize(DeviceBuffer *buffer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceBuffer_bytesize", TRACY_COLOR_GRAY);
  return buffer->byteSize();
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_stream(DeviceStream **result,
                                                        DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_stream", TRACY_COLOR_GRAY);
  return checkRef(result, ctx->stream());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_createStream(DeviceStream **stream, int priority,
                                   DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_createStream", TRACY_COLOR_GRAY);
  return checkRef(stream, ctx->createStream(priority));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_streamPriorityRange(
    int *leastPriority, int *greatestPriority, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_streamPriorityRange",
                       TRACY_COLOR_GRAY);
  return checkSuccess(
      ctx->streamPriorityRange(leastPriority, greatestPriority));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceStream_synchronize(DeviceStream *stream) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceStream_synchronize", TRACY_COLOR_GRAY);
  return checkSuccess(stream->synchronize());
}

MODULAR_EXPORT void AsyncRT_DeviceStream_retain(DeviceStream *stream) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceStream_retain", TRACY_COLOR_GRAY);
  // Increment the low-level reference count.
  DeviceStreamRef::lowLevelAddRef(stream);
}

MODULAR_EXPORT void AsyncRT_DeviceStream_release(DeviceStream *stream) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceStream_release", TRACY_COLOR_GRAY);
  // Decrement the low-level reference count.
  DeviceStreamRef::lowLevelDropRef(stream);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceStream_waitForEvent(DeviceStream *stream, DeviceEvent *event) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceStream_waitForEvent", TRACY_COLOR_GRAY);
  return checkSuccess(stream->enqueueWaitForEvent(*event));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceStream_eventRecord(DeviceStream *stream, DeviceEvent *event) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceStream_eventRecord", TRACY_COLOR_GRAY);
  return checkSuccess(stream->eventRecord(*event));
}

MODULAR_EXPORT const char *AsyncRT_DeviceEvent_synchronize(DeviceEvent *event) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceEvent_synchronize", TRACY_COLOR_GRAY);
  return checkSuccess(event->synchronize());
}

MODULAR_EXPORT void AsyncRT_DeviceEvent_retain(DeviceEvent *event) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceEvent_retain", TRACY_COLOR_GRAY);
  // Increment the low-level reference count.
  DeviceEventRef::lowLevelAddRef(event);
}

MODULAR_EXPORT void AsyncRT_DeviceEvent_release(DeviceEvent *event) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceEvent_release", TRACY_COLOR_GRAY);
  // Decrement the low-level reference count.
  DeviceEventRef::lowLevelDropRef(event);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_synchronize(DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_synchronize", TRACY_COLOR_GRAY);
  return checkSuccess(ctx->synchronize());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_enqueue_wait_for_context(DeviceContext *ctx,
                                               DeviceContext *other) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_enqueue_wait_for_context",
                       TRACY_COLOR_GRAY);
  return checkSuccess(ctx->enqueueWaitFor(*other));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_eventCreate(DeviceEvent **result, DeviceContext *ctx,
                                  unsigned int flags) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_eventCreate", TRACY_COLOR_GRAY);
  return checkRef(result, ctx->eventCreate(flags));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_waitForEvent(DeviceContext *ctx, DeviceEvent *event) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_waitForEvent", TRACY_COLOR_GRAY);
  return checkSuccess(ctx->enqueueWaitFor(*event));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_HtoD_async(DeviceContext *ctx,
                                                            DeviceBuffer *dst,
                                                            const void *src) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_HtoD_async", TRACY_COLOR_GRAY);
  assert((ctx == &dst->context()) && "ctx not matching dst");
  return checkSuccess(ctx->enqueueCopyToDevice(*dst, src));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_DtoH_async(DeviceContext *ctx,
                                                            void *dst,
                                                            DeviceBuffer *src) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_DtoH_async", TRACY_COLOR_GRAY);
  assert((ctx == &src->context()) && "ctx not matching src");
  return checkSuccess(ctx->enqueueCopyFromDevice(dst, *src));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_HtoD_async_sized(DeviceContext *ctx, void *dst, void *src,
                                       size_t size) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_HtoD_async_sized",
                       TRACY_COLOR_GRAY);
  return checkSuccess(ctx->enqueueCopyToDevice(dst, src, size));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_DtoH_async_sized(DeviceContext *ctx, void *dst, void *src,
                                       size_t size) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_DtoH_async_sized",
                       TRACY_COLOR_GRAY);
  return checkSuccess(ctx->enqueueCopyFromDevice(dst, src, size));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_DtoD_async(DeviceContext *ctx,
                                                            DeviceBuffer *dst,
                                                            DeviceBuffer *src) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_DtoD_async", TRACY_COLOR_GRAY);
  return checkSuccess(ctx->enqueueCopyDeviceToDevice(*dst, *src));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_setMemory_async(DeviceContext *ctx, DeviceBuffer *dst,
                                      uint64_t val, size_t valSize) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_setMemory_async",
                       TRACY_COLOR_GRAY);
  return checkSuccess(ctx->enqueueSetMemory(*dst, val, valSize));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_loadFunction(
    DeviceFunction **result, DeviceContext *ctx, const char *moduleName,
    const char *functionName, const char *data, size_t dataLen,
    int32_t maxDynamicSharedBytes, const char *debugLevel,
    int32_t optimizationLevel) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_loadFunction", TRACY_COLOR_GRAY);
  return checkRef(result, ctx->loadFunction(moduleName, functionName,
                                            StringRef(data, dataLen),
                                            maxDynamicSharedBytes, debugLevel,
                                            optimizationLevel));
}

MODULAR_EXPORT void AsyncRT_DeviceFunction_retain(DeviceFunction *func) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceFunction_retain", TRACY_COLOR_GRAY);
  // Increment the low-level reference count.
  DeviceFunctionRef::lowLevelAddRef(func);
}

MODULAR_EXPORT void AsyncRT_DeviceFunction_release(DeviceFunction *func) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceFunction_release", TRACY_COLOR_GRAY);
  // Decrement the low-level reference count.
  DeviceFunctionRef::lowLevelDropRef(func);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceFunction_copyToConstantMemory(DeviceFunction *func,
                                            const char *name, size_t nameSize,
                                            const void *data, size_t dataSize) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceFunction_copyToConstantMemory",
                       TRACY_COLOR_GRAY);
  assert(name != nullptr && "name is required");
  assert(nameSize > 0 && "nameSize is required");
  std::string nameStr(name, nameSize);
  return checkSuccess(
      func->copyToNamedConstant(nameStr.c_str(), data, dataSize));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_enqueueFunctionDirect(
    DeviceContext *ctx, DeviceFunction *func, uint32_t gridX, uint32_t gridY,
    uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
    uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args,
    const size_t *argSizes) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_enqueueFunctionDirect",
                       TRACY_COLOR_GRAY);
  assert((ctx == &func->context()) && "ctx not matching func");
  return checkSuccess(ctx->enqueueFunctionExecDirect(
      *func, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes, attrs,
      numAttrs, args, argSizes));
}

MODULAR_EXPORT const char *AsyncRT_DeviceStream_enqueueFunctionDirect(
    DeviceStream *stream, DeviceFunction *func, uint32_t gridX, uint32_t gridY,
    uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
    uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceStream_enqueueFunctionDirect",
                       TRACY_COLOR_GRAY);
  return checkSuccess(stream->enqueueFunctionExecDirect(
      *func, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
      reinterpret_cast<CUlaunchAttribute *>(attrs), numAttrs, args));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceFunction_getAttribute(int32_t *result, DeviceFunction *func,
                                    int32_t attrCode) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceFunction_getAttribute", TRACY_COLOR_GRAY);
  return checkVal(result, 0, func->getAttribute(attrCode));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_startTimer(DeviceTimer **result, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_startTimer", TRACY_COLOR_GRAY);
  return checkRef(result, ctx->startTimer());
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_stopTimer(int64_t *result,
                                                           DeviceContext *ctx,
                                                           DeviceTimer *timer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_stopTimer", TRACY_COLOR_GRAY);
  std::chrono::nanoseconds elapsedNanos;
  const char *errMsg = checkVal(&elapsedNanos, std::chrono::nanoseconds{0},
                                ctx->stopTimer(*timer));
  *result = elapsedNanos.count();
  return errMsg;
}

MODULAR_EXPORT void AsyncRT_DeviceTimer_retain(DeviceTimer *timer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceTimer_retain", TRACY_COLOR_GRAY);
  // Increment the low-level reference count.
  DeviceTimerRef::lowLevelAddRef(timer);
}

MODULAR_EXPORT void AsyncRT_DeviceTimer_release(DeviceTimer *timer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceTimer_release", TRACY_COLOR_GRAY);
  // Decrement the low-level reference count.
  DeviceTimerRef::lowLevelDropRef(timer);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_getApiVersion(int *result, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_getApiVersion", TRACY_COLOR_GRAY);
  return checkVal(result, 0, ctx->device().getApiVersion());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_getAttribute(int *result, DeviceContext *ctx, int attr) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_getAttribute", TRACY_COLOR_GRAY);
  return checkVal(
      result, 0,
      ctx->device().getAttribute(static_cast<Device::Attribute>(attr)));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_isCompatible(DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_isCompatible", TRACY_COLOR_GRAY);
  return checkSuccess(ctx->device().isCompatible());
}

MODULAR_EXPORT int64_t AsyncRT_DeviceContext_id(DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_id", TRACY_COLOR_GRAY);
  return ctx->device().id();
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_computeCapability(int32_t *result, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_computeCapability",
                       TRACY_COLOR_GRAY);
  int compute;
  const char *errMsg = checkVal(&compute, 0, ctx->device().computeCapability());
  *result = compute;
  return errMsg;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_getMemoryInfo(DeviceContext *ctx, size_t *free,
                                    size_t *total) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_getMemoryInfo", TRACY_COLOR_GRAY);
  Device::MemInfo memInfo;
  const char *errMsg =
      checkVal(&memInfo, Device::MemInfo(), ctx->getMemoryInfo());
  *free = memInfo.free;
  *total = memInfo.total;
  return errMsg;
}

MODULAR_EXPORT const char *AsyncRT_occupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, DeviceFunction *func, int blockSize,
    size_t dynamicSharedMemSize) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_occupancyMaxActiveBlocksPerMultiprocessor",
                       TRACY_COLOR_GRAY);
  auto ctx = &func->context();
  return checkSuccess(ctx->occupancyMaxActiveBlocksPerMultiprocessor(
      numBlocks, *func, blockSize, dynamicSharedMemSize));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_canAccess(bool *result, DeviceContext *ctx,
                                DeviceContext *peer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_canAccess", TRACY_COLOR_GRAY);
  return checkVal(result, false, ctx->device().canAccess(peer->device()));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_enablePeerAccess(DeviceContext *ctx,
                                       DeviceContext *peer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_enablePeerAccess",
                       TRACY_COLOR_GRAY);
  return checkSuccess(ctx->device().enablePeerAccess(peer->device()));
}

// Enable peer access for all CUDA devices.
MODULAR_EXPORT const char *AsyncRT_DeviceContext_enableAllPeerAccess() {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_enableAllPeerAccess",
                       TRACY_COLOR_GRAY);
  return checkSuccess(Device::enableAllPeerAccess());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_allPeerAccessEnabled(bool *result) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_allPeerAccessEnabled",
                       TRACY_COLOR_GRAY);
  return checkVal(result, false, Device::allPeerAccessEnabled());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_supportsMulticast(bool *result, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_supportsMulticast",
                       TRACY_COLOR_GRAY);
  return checkVal(result, false, ctx->device().supportsMulticast());
}

MODULAR_EXPORT int32_t AsyncRT_DeviceContext_numberOfDevices(const char *kind) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_numberOfDevices",
                       TRACY_COLOR_GRAY);
  return Device::numberOfDevices(kind);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceMulticastBuffer_allocate(DeviceMulticastBuffer **result,
                                       size_t ctxsLen, DeviceContext **ctxs,
                                       size_t len, size_t elemSize) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceMulticastBuffer_allocate",
                       TRACY_COLOR_GRAY);
  std::vector<DeviceContextRef> contexts;
  for (size_t i = 0; i < ctxsLen; ++i) {
    contexts.push_back(DeviceContextRef::copy(ctxs[i]));
  }
  return checkRef(result,
                  DeviceMulticastBuffer::allocate(contexts, len * elemSize));
}

MODULAR_EXPORT const char *AsyncRT_DeviceMulticastBuffer_unicastBufferFor(
    DeviceBuffer **result, void **devicePtr,
    const DeviceMulticastBuffer *multiBuffer, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceMulticastBuffer_unicastBufferFor",
                       TRACY_COLOR_GRAY);
  const char *errMsg = checkRef(result, multiBuffer->unicastBufferFor(*ctx));
  if (errMsg == nullptr) {
    *devicePtr = (*result)->ptr();
  }
  return errMsg;
}

MODULAR_EXPORT const char *AsyncRT_DeviceMulticastBuffer_multicastBufferFor(
    DeviceBuffer **result, void **devicePtr,
    const DeviceMulticastBuffer *multiBuffer, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceMulticastBuffer_multicastBufferFor",
                       TRACY_COLOR_GRAY);
  const char *errMsg = checkRef(result, multiBuffer->multicastBuffer(*ctx));
  if (errMsg == nullptr) {
    *devicePtr = (*result)->ptr();
  }
  return errMsg;
}

//
// Raw CUDA accessors
//
MODULAR_EXPORT const char *
AsyncRT_DeviceContext_cuda_context(CUcontext *result, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_cuda_context", TRACY_COLOR_GRAY);
  const CUDADeviceContext *verifiedCtx = CUDADeviceContext::verified_cast(ctx);
  if (verifiedCtx == nullptr) {
    return raiseError("Not a CUDAContext");
  }
  *result = verifiedCtx->cudaContext();
  return nullptr;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceStream_cuda_stream(CUstream *result, DeviceStream *stream) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceStream_cuda_stream", TRACY_COLOR_GRAY);
  const CUDADeviceStream *verifiedStream =
      CUDADeviceStream::verified_cast(stream);
  if (verifiedStream == nullptr) {
    return raiseError("Not a CUDAStream");
  }
  *result = verifiedStream->cudaStream();
  return nullptr;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceFunction_cuda_module(CUmodule *result, DeviceFunction *func) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceFunction_cuda_module", TRACY_COLOR_GRAY);
  // Check if the device context is CUDA
  if (func->context().device().api() != Device::cudaAPI) {
    return raiseError("Not a CUDA DeviceFunction");
  }

  const CUDADeviceFunction &cudaFunc = CUDADeviceFunction::cast(*func);
  *result = cudaFunc.cudaModule();
  return nullptr;
}

MODULAR_EXPORT const char *AsyncRT_cuda_tensorMapEncodeTiled(
    void *tensorMap, int32_t tensorDataType, uint32_t tensorRank,
    DeviceBuffer *globalAddress, const uint64_t *globalDim,
    const uint64_t *globalStrides, const uint32_t *boxDim,
    const uint32_t *elementStrides, int32_t interleave, int32_t swizzle,
    int32_t l2Promotion, int32_t oobFill) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_cuda_tensorMapEncodeTiled", TRACY_COLOR_GRAY);
  // Extract the device context from the device buffer.
  if (CUDADeviceContext::verified_cast(&globalAddress->context()) == nullptr) {
    return raiseError("Not a CUDADeviceBuffer");
  }
  CUDADeviceContext &ctx = CUDADeviceContext::cast(globalAddress->context());
  return checkSuccess(ctx.cudaTensorMapEncodeTiled(
      tensorMap, static_cast<CUtensorMapDataType>(tensorDataType), tensorRank,
      *globalAddress, globalDim, globalStrides, boxDim, elementStrides,
      static_cast<CUtensorMapInterleave>(interleave),
      static_cast<CUtensorMapSwizzle>(swizzle),
      static_cast<CUtensorMapL2promotion>(l2Promotion),
      static_cast<CUtensorMapFloatOOBfill>(oobFill)));
}

MODULAR_EXPORT const char *AsyncRT_cuda_tensorMapEncodeIm2col(
    void *tensorMap, int32_t tensorDataType, uint32_t tensorRank,
    DeviceBuffer *globalAddress, const uint64_t *globalDim,
    const uint64_t *globalStrides, const int32_t *pixelBoxLowerCorner,
    const int32_t *pixelBoxUpperCorner, uint32_t channelsPerPixel,
    uint32_t pixelsPerColumn, const uint32_t *elementStrides,
    int32_t interleave, int32_t swizzle, int32_t l2Promotion, int32_t oobFill) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_cuda_tensorMapEncodeIm2col", TRACY_COLOR_GRAY);
  // Extract the device context from the device buffer.
  if (CUDADeviceContext::verified_cast(&globalAddress->context()) == nullptr) {
    return raiseError("Not a CUDADeviceBuffer");
  }
  CUDADeviceContext &ctx = CUDADeviceContext::cast(globalAddress->context());
  return checkSuccess(ctx.cudaTensorMapEncodeIm2col(
      tensorMap, static_cast<CUtensorMapDataType>(tensorDataType), tensorRank,
      *globalAddress, globalDim, globalStrides, pixelBoxLowerCorner,
      pixelBoxUpperCorner, channelsPerPixel, pixelsPerColumn, elementStrides,
      static_cast<CUtensorMapInterleave>(interleave),
      static_cast<CUtensorMapSwizzle>(swizzle),
      static_cast<CUtensorMapL2promotion>(l2Promotion),
      static_cast<CUtensorMapFloatOOBfill>(oobFill)));
}

//
// Raw HIP accessors
//
MODULAR_EXPORT const char *
AsyncRT_DeviceContext_hip_device(hipDevice_t *result, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_hip_device", TRACY_COLOR_GRAY);
  const HIPDeviceContext *verifiedCtx = HIPDeviceContext::verified_cast(ctx);
  if (verifiedCtx == nullptr) {
    return raiseError("Not a HIPContext");
  }
  *result = verifiedCtx->hipDevice();
  return nullptr;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceStream_hip_stream(hipStream_t *result, DeviceStream *stream) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceStream_hip_stream", TRACY_COLOR_GRAY);
  const HIPDeviceStream *verifiedStream =
      HIPDeviceStream::verified_cast(stream);
  if (verifiedStream == nullptr) {
    return raiseError("Not a HIPStream");
  }
  *result = verifiedStream->hipStream();
  return nullptr;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceFunction_hip_module(hipModule_t *result, DeviceFunction *func) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceFunction_hip_module", TRACY_COLOR_GRAY);
  // Check if the device context is HIP
  if (func->context().device().api() != Device::hipAPI) {
    return raiseError("Not a HIP DeviceFunction");
  }

  const HIPDeviceFunction &hipFunc = HIPDeviceFunction::cast(*func);
  *result = hipFunc.hipModule();
  return nullptr;
}

//
// Raw Metal accessors
//
namespace MTL {
class Device;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_metal_device(MTL::Device **result, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_metal_device", TRACY_COLOR_GRAY);
#if MODULAR_MACOS
  const MetalDeviceContext *verifiedCtx =
      MetalDeviceContext::verified_cast(ctx);
  if (verifiedCtx == nullptr)
    return raiseError("Not a MetalContext");
  *result = verifiedCtx->metalDevice();
  return nullptr;
#else
  return raiseError("Not a MetalContext");
#endif
}

//
// BinaryID
//
MODULAR_CXX_EXPORT std::string M::Driver::getMojoBindingsBinaryID() {
  // M::getBinaryID() returns the binary ID of the shared library that contains
  // it. For the purposes of MEF cache invalidation, we need to know when
  // there's been a change in these shared libraries.
  return M::getBinaryID();
}
