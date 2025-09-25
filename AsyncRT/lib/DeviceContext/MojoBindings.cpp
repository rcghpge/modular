//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains bindings for the C++ DeviceContext
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/DeviceContext/BinaryID.h"
#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "AsyncRT/lib/DeviceContext/CUDA/CUDADeviceContext.h"
#include "AsyncRT/lib/DeviceContext/CUDA/CUDADriver.h"
#include "AsyncRT/lib/DeviceContext/HIP/HIPDeviceContext.h"
#include "AsyncRT/lib/DeviceContext/HIP/HIPDriver.h"
#include "AsyncRT/lib/DeviceContext/Metal/MetalDeviceContext.h"
#include "Support/BinaryID.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringRef.h"

#include <chrono>

using namespace M;
using namespace M::AsyncRT;

//===----------------------------------------------------------------------===//
// DeviceContext
//===----------------------------------------------------------------------===//

const char *raiseError(const char *msg) { return strdup(msg); }

const char *checkSuccess(ErrorOrSuccess errOr) {
  if (errOr) {
    // Make a malloc'd copy of the string, so that we can safely free the
    // pointer from Mojo.
    return strdup(errOr.getError());
  }
  return nullptr;
}

template <typename T, typename R>
const char *checkRef(const T **result, ErrorOr<R> errOr) {
  if (errOr) {
    *result = nullptr;
    // Make a malloc'd copy of the string, so that we can safely free the
    // pointer from Mojo.
    return strdup(errOr.getError());
  }
  // Return the pointer, transferring the ownership to Mojo.
  *result = errOr.get().release();
  return nullptr;
}

template <typename T, typename V>
const char *checkVal(T *result, V errVal, ErrorOr<T> errOr) {
  if (errOr) {
    *result = errVal;
    // Make a malloc'd copy of the string, so that we can safely free the
    // pointer from Mojo.
    return strdup(errOr.getError());
  }
  *result = errOr.get();
  return nullptr;
}

MODULAR_EXPORT void AsyncRT_DeviceContext_strfree(const char *ptr) {
  free(const_cast<char *>(ptr));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_create(const DeviceContext **result, const char *kind,
                             int id) {
  return checkRef(result, DeviceContext::create(kind, id));
}

MODULAR_EXPORT void AsyncRT_DeviceContext_retain(const DeviceContext *ctx) {
  // Increment the low-level reference count.
  DeviceContextRef::lowLevelAddRef(ctx);
}

MODULAR_EXPORT void AsyncRT_DeviceContext_release(const DeviceContext *ctx) {
  // Give the event handlers opportunity to run, which can decrement the
  // reference count.
  [[maybe_unused]] ErrorOrSuccess ignored = ctx->checkPendingWork();
  // Decrement the low-level reference count.
  DeviceContextRef::lowLevelDropRef(ctx);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_deviceName(const DeviceContext *ctx) {
  return strdup(ctx->device().name().c_str());
}

MODULAR_EXPORT void AsyncRT_DeviceContext_deviceApi(llvm::StringRef *result,
                                                    const DeviceContext *ctx) {
  *result = ctx->device().api();
}

MODULAR_EXPORT void AsyncRT_DeviceContext_archName(llvm::StringRef *result,
                                                   const DeviceContext *ctx) {
  ErrorOr<std::string> archName = ctx->device().getArchName();
  if (archName.isError())
    *result = "";
  else
    *result = archName.get();
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContextScope_create(const DeviceContextScope **result,
                                  const DeviceContext *ctx) {
  DeviceContextScope *scope = new DeviceContextScope(*ctx);
  const char *errMsg = checkSuccess(scope->err());
  if (errMsg == nullptr) {
    *result = scope;
  }
  return errMsg;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_setAsCurrent(const DeviceContext *ctx) {
  ErrorOrSuccess err = ctx->setAsCurrent();
  return checkSuccess(std::move(err));
}

MODULAR_EXPORT void
AsyncRT_DeviceContextScope_release(const DeviceContextScope *scope) {
  delete scope;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_cuda_current_context(CUcontext *result) {
  ErrorOr<const CUDADriver &> errDriver = CUDADriver::get();
  if (errDriver.isError()) {
    return raiseError(errDriver.getError());
  }
  if ((*errDriver).CtxGetCurrent(result) != CUDADriver::SUCCESS) {
    return raiseError("cuCtxGetCurrent failed.");
  }
  return nullptr;
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_createBuffer_async(
    const DeviceBuffer **result, void **devicePtr, const DeviceContext *ctx,
    size_t len, size_t elemSize) {
  const char *errMsg =
      checkRef(result, ctx->enqueueCreateBuffer(len * elemSize));
  if (errMsg == nullptr) {
    *devicePtr = (*result)->ptr();
  }
  return errMsg;
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_createBuffer_sync(
    const DeviceBuffer **result, void **devicePtr, const DeviceContext *ctx,
    size_t len, size_t elemSize) {
  const char *errMsg = checkRef(result, ctx->createBuffer(len * elemSize));
  if (errMsg == nullptr) {
    *devicePtr = (*result)->ptr();
  }
  return errMsg;
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_createHostBuffer(
    const DeviceBuffer **result, void **devicePtr, const DeviceContext *ctx,
    size_t len, size_t elemSize) {
  const char *errMsg = checkRef(result, ctx->createHostBuffer(len * elemSize));
  if (errMsg == nullptr) {
    *devicePtr = (*result)->ptr();
  }
  return errMsg;
}

// TODO: Remove the V1 compatibility.
MODULAR_EXPORT void AsyncRT_DeviceContext_createBuffer_owning(
    const DeviceBuffer **result, const DeviceContext *ctx, void *devicePtr,
    size_t len, size_t elemSize, bool owning) {
  *result =
      ctx->createBufferOwning(devicePtr, len * elemSize, owning).release();
}
MODULAR_EXPORT const char *AsyncRT_DeviceBuffer_createSubBuffer(
    const DeviceBuffer **result, void **devicePtr, const DeviceBuffer *buf,
    size_t offset, size_t len, size_t elemSize) {
  const char *errMsg =
      checkRef(result, buf->createSubBuffer(offset * elemSize, len * elemSize));
  if (errMsg == nullptr) {
    *devicePtr = (*result)->ptr();
  }
  return errMsg;
}

MODULAR_EXPORT void AsyncRT_DeviceBuffer_retain(const DeviceBuffer *buffer) {
  // Increment the low-level reference count.
  DeviceBufferRef::lowLevelAddRef(buffer);
}

MODULAR_EXPORT void AsyncRT_DeviceBuffer_release(const DeviceBuffer *buffer) {
  // Decrement the low-level reference count.
  DeviceBufferRef::lowLevelDropRef(buffer);
}

MODULAR_EXPORT const DeviceContext *
AsyncRT_DeviceBuffer_context(const DeviceBuffer *buffer) {
  return &buffer->context();
}

MODULAR_EXPORT const char *
AsyncRT_DeviceBuffer_reassignOwnershipTo(const DeviceBuffer *buf,
                                         const DeviceContext *ctx) {
  return checkSuccess(buf->reassignOwnershipTo(*ctx));
}

// TODO: Remove the V1 compatibility.
MODULAR_EXPORT void
AsyncRT_DeviceBuffer_release_ptr(const DeviceBuffer *buffer) {
  buffer->releasePtr();
}

MODULAR_EXPORT int64_t
AsyncRT_DeviceBuffer_bytesize(const DeviceBuffer *buffer) {
  return buffer->byteSize();
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_stream(const DeviceStream **result,
                             const DeviceContext *ctx) {
  return checkRef(result, ctx->stream());
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_streamCreate(
    const DeviceStream **stream, const DeviceContext *ctx, unsigned int flags) {

  return checkRef(stream, ctx->streamCreate(flags));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_streamCreateWithPriority(const DeviceStream **stream,
                                               unsigned int flags, int priority,
                                               const DeviceContext *ctx) {
  return checkRef(stream, ctx->streamCreateWithPriority(flags, priority));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_streamPriorityRange(
    int *leastPriority, int *greatestPriority, const DeviceContext *ctx) {
  return checkSuccess(
      ctx->streamPriorityRange(leastPriority, greatestPriority));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceStream_synchronize(const DeviceStream *stream) {
  return checkSuccess(stream->synchronize());
}

MODULAR_EXPORT void AsyncRT_DeviceStream_retain(const DeviceStream *stream) {
  // Increment the low-level reference count.
  DeviceStreamRef::lowLevelAddRef(stream);
}

MODULAR_EXPORT void AsyncRT_DeviceStream_release(const DeviceStream *stream) {
  // Decrement the low-level reference count.
  DeviceStreamRef::lowLevelDropRef(stream);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceStream_waitForEvent(const DeviceStream *stream,
                                  const DeviceEvent *event) {
  return checkSuccess(stream->enqueueWaitForEvent(*event));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceStream_eventRecord(const DeviceStream *stream,
                                 const DeviceEvent *event) {
  return checkSuccess(stream->eventRecord(*event));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceEvent_synchronize(const DeviceEvent *event) {
  return checkSuccess(event->synchronize());
}

MODULAR_EXPORT void AsyncRT_DeviceEvent_retain(const DeviceEvent *event) {
  // Increment the low-level reference count.
  DeviceEventRef::lowLevelAddRef(event);
}

MODULAR_EXPORT void AsyncRT_DeviceEvent_release(const DeviceEvent *event) {
  // Decrement the low-level reference count.
  DeviceEventRef::lowLevelDropRef(event);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_synchronize(const DeviceContext *ctx) {
  return checkSuccess(ctx->synchronize());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_enqueue_wait_for_context(const DeviceContext *ctx,
                                               const DeviceContext *other) {
  return checkSuccess(ctx->enqueueWaitFor(*other));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_eventCreate(
    const DeviceEvent **result, const DeviceContext *ctx, unsigned int flags) {
  return checkRef(result, ctx->eventCreate(flags));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_waitForEvent(const DeviceContext *ctx,
                                   const DeviceEvent *event) {
  return checkSuccess(ctx->enqueueWaitFor(*event));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_HtoD_async(const DeviceContext *ctx,
                                 const DeviceBuffer *dst, const void *src) {
  assert((ctx == &dst->context()) && "ctx not matching dst");
  return checkSuccess(ctx->enqueueCopyToDevice(*dst, src));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_DtoH_async(const DeviceContext *ctx, void *dst,
                                 const DeviceBuffer *src) {
  assert((ctx == &src->context()) && "ctx not matching src");
  return checkSuccess(ctx->enqueueCopyFromDevice(dst, *src));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_HtoD_async_sized(const DeviceContext *ctx, void *dst,
                                       void *src, size_t size) {
  return checkSuccess(ctx->enqueueCopyToDeviceSized(dst, src, size));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_DtoH_async_sized(const DeviceContext *ctx, void *dst,
                                       void *src, size_t size) {
  return checkSuccess(ctx->enqueueCopyFromDeviceSized(dst, src, size));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_DtoD_async(const DeviceContext *ctx,
                                 const DeviceBuffer *dst,
                                 const DeviceBuffer *src) {
  return checkSuccess(ctx->enqueueCopyDeviceToDevice(*dst, *src));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_setMemory_async(const DeviceContext *ctx,
                                      const DeviceBuffer *dst, uint64_t val,
                                      size_t valSize) {
  return checkSuccess(ctx->enqueueSetMemory(*dst, val, valSize));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_loadFunction(
    const DeviceFunction **result, const DeviceContext *ctx,
    const char *moduleName, const char *functionName, const char *data,
    size_t dataLen, int32_t maxDynamicSharedBytes, const char *debugLevel,
    int32_t optimizationLevel) {
  DeviceFunction::SignatureBuilder sig;
  return checkRef(result, ctx->loadFunction(moduleName, functionName, sig,
                                            StringRef(data, dataLen),
                                            maxDynamicSharedBytes, debugLevel,
                                            optimizationLevel));
}

MODULAR_EXPORT void AsyncRT_DeviceFunction_retain(const DeviceFunction *func) {
  // Increment the low-level reference count.
  DeviceFunctionRef::lowLevelAddRef(func);
}

MODULAR_EXPORT void AsyncRT_DeviceFunction_release(const DeviceFunction *func) {
  // Decrement the low-level reference count.
  DeviceFunctionRef::lowLevelDropRef(func);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceFunction_copyToConstantMemory(const DeviceFunction *func,
                                            const char *name, size_t nameSize,
                                            const void *data, size_t dataSize) {
  assert(name != nullptr && "name is required");
  assert(nameSize > 0 && "nameSize is required");
  std::string nameStr(name, nameSize);
  return checkSuccess(
      func->copyToNamedConstant(nameStr.c_str(), data, dataSize));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_enqueueFunctionDirect(
    const DeviceContext *ctx, const DeviceFunction *func, uint32_t gridX,
    uint32_t gridY, uint32_t gridZ, uint32_t blockX, uint32_t blockY,
    uint32_t blockZ, uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs,
    void **args, const size_t *argSizes) {
  assert((ctx == &func->context()) && "ctx not matching func");
  return checkSuccess(ctx->enqueueFunctionExecDirect(
      *func, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes, attrs,
      numAttrs, args, argSizes));
}

MODULAR_EXPORT const char *AsyncRT_DeviceStream_enqueueFunctionDirect(
    const DeviceStream *stream, const DeviceFunction *func, uint32_t gridX,
    uint32_t gridY, uint32_t gridZ, uint32_t blockX, uint32_t blockY,
    uint32_t blockZ, uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs,
    void **args) {
  return checkSuccess(stream->enqueueFunctionExecDirect(
      *func, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
      reinterpret_cast<CUlaunchAttribute *>(attrs), numAttrs, args));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceFunction_getAttribute(int32_t *result, const DeviceFunction *func,
                                    int32_t attrCode) {
  return checkVal(result, 0, func->getAttribute(attrCode));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_startTimer(const DeviceTimer **result,
                                 const DeviceContext *ctx) {
  return checkRef(result, ctx->startTimer());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_stopTimer(int64_t *result, const DeviceContext *ctx,
                                const DeviceTimer *timer) {
  std::chrono::nanoseconds elapsedNanos;
  const char *errMsg = checkVal(&elapsedNanos, std::chrono::nanoseconds{0},
                                ctx->stopTimer(*timer));
  *result = elapsedNanos.count();
  return errMsg;
}

MODULAR_EXPORT void AsyncRT_DeviceTimer_retain(const DeviceTimer *timer) {
  // Increment the low-level reference count.
  DeviceTimerRef::lowLevelAddRef(timer);
}

MODULAR_EXPORT void AsyncRT_DeviceTimer_release(const DeviceTimer *timer) {
  // Decrement the low-level reference count.
  DeviceTimerRef::lowLevelDropRef(timer);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_getApiVersion(int *result, const DeviceContext *ctx) {
  return checkVal(result, 0, ctx->device().getApiVersion());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_getAttribute(int *result, const DeviceContext *ctx,
                                   int attr) {
  return checkVal(
      result, 0,
      ctx->device().getAttribute(static_cast<Device::Attribute>(attr)));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_isCompatible(const DeviceContext *ctx) {
  return checkSuccess(ctx->device().isCompatible());
}

MODULAR_EXPORT int64_t AsyncRT_DeviceContext_id(const DeviceContext *ctx) {
  return ctx->device().id();
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_computeCapability(int32_t *result,
                                        const DeviceContext *ctx) {
  int compute;
  const char *errMsg = checkVal(&compute, 0, ctx->device().computeCapability());
  *result = compute;
  return errMsg;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_getMemoryInfo(const DeviceContext *ctx, size_t *free,
                                    size_t *total) {
  Device::MemInfo memInfo;
  const char *errMsg =
      checkVal(&memInfo, Device::MemInfo(), ctx->getMemoryInfo());
  *free = memInfo.free;
  *total = memInfo.total;
  return errMsg;
}

MODULAR_EXPORT const char *AsyncRT_occupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, const DeviceFunction *func, int blockSize,
    size_t dynamicSharedMemSize) {
  auto ctx = &func->context();
  return checkSuccess(ctx->occupancyMaxActiveBlocksPerMultiprocessor(
      numBlocks, *func, blockSize, dynamicSharedMemSize));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_canAccess(bool *result, const DeviceContext *ctx,
                                const DeviceContext *peer) {
  return checkVal(result, false, ctx->device().canAccess(peer->device()));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_enablePeerAccess(const DeviceContext *ctx,
                                       const DeviceContext *peer) {
  return checkSuccess(ctx->device().enablePeerAccess(peer->device()));
}

// Enable peer access for all CUDA devices.
MODULAR_EXPORT const char *AsyncRT_DeviceContext_enableAllPeerAccess() {
  int numDevices = Device::numberOfDevices(Device::gpuAPI);
  if (numDevices <= 1)
    return nullptr;

  SmallVector<DeviceRef> devices;
  devices.reserve(numDevices);
  for (int id = 0; id < numDevices; ++id) {
    ErrorOr<DeviceRef> devOr = Device::create(Device::gpuAPI, id);
    if (devOr)
      return strdup(devOr.getError());
    devices.emplace_back(std::move(devOr.get()));
  }

  for (int i : llvm::seq(0, numDevices)) {
    for (int j : llvm::seq(i + 1, numDevices)) {
      // i -> j
      if (ErrorOrSuccess errOr = devices[i]->enablePeerAccess(*devices[j]))
        return strdup(errOr.getError());

      // j -> i
      if (ErrorOrSuccess errOr = devices[j]->enablePeerAccess(*devices[i]))
        return strdup(errOr.getError());
    }
  }

  return nullptr;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_supportsMulticast(bool *result,
                                        const DeviceContext *ctx) {
  return checkVal(result, false, ctx->device().supportsMulticast());
}

MODULAR_EXPORT int32_t AsyncRT_DeviceContext_numberOfDevices(const char *kind) {
  return Device::numberOfDevices(kind);
}

MODULAR_EXPORT const char *AsyncRT_DeviceMulticastBuffer_allocate(
    const DeviceMulticastBuffer **result, size_t ctxsLen,
    const DeviceContext **ctxs, size_t len, size_t elemSize) {
  std::vector<DeviceContextRef> contexts;
  for (size_t i = 0; i < ctxsLen; i++) {
    contexts.push_back(DeviceContextRef::copy(ctxs[i]));
  }
  return checkRef(result,
                  DeviceMulticastBuffer::allocate(contexts, len * elemSize));
}

MODULAR_EXPORT const char *AsyncRT_DeviceMulticastBuffer_unicastBufferFor(
    const DeviceBuffer **result, void **devicePtr,
    const DeviceMulticastBuffer *multiBuffer, const DeviceContext *ctx) {
  const char *errMsg = checkRef(result, multiBuffer->unicastBufferFor(*ctx));
  if (errMsg == nullptr) {
    *devicePtr = (*result)->ptr();
  }
  return errMsg;
}

MODULAR_EXPORT const char *AsyncRT_DeviceMulticastBuffer_multicastBufferFor(
    const DeviceBuffer **result, void **devicePtr,
    const DeviceMulticastBuffer *multiBuffer, const DeviceContext *ctx) {
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
AsyncRT_DeviceContext_cuda_context(CUcontext *result,
                                   const DeviceContext *ctx) {
  const CUDADeviceContext *verifiedCtx = CUDADeviceContext::verified_cast(ctx);
  if (verifiedCtx == nullptr) {
    return strdup("Not a CUDAContext");
  }
  *result = verifiedCtx->cudaContext();
  return nullptr;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceStream_cuda_stream(CUstream *result, const DeviceStream *stream) {
  const CUDADeviceStream *verifiedStream =
      CUDADeviceStream::verified_cast(stream);
  if (verifiedStream == nullptr) {
    return strdup("Not a CUDAStream");
  }
  *result = verifiedStream->cudaStream();
  return nullptr;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceFunction_cuda_module(CUmodule *result,
                                   const DeviceFunction *func) {
  // Check if the device context is CUDA
  if (func->context().device().api() != Device::cudaAPI) {
    return strdup("Not a CUDA DeviceFunction");
  }

  const CUDADeviceFunction &cudaFunc = CUDADeviceFunction::cast(*func);
  *result = cudaFunc.cudaModule();
  return nullptr;
}

MODULAR_EXPORT const char *AsyncRT_cuda_tensorMapEncodeTiled(
    void *tensorMap, int32_t tensorDataType, uint32_t tensorRank,
    const DeviceBuffer *globalAddress, const uint64_t *globalDim,
    const uint64_t *globalStrides, const uint32_t *boxDim,
    const uint32_t *elementStrides, int32_t interleave, int32_t swizzle,
    int32_t l2Promotion, int32_t oobFill) {
  // Extract the device context from the device buffer.
  const CUDADeviceContext *verifiedCtx =
      CUDADeviceContext::verified_cast(&globalAddress->context());
  if (verifiedCtx == nullptr) {
    return strdup("Not a CUDADeviceBuffer");
  }
  return checkSuccess(verifiedCtx->cudaTensorMapEncodeTiled(
      tensorMap, static_cast<CUtensorMapDataType>(tensorDataType), tensorRank,
      *globalAddress, globalDim, globalStrides, boxDim, elementStrides,
      static_cast<CUtensorMapInterleave>(interleave),
      static_cast<CUtensorMapSwizzle>(swizzle),
      static_cast<CUtensorMapL2promotion>(l2Promotion),
      static_cast<CUtensorMapFloatOOBfill>(oobFill)));
}

//
// Raw HIP accessors
//
MODULAR_EXPORT const char *
AsyncRT_DeviceContext_hip_device(hipDevice_t *result,
                                 const DeviceContext *ctx) {
  const HIPDeviceContext *verifiedCtx = HIPDeviceContext::verified_cast(ctx);
  if (verifiedCtx == nullptr) {
    return strdup("Not a HIPContext");
  }
  *result = verifiedCtx->hipDevice();
  return nullptr;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceStream_hip_stream(hipStream_t *result,
                                const DeviceStream *stream) {
  const HIPDeviceStream *verifiedStream =
      HIPDeviceStream::verified_cast(stream);
  if (verifiedStream == nullptr) {
    return strdup("Not a HIPStream");
  }
  *result = verifiedStream->hipStream();
  return nullptr;
}

//
// Raw Metal accessors
//
namespace MTL {
class Device;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_metal_device(MTL::Device **result,
                                   const DeviceContext *ctx) {
#ifdef MODULAR_MACOS
  const MetalDeviceContext *verifiedCtx =
      MetalDeviceContext::verified_cast(ctx);
  if (verifiedCtx == nullptr)
    return strdup("Not a MetalContext");
  *result = verifiedCtx->metalDevice();
  return nullptr;
#else
  return strdup("Not a MetalContext");
#endif
}

//
// BinaryID
//
MODULAR_CXX_EXPORT std::string M::AsyncRT::getMojoBindingsBinaryID() {
  // M::getBinaryID() returns the binary ID of the shared library that contains
  // it. For the purposes of MEF cache invalidation, we need to know when
  // there's been a change in these shared libraries.
  return M::getBinaryID();
}
