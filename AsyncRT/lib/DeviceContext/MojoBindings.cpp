//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains bindings for the C++ DeviceContext
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "AsyncRT/lib/DeviceContext/CUDA/CUDADeviceContext.h"
#include "AsyncRT/lib/DeviceContext/CUDA/CUDADriver.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/StringRef.h"

#include <chrono>

using namespace M;
using namespace M::AsyncRT;

//===----------------------------------------------------------------------===//
// DeviceContext
//===----------------------------------------------------------------------===//

const char *check_success(ErrorOrSuccess err_or) {
  if (err_or) {
    // Make a malloc'd copy of the string, so that we can safely free the
    // pointer from Mojo.
    return strdup(err_or.getError());
  }
  return nullptr;
}

template <typename T, typename R>
const char *check_ref(const T **result, ErrorOr<R> err_or) {
  if (err_or) {
    *result = nullptr;
    // Make a malloc'd copy of the string, so that we can safely free the
    // pointer from Mojo.
    return strdup(err_or.getError());
  }
  // Return the pointer, transfering the ownership to Mojo.
  *result = err_or.get().release();
  return nullptr;
}

template <typename T, typename V>
const char *check_val(T *result, V err_val, ErrorOr<T> err_or) {
  if (err_or) {
    *result = err_val;
    // Make a malloc'd copy of the string, so that we can safely free the
    // pointer from Mojo.
    return strdup(err_or.getError());
  }
  *result = err_or.get();
  return nullptr;
}

MODULAR_EXPORT void AsyncRT_DeviceContext_strfree(const char *ptr) {
  free(const_cast<char *>(ptr));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_create(const DeviceContext **result, const char *kind,
                             int id, size_t buffer_cache_size) {
  return check_ref(result, DeviceContext::create(kind, id, buffer_cache_size));
}

MODULAR_EXPORT void AsyncRT_DeviceContext_retain(const DeviceContext *ctx) {
  // Increment the low-level reference count.
  DeviceContextRef::lowLevelAddRef(ctx);
}

MODULAR_EXPORT void AsyncRT_DeviceContext_release(const DeviceContext *ctx) {
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

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_mallocHost(void **result, const DeviceContext *ctx,
                                 size_t size) {
  return check_val(result, nullptr, ctx->malloc_host(size));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_freeHost(const DeviceContext *ctx, void *ptr) {
  return check_success(ctx->free_host(ptr));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_createBuffer_async(
    const DeviceBuffer **result, void **device_ptr, const DeviceContext *ctx,
    size_t len, size_t elem_size) {
  const char *err_msg = check_ref(
      result, ctx->createBuffer(len * elem_size, DeviceContext::ASYNC));
  if (err_msg == nullptr) {
    *device_ptr = (*result)->ptr();
  }
  return err_msg;
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_createBuffer_sync(
    const DeviceBuffer **result, void **device_ptr, const DeviceContext *ctx,
    size_t len, size_t elem_size) {
  const char *err_msg = check_ref(
      result, ctx->createBuffer(len * elem_size, DeviceContext::SYNC));
  if (err_msg == nullptr) {
    *device_ptr = (*result)->ptr();
  }
  return err_msg;
}

// TODO: Remove the V1 compatibility.
MODULAR_EXPORT void AsyncRT_DeviceContext_createBuffer_owning(
    const DeviceBuffer **result, const DeviceContext *ctx, void *device_ptr,
    size_t len, size_t elem_size, bool owning) {
  *result =
      ctx->createBufferOwning(device_ptr, len * elem_size, owning).release();
}
MODULAR_EXPORT const char *AsyncRT_DeviceBuffer_createSubBuffer(
    const DeviceBuffer **result, void **device_ptr, const DeviceBuffer *buf,
    size_t offset, size_t len, size_t elem_size) {
  const char *err_msg = check_ref(
      result, buf->createSubBuffer(offset * elem_size, len * elem_size));
  if (err_msg == nullptr) {
    *device_ptr = (*result)->ptr();
  }
  return err_msg;
}

MODULAR_EXPORT void AsyncRT_DeviceBuffer_retain(const DeviceBuffer *buffer) {
  // Increment the low-level reference count.
  DeviceBufferRef::lowLevelAddRef(buffer);
}

MODULAR_EXPORT void AsyncRT_DeviceBuffer_release(const DeviceBuffer *buffer) {
  // Decrement the low-level reference count.
  DeviceBufferRef::lowLevelDropRef(buffer);
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
  return check_ref(result, ctx->stream());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceStream_synchronize(const DeviceStream *stream) {
  return check_success(stream->synchronize());
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
AsyncRT_DeviceContext_synchronize(const DeviceContext *ctx) {
  return check_success(ctx->synchronize());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_HtoD_async(const DeviceContext *ctx,
                                 const DeviceBuffer *dst, const void *src) {
  assert((ctx == &dst->context()) && "ctx not matching dst");
  return check_success(ctx->copyToDevice(*dst, src, DeviceContext::ASYNC));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_DtoH_async(const DeviceContext *ctx, void *dst,
                                 const DeviceBuffer *src) {
  assert((ctx == &src->context()) && "ctx not matching src");
  return check_success(ctx->copyFromDevice(dst, *src, DeviceContext::ASYNC));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_DtoD_async(const DeviceContext *ctx,
                                 const DeviceBuffer *dst,
                                 const DeviceBuffer *src) {
  return check_success(
      ctx->copyDeviceToDevice(*dst, *src, DeviceContext::ASYNC));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_HtoD_sync(const DeviceContext *ctx,
                                const DeviceBuffer *dst, const void *src) {
  assert((ctx == &dst->context()) && "ctx not matching dst");
  return check_success(ctx->copyToDevice(*dst, src, DeviceContext::SYNC));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_DtoH_sync(const DeviceContext *ctx, void *dst,
                                const DeviceBuffer *src) {
  assert((ctx == &src->context()) && "ctx not matching src");
  return check_success(ctx->copyFromDevice(dst, *src, DeviceContext::SYNC));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_DtoD_sync(const DeviceContext *ctx,
                                const DeviceBuffer *dst,
                                const DeviceBuffer *src) {
  return check_success(
      ctx->copyDeviceToDevice(*dst, *src, DeviceContext::SYNC));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_setMemory_async(const DeviceContext *ctx,
                                      const DeviceBuffer *dst, uint32_t val,
                                      size_t val_size) {
  return check_success(
      ctx->setMemory(*dst, val, val_size, DeviceContext::ASYNC));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_setMemory_sync(const DeviceContext *ctx,
                                     const DeviceBuffer *dst, uint32_t val,
                                     size_t val_size) {
  return check_success(
      ctx->setMemory(*dst, val, val_size, DeviceContext::SYNC));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_loadFunction(
    const DeviceFunction **result, const DeviceContext *ctx,
    const char *module_name, const char *function_name, const void *data,
    int32_t max_registers, int32_t threads_per_block, int32_t cache_mode,
    int32_t cache_config, int32_t max_dynamic_shared_bytes,
    const char *debug_level, int32_t optimization_level) {
  DeviceFunction::SignatureBuilder sig;
  return check_ref(
      result, ctx->loadFunction(module_name, function_name, sig, data,
                                max_registers, threads_per_block, cache_mode,
                                cache_config, max_dynamic_shared_bytes,
                                debug_level, optimization_level));
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
                                            const char *name, const void *data,
                                            size_t byteSize) {
  return check_success(func->copyToNamedConstant(name, data, byteSize));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_enqueueFunctionDirect(
    const DeviceContext *ctx, const DeviceFunction *func, uint32_t gridX,
    uint32_t gridY, uint32_t gridZ, uint32_t blockX, uint32_t blockY,
    uint32_t blockZ, uint32_t sharedMemBytes, void *attrs, uint32_t num_attrs,
    void **args) {
  assert((ctx == &func->context()) && "ctx not matching func");
  return check_success(ctx->enqueueFunctionExecDirect(
      *func, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes, attrs,
      num_attrs, args));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceFunction_getAttribute(int32_t *result, const DeviceFunction *func,
                                    int32_t attr_code) {
  return check_val(result, 0, func->getAttribute(attr_code));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_startTimer(const DeviceTimer **result,
                                 const DeviceContext *ctx) {
  return check_ref(result, ctx->startTimer());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_stopTimer(int64_t *result, const DeviceContext *ctx,
                                const DeviceTimer *timer) {
  std::chrono::nanoseconds elapsed_nanos;
  const char *err_msg = check_val(&elapsed_nanos, std::chrono::nanoseconds{0},
                                  ctx->stopTimer(*timer));
  *result = elapsed_nanos.count();
  return err_msg;
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
AsyncRT_DeviceContext_getDriverVersion(int *result, const DeviceContext *ctx) {
  return check_val(result, 0, ctx->device().getDriverVersion());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_getAttribute(int *result, const DeviceContext *ctx,
                                   int attr) {
  return check_val(
      result, 0,
      ctx->device().getAttribute(static_cast<Device::Attribute>(attr)));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_isCompatibleWithMAX(const DeviceContext *ctx) {
  return check_success(ctx->device().isCompatibleWithMAX());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_computeCapability(int32_t *result,
                                        const DeviceContext *ctx) {
  int compute;
  const char *err_msg =
      check_val(&compute, 0, ctx->device().computeCapability());
  *result = compute;
  return err_msg;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_getMemoryInfo(const DeviceContext *ctx, size_t *free,
                                    size_t *total) {
  Device::MemInfo mem_info;
  const char *err_msg =
      check_val(&mem_info, Device::MemInfo(), ctx->device().getMemoryInfo());
  *free = mem_info.free;
  *total = mem_info.total;
  return err_msg;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_canAccess(bool *result, const DeviceContext *ctx,
                                const DeviceContext *peer) {
  return check_val(result, false, ctx->device().canAccess(peer->device()));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_enablePeerAccess(const DeviceContext *ctx,
                                       const DeviceContext *peer) {
  return check_success(ctx->enablePeerAccess(*peer));
}

MODULAR_EXPORT int32_t AsyncRT_DeviceContext_numberOfDevices(const char *kind) {
  return Device::numberOfDevices(kind);
}

//
// Raw CUDA accessors
//
MODULAR_EXPORT const char *
AsyncRT_DeviceContext_cuda_context(CUcontext *result,
                                   const DeviceContext *ctx) {
  const CUDADeviceContext *verified_ctx = CUDADeviceContext::verified_cast(ctx);
  if (verified_ctx == nullptr) {
    return strdup("Not a CUDAContext");
  }
  *result = verified_ctx->cuda_context();
  return nullptr;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceStream_cuda_stream(CUstream *result, const DeviceStream *stream) {
  const CUDADeviceStream *verified_stream =
      static_cast<const CUDADeviceStream *>(stream);
  if (verified_stream == nullptr) {
    return strdup("Not a CUDAStream");
  }
  *result = verified_stream->cuda_stream();
  return nullptr;
}

MODULAR_EXPORT const char *AsyncRT_cuda_tensorMapEncodeTiled(
    void *tensorMap, int32_t tensorDataType, uint32_t tensorRank,
    const DeviceBuffer *globalAddress, const uint64_t *globalDim,
    const uint64_t *globalStrides, const uint32_t *boxDim,
    const uint32_t *elementStrides, int32_t interleave, int32_t swizzle,
    int32_t l2Promotion, int32_t oobFill) {
  // Extract the device context from the device buffer.
  const CUDADeviceContext *verified_ctx =
      CUDADeviceContext::verified_cast(&globalAddress->context());
  if (verified_ctx == nullptr) {
    return strdup("Not a CUDADeviceBuffer");
  }
  return check_success(verified_ctx->cuda_TensorMapEncodeTiled(
      tensorMap, static_cast<CUtensorMapDataType>(tensorDataType), tensorRank,
      *globalAddress, globalDim, globalStrides, boxDim, elementStrides,
      static_cast<CUtensorMapInterleave>(interleave),
      static_cast<CUtensorMapSwizzle>(swizzle),
      static_cast<CUtensorMapL2promotion>(l2Promotion),
      static_cast<CUtensorMapFloatOOBfill>(oobFill)));
}
