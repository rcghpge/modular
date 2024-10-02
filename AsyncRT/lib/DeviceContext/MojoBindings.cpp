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
#include "Support/SymbolExport.h"

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

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_create(const DeviceContext **result, const char *kind,
                             int id) {
  return check_ref(result, DeviceContext::create(kind, id));
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
AsyncRT_DeviceContext_mallocHost(void **result, const DeviceContext *ctx,
                                 size_t size) {
  return check_val(result, nullptr, ctx->malloc_host(size));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_freeHost(const DeviceContext *ctx, void *ptr) {
  return check_success(ctx->free_host(ptr));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_createBuffer(const DeviceBuffer **result,
                                   void **device_ptr, const DeviceContext *ctx,
                                   size_t len, size_t elem_size) {
  const char *err_msg = check_ref(result, ctx->createBuffer(len, elem_size));
  *device_ptr = (*result)->ptr();
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

MODULAR_EXPORT int64_t AsyncRT_DeviceBuffer_len(const DeviceBuffer *buffer) {
  return buffer->len();
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_synchronize(const DeviceContext *ctx) {
  return check_success(ctx->synchronize());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_HtoD_async(const DeviceContext *ctx,
                                 const DeviceBuffer *dst, const void *src) {
  assert((ctx == &dst->context()) && "ctx not matching dst");
  return check_success(ctx->enqueueCopyToDevice(*dst, src));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_DtoH_async(const DeviceContext *ctx, void *dst,
                                 const DeviceBuffer *src) {
  assert((ctx == &src->context()) && "ctx not matching src");
  return check_success(ctx->enqueueCopyFromDevice(dst, *src));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_DtoD_async(const DeviceContext *ctx,
                                 const DeviceBuffer *dst,
                                 const DeviceBuffer *src) {
  return check_success(ctx->enqueueCopyDeviceToDevice(*dst, *src));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_compileFunction(
    const DeviceFunction **result, const DeviceContext *ctx,
    const char *function_name, const void *data) {
  DeviceFunction::SignatureBuilder sig;
  return check_ref(result, ctx->compileFunction(function_name, sig, data));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_enqueueFunctionDirect(
    const DeviceContext *ctx, const DeviceFunction *func, uint32_t gridX,
    uint32_t gridY, uint32_t gridZ, uint32_t blockX, uint32_t blockY,
    uint32_t blockZ, void **args) {
  assert((ctx == &func->context()) && "ctx not matching func");
  return check_success(ctx->enqueueFunctionExecDirect(
      *func, gridX, gridY, gridZ, blockX, blockY, blockZ, args));
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
