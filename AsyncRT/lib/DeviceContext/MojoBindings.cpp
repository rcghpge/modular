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

const char *check_ptr(void **result, ErrorOr<void *> err_or) {
  if (err_or) {
    *result = nullptr;
    // Make a malloc'd copy of the string, so that we can safely free the
    // pointer from Mojo.
    return strdup(err_or.getError());
  }
  *result = err_or.get();
  return nullptr;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_create(const DeviceContext **result, const char *kind) {
  return check_ref(result, DeviceContext::create(kind));
}

MODULAR_EXPORT void AsyncRT_DeviceContext_release(const DeviceContext *ctx) {
  // Decrement the reference count by adopting it in the ref.
  DeviceContextRef::take(ctx);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_mallocHost(void **result, const DeviceContext *ctx,
                                 size_t size) {
  return check_ptr(result, ctx->malloc_host(size));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_freeHost(const DeviceContext *ctx, void *ptr) {
  return check_success(ctx->free_host(ptr));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_createBuffer(const DeviceBuffer **result,
                                   const DeviceContext *ctx, size_t len,
                                   size_t size) {
  return check_ref(result, ctx->createBuffer(len, size));
}

MODULAR_EXPORT void AsyncRT_DeviceBuffer_release(const DeviceBuffer *buffer) {
  // Decrement the reference count by adopting it in the ref.
  DeviceBufferRef::take(buffer);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_synchronize(const DeviceContext *ctx) {
  return check_success(ctx->synchronize());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_HtoD_async(const DeviceContext *ctx,
                                 const DeviceBuffer *dst, const void *src) {
  return check_success(ctx->enqueueCopyToDevice(*dst, src));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_DtoH_async(const DeviceContext *ctx, void *dst,
                                 const DeviceBuffer *src) {
  return check_success(ctx->enqueueCopyFromDevice(dst, *src));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_DtoD_async(const DeviceContext *ctx,
                                 const DeviceBuffer *dst,
                                 const DeviceBuffer *src) {
  return check_success(ctx->enqueueCopyDeviceToDevice(*dst, *src));
}
