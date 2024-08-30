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

MODULAR_EXPORT const DeviceContext *
AsyncRT_DeviceContext_create(const char *kind) {
  auto ref = DeviceContext::create(kind);
  // Return the pointer while keeping the reference count.
  return ref.release();
}

MODULAR_EXPORT void AsyncRT_DeviceContext_release(const DeviceContext *ctx) {
  // Decrement the reference count
  DeviceContextRef::take(ctx);
}

MODULAR_EXPORT void *AsyncRT_DeviceContext_mallocHost(const DeviceContext *ctx,
                                                      size_t size) {
  return ctx->malloc_host(size);
}

MODULAR_EXPORT void AsyncRT_DeviceContext_freeHost(const DeviceContext *ctx,
                                                   void *ptr) {
  ctx->free_host(ptr);
}

MODULAR_EXPORT const DeviceBuffer *
AsyncRT_DeviceContext_createBuffer(const DeviceContext *ctx, size_t len,
                                   size_t size) {
  auto ref = ctx->createBuffer(len, size);
  // Return the pointer while keeping the reference count.
  return ref.release();
}

MODULAR_EXPORT void AsyncRT_DeviceBuffer_release(const DeviceBuffer *buffer) {
  // Decrement the reference count
  DeviceBufferRef::take(buffer);
}

MODULAR_EXPORT void
AsyncRT_DeviceContext_synchronize(const DeviceContext *ctx) {
  ctx->synchronize();
}

MODULAR_EXPORT void AsyncRT_DeviceContext_HtoD_async(const DeviceContext *ctx,
                                                     const DeviceBuffer *dst,
                                                     const void *src) {
  ctx->enqueueCopyToDevice(*dst, src);
}

MODULAR_EXPORT void AsyncRT_DeviceContext_DtoH_async(const DeviceContext *ctx,
                                                     void *dst,
                                                     const DeviceBuffer *src) {
  ctx->enqueueCopyFromDevice(dst, *src);
}

MODULAR_EXPORT void AsyncRT_DeviceContext_DtoD_async(const DeviceContext *ctx,
                                                     const DeviceBuffer *dst,
                                                     const DeviceBuffer *src) {
  ctx->enqueueCopyDeviceToDevice(*dst, *src);
}
