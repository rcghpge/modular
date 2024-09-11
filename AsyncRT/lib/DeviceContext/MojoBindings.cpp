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

#include <iostream>

using namespace M;
using namespace M::AsyncRT;

//===----------------------------------------------------------------------===//
// DeviceContext
//===----------------------------------------------------------------------===//

const char *check_success(ErrorOrSuccess err_or) {
  if (err_or) {
    // Make a malloc'd copy of the string, so that we can safely free the
    // pointer from Mojo.
    std::cerr << "      +-err-> " << err_or.getError() << "\n";
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
    std::cerr << "      +-err-> " << err_or.getError() << "\n";
    return strdup(err_or.getError());
  }
  // Return the pointer, transfering the ownership to Mojo.
  *result = err_or.get().release();
  std::cerr << "      +-----> " << *result << "\n";
  return nullptr;
}

const char *check_ptr(void **result, ErrorOr<void *> err_or) {
  if (err_or) {
    *result = nullptr;
    // Make a malloc'd copy of the string, so that we can safely free the
    // pointer from Mojo.
    std::cerr << "      +-err-> " << err_or.getError() << "\n";
    return strdup(err_or.getError());
  }
  *result = err_or.get();
  std::cerr << "      +-----> " << *result << "\n";
  return nullptr;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_create(const DeviceContext **result, const char *kind,
                             int id) {
  std::cerr << "@#@#@ AsyncRT_DeviceContext_create: " << kind << ", " << id
            << "\n";
  return check_ref(result, DeviceContext::create(kind, id));
}

MODULAR_EXPORT void AsyncRT_DeviceContext_release(const DeviceContext *ctx) {
  // Decrement the reference count by adopting it in the ref.
  std::cerr << "@#@#@ AsyncRT_DeviceContext_release: " << ctx << "\n";
  DeviceContextRef::take(ctx);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_mallocHost(void **result, const DeviceContext *ctx,
                                 size_t size) {
  std::cerr << "@#@#@ AsyncRT_DeviceContext_mallocHost: " << ctx << ", " << size
            << "\n";
  return check_ptr(result, ctx->malloc_host(size));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_freeHost(const DeviceContext *ctx, void *ptr) {
  std::cerr << "@#@#@ AsyncRT_DeviceContext_freeHost: " << ctx << ", " << ptr
            << "\n";
  return check_success(ctx->free_host(ptr));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_createBuffer(const DeviceBuffer **result,
                                   const DeviceContext *ctx, size_t len,
                                   size_t elem_size) {
  std::cerr << "@#@#@ AsyncRT_DeviceContext_createBuffer: " << ctx << ", "
            << len << ", " << elem_size << "\n";
  return check_ref(result, ctx->createBuffer(len, elem_size));
}

MODULAR_EXPORT void AsyncRT_DeviceBuffer_release(const DeviceBuffer *buffer) {
  // Decrement the reference count by adopting it in the ref.
  std::cerr << "@#@#@ AsyncRT_DeviceBuffer_release: " << buffer << "\n";
  DeviceBufferRef::take(buffer);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_synchronize(const DeviceContext *ctx) {
  std::cerr << "@#@#@ AsyncRT_DeviceContext_synchronize: " << ctx << "\n";
  return check_success(ctx->synchronize());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_HtoD_async(const DeviceContext *ctx,
                                 const DeviceBuffer *dst, const void *src) {
  assert((ctx == &dst->context()) && "ctx not matching dst");
  std::cerr << "@#@#@ AsyncRT_DeviceContext_HtoD_async: " << ctx << ", " << dst
            << ", " << src << "\n";
  return check_success(ctx->enqueueCopyToDevice(*dst, src));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_DtoH_async(const DeviceContext *ctx, void *dst,
                                 const DeviceBuffer *src) {
  assert((ctx == &src->context()) && "ctx not matching src");
  std::cerr << "@#@#@ AsyncRT_DeviceContext_DtoH_async: " << ctx << ", " << dst
            << ", " << src << "\n";
  return check_success(ctx->enqueueCopyFromDevice(dst, *src));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_DtoD_async(const DeviceContext *ctx,
                                 const DeviceBuffer *dst,
                                 const DeviceBuffer *src) {
  std::cerr << "@#@#@ AsyncRT_DeviceContext_DtoD_async: " << ctx << ", " << dst
            << ", " << src << "\n";
  return check_success(ctx->enqueueCopyDeviceToDevice(*dst, *src));
}
