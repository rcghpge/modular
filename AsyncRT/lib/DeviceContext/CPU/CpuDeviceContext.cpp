//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CPU device context.
//
//===----------------------------------------------------------------------===//

#include "CpuDeviceContext.h"

#include "AsyncRT/DeviceContext/DeviceContext.h"

#include <cstring>
#include <iostream>
#include <string>

#define RAISE_ERROR_RETURN(result)                                             \
  do {                                                                         \
    std::cerr << __FILE__ << ":" << __LINE__ << " -  " << __FUNCTION__         \
              << " failed.\n";                                                 \
    abort();                                                                   \
    return result;                                                             \
  } while (0)

namespace M::AsyncRT {

//
// ===== CpuDevice ===== //
//
CpuDevice::CpuDevice() : Device() {}

CpuDevice::~CpuDevice() {}

std::string CpuDevice::name() const {
  // TODO(iposva): Return the actual device name.
  return "cpu";
}

//
// ===== CpuDeviceBuffer === //
//
CpuDeviceBuffer::CpuDeviceBuffer(CpuDeviceContextRef context, size_t len,
                                 size_t elem_size, void *ptr)
    : DeviceBuffer(std::move(context), len, elem_size), ptr_(ptr) {}

CpuDeviceBuffer::~CpuDeviceBuffer() { free(ptr_); }

void *CpuDeviceBuffer::cpu_ptr() const { return ptr_; }

const CpuDeviceBuffer &CpuDeviceBuffer::cast(const DeviceBuffer &buf) {
  return static_cast<const CpuDeviceBuffer &>(buf);
}

//
// ===== CpuDeviceContext === //
//
CpuDeviceContext::CpuDeviceContext(CpuDeviceRef device)
    : DeviceContext(std::move(device)) {}

CpuDeviceContext::~CpuDeviceContext() {
  // A CpuDeviceContext does not currently hold any local information.
}

void *CpuDeviceContext::malloc_host(size_t size) const {
  void *ptr = malloc(size);
  if (ptr == nullptr) {
    RAISE_ERROR_RETURN(nullptr);
  }
  return ptr;
}

void CpuDeviceContext::free_host(void *ptr) const { free(ptr); }

DeviceBufferRef CpuDeviceContext::createBuffer(size_t len,
                                               size_t elem_size) const {
  size_t bytesize = len * elem_size;
  void *ptr = malloc(bytesize);
  if (ptr == nullptr) {
    RAISE_ERROR_RETURN(CpuDeviceBufferRef());
  }
  return CpuDeviceBufferRef::create(CpuDeviceContextRef::copy(this), len,
                                    elem_size, ptr);
}

void CpuDeviceContext::enqueueCopyToDevice(const DeviceBuffer &dst,
                                           const void *src) const {
  void *dst_ptr = CpuDeviceBuffer::cast(dst).cpu_ptr();
  size_t byte_size = dst.len() * dst.elem_size();
  memcpy(dst_ptr, src, byte_size);
}

void CpuDeviceContext::enqueueCopyFromDevice(void *dst,
                                             const DeviceBuffer &src) const {
  void *src_ptr = CpuDeviceBuffer::cast(src).cpu_ptr();
  size_t byte_size = src.len() * src.elem_size();
  memcpy(dst, src_ptr, byte_size);
}
void CpuDeviceContext::enqueueCopyDeviceToDevice(
    const DeviceBuffer &dst, const DeviceBuffer &src) const {
  void *src_ptr = CpuDeviceBuffer::cast(src).cpu_ptr();
  void *dst_ptr = CpuDeviceBuffer::cast(dst).cpu_ptr();
  size_t byte_size = dst.len() * dst.elem_size();
  memcpy(dst_ptr, src_ptr, byte_size);
}

void CpuDeviceContext::synchronize() const {
  // Currently we do not schedule asynchronous work.
}

const CpuDeviceContext &CpuDeviceContext::cast(const DeviceContext &ctx) {
  return static_cast<const CpuDeviceContext &>(ctx);
}

CpuDeviceContextRef CpuDeviceContext::create() {
  CpuDeviceRef device = CpuDeviceRef::create();
  if (!device) {
    return CpuDeviceContextRef();
  }

  return CpuDeviceContextRef::create(std::move(device));
}

} // namespace M::AsyncRT
