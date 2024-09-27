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
#include "Support/ErrorOr.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

namespace M::AsyncRT {
namespace {
Error formatError(const char *file, int line, const char *func,
                  const char *msg = nullptr) {
  return Error(llvm::Twine(file) + ":" + llvm::Twine(line) + " - " + func +
               " failed" + ((msg == nullptr) ? "." : ": ") +
               ((msg == nullptr) ? "" : msg));
}
} // namespace
} // namespace M::AsyncRT

#define RAISE_ERROR()                                                          \
  do {                                                                         \
    return formatError(__FILE__, __LINE__, __FUNCTION__);                      \
  } while (0)

#define RAISE_ERROR_MESSAGE(msg)                                               \
  do {                                                                         \
    return formatError(__FILE__, __LINE__, __FUNCTION__, msg);                 \
  } while (0)

namespace M::AsyncRT {

//
// ===== CpuDevice ===== //
//
int CpuDevice::id() const { return 0; }

std::string CpuDevice::name() const {
  // TODO(iposva): Return the actual device name.
  return "cpu";
}

ErrorOr<int> CpuDevice::numberOfDevices() {
  // Only a single CpuDevice is supported.
  return 1;
}

//
// ===== CpuDeviceBuffer === //
//
CpuDeviceBuffer::CpuDeviceBuffer(const CpuDeviceContext &context, size_t len,
                                 size_t elem_size, void *ptr)
    : DeviceBuffer(context, len, elem_size), ptr_(ptr) {}

CpuDeviceBuffer::~CpuDeviceBuffer() { free(ptr_); }

void *CpuDeviceBuffer::cpu_ptr() const { return ptr_; }
void *CpuDeviceBuffer::ptr() const { return cpu_ptr(); }

const CpuDeviceBuffer &CpuDeviceBuffer::cast(const DeviceBuffer &buf) {
  return static_cast<const CpuDeviceBuffer &>(buf);
}

//
// ===== CpuDeviceContext === //
//
CpuDeviceContext::CpuDeviceContext(const CpuDevice &device)
    : DeviceContext(device) {}

ErrorOr<void *> CpuDeviceContext::malloc_host(size_t size) const {
  void *ptr = malloc(size);
  if (ptr == nullptr) {
    RAISE_ERROR();
  }
  return ptr;
}

ErrorOrSuccess CpuDeviceContext::free_host(void *ptr) const {
  free(ptr);
  return success();
}

ErrorOr<DeviceBufferRef>
CpuDeviceContext::createBuffer(size_t len, size_t elem_size) const {
  size_t bytesize = len * elem_size;
  void *ptr = malloc(bytesize);
  if (ptr == nullptr) {
    RAISE_ERROR();
  }
  return CpuDeviceBufferRef::create(*this, len, elem_size, ptr);
}

ErrorOrSuccess CpuDeviceContext::enqueueCopyToDevice(const DeviceBuffer &dst,
                                                     const void *src) const {
  void *dst_ptr = CpuDeviceBuffer::cast(dst).cpu_ptr();
  size_t byte_size = dst.len() * dst.elem_size();
  memcpy(dst_ptr, src, byte_size);
  return success();
}

ErrorOrSuccess
CpuDeviceContext::enqueueCopyFromDevice(void *dst,
                                        const DeviceBuffer &src) const {
  void *src_ptr = CpuDeviceBuffer::cast(src).cpu_ptr();
  size_t byte_size = src.len() * src.elem_size();
  memcpy(dst, src_ptr, byte_size);
  return success();
}

ErrorOrSuccess
CpuDeviceContext::enqueueCopyDeviceToDevice(const DeviceBuffer &dst,
                                            const DeviceBuffer &src) const {
  void *src_ptr = CpuDeviceBuffer::cast(src).cpu_ptr();
  void *dst_ptr = CpuDeviceBuffer::cast(dst).cpu_ptr();
  size_t byte_size = dst.len() * dst.elem_size();
  memcpy(dst_ptr, src_ptr, byte_size);
  return success();
}

ErrorOr<DeviceFunctionRef> CpuDeviceContext::compileFunction(
    const char *function_name,
    const DeviceFunction::SignatureBuilder &signature, const void *data) const {
  // Currently not implemented. Maybe never to be implemented.
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOrSuccess CpuDeviceContext::enqueueFunctionExec(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ,
    const DeviceFunction::Arguments &args) const {
  // Currently not implemented. Maybe never to be implemented.
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOrSuccess CpuDeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, void **args) const {
  // Currently not implemented. Maybe never to be implemented.
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOrSuccess CpuDeviceContext::synchronize() const {
  // Currently we do not schedule asynchronous work.
  return success();
}

const CpuDeviceContext &CpuDeviceContext::cast(const DeviceContext &ctx) {
  return static_cast<const CpuDeviceContext &>(ctx);
}

ErrorOr<CpuDeviceContextRef> CpuDeviceContext::create(int id) {
  if (id != 0) {
    return Error("CPU device only supports id == 0.");
  }
  return CpuDeviceContextRef::create(*CpuDeviceRef::create());
}

} // namespace M::AsyncRT
