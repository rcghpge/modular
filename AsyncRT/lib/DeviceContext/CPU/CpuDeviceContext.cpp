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

#include <chrono>
#include <cstdint>
#include <cstring>
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
// TODO(iposva): Consider using the actual device name.
CpuDevice::CpuDevice() : Device("cpu") {}

int CpuDevice::id() const { return 0; }

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
CpuDeviceBuffer::CpuDeviceBuffer(const CpuDeviceContext &context,
                                 const CpuDeviceBuffer &parent, size_t len,
                                 size_t elem_size, void *ptr)
    : DeviceBuffer(context, parent, len, elem_size), ptr_(ptr) {}
// TODO: Remove the V1 compatibility.
CpuDeviceBuffer::CpuDeviceBuffer(const CpuDeviceContext &context, size_t len,
                                 size_t elem_size, void *ptr, bool owning)
    : DeviceBuffer(context, len, elem_size), ptr_(ptr), owning_(owning) {}

CpuDeviceBuffer::~CpuDeviceBuffer() {
  if (!owning_) {
    // TODO: Remove the V1 compatibility.
    // This is a device buffer created by pointer in Mojo while ownership is
    // maintained elsewhere.
    return;
  }
  if (parent() != nullptr) {
    // Parent owns the memory.
    return;
  }
  free(ptr_);
}

void *CpuDeviceBuffer::cpu_ptr() const { return ptr_; }
void *CpuDeviceBuffer::ptr() const { return cpu_ptr(); }

// TODO: Remove the V1 compatibility.
void CpuDeviceBuffer::releasePtr() const {
  ptr_ = nullptr;
  owning_ = false;
}

ErrorOr<DeviceBufferRef>
CpuDeviceBuffer::createSubBuffer(size_t offset, size_t len,
                                 size_t elem_size) const {
  char *orig_ptr = static_cast<char *>(this->cpu_ptr());
  size_t orig_byte_size = this->len() * this->elem_size();

  size_t byte_offset = offset * elem_size;
  size_t byte_size = len * elem_size;
  if ((byte_size > orig_byte_size) ||
      (orig_byte_size - byte_size < byte_offset)) {
    RAISE_ERROR_MESSAGE("offset, len and elem_size are out of range");
  }
  return CpuDeviceBufferRef::create(CpuDeviceContext::cast(context()), *this,
                                    len, elem_size, orig_ptr + byte_offset);
}

const CpuDeviceBuffer &CpuDeviceBuffer::cast(const DeviceBuffer &buf) {
  return static_cast<const CpuDeviceBuffer &>(buf);
}

// ===== CpuDeviceTimer ===== //
CpuDeviceTimer::CpuDeviceTimer(
    const CpuDeviceContext &context,
    std::chrono::time_point<std::chrono::steady_clock> start)
    : DeviceTimer(context), start_(start) {}

std::chrono::time_point<std::chrono::steady_clock>
CpuDeviceTimer::cpu_start() const {
  return start_;
}

const CpuDeviceTimer &CpuDeviceTimer::cast(const DeviceTimer &timer) {
  return static_cast<const CpuDeviceTimer &>(timer);
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

ErrorOr<DeviceBufferRef> CpuDeviceContext::createBuffer(size_t len,
                                                        size_t elem_size,
                                                        SyncMode mode) const {
  size_t bytesize = len * elem_size;
  void *ptr = malloc(bytesize);
  if (ptr == nullptr) {
    RAISE_ERROR();
  }
  return CpuDeviceBufferRef::create(*this, len, elem_size, ptr);
}

DeviceBufferRef CpuDeviceContext::createBufferOwning(void *ptr, size_t len,
                                                     size_t elem_size,
                                                     bool owning) const {
  return CpuDeviceBufferRef::create(*this, len, elem_size, ptr, owning);
}

ErrorOrSuccess CpuDeviceContext::copyToDevice(const DeviceBuffer &dst,
                                              const void *src,
                                              SyncMode mode) const {
  void *dst_ptr = CpuDeviceBuffer::cast(dst).cpu_ptr();
  size_t byte_size = dst.len() * dst.elem_size();
  memcpy(dst_ptr, src, byte_size);
  return success();
}

ErrorOrSuccess CpuDeviceContext::copyFromDevice(void *dst,
                                                const DeviceBuffer &src,
                                                SyncMode mode) const {
  void *src_ptr = CpuDeviceBuffer::cast(src).cpu_ptr();
  size_t byte_size = src.len() * src.elem_size();
  memcpy(dst, src_ptr, byte_size);
  return success();
}

ErrorOrSuccess CpuDeviceContext::copyDeviceToDevice(const DeviceBuffer &dst,
                                                    const DeviceBuffer &src,
                                                    SyncMode mode) const {
  void *src_ptr = CpuDeviceBuffer::cast(src).cpu_ptr();
  void *dst_ptr = CpuDeviceBuffer::cast(dst).cpu_ptr();
  size_t dst_byte_size = dst.len() * dst.elem_size();
  size_t src_byte_size = src.len() * src.elem_size();
  if (src_byte_size < dst_byte_size) {
    RAISE_ERROR_MESSAGE("not enough data in src");
  }
  memcpy(dst_ptr, src_ptr, dst_byte_size);
  return success();
}

ErrorOrSuccess CpuDeviceContext::setMemory(const DeviceBuffer &dst,
                                           uint32_t val, size_t elem_size,
                                           SyncMode mode) const {
  void *ptr = CpuDeviceBuffer::cast(dst).cpu_ptr();
  size_t len = dst.len();
  if (elem_size == 1) {
    memset(ptr, static_cast<uint8_t>(val), len);
  } else if (elem_size == 2) {
    uint16_t *cur_ptr = static_cast<uint16_t *>(ptr);
    for (size_t i = 0; i < len; i++) {
      *cur_ptr++ = static_cast<uint16_t>(val);
    }
  } else {
    uint32_t *cur_ptr = static_cast<uint32_t *>(ptr);
    for (size_t i = 0; i < len; i++) {
      *cur_ptr++ = val;
    }
  }
  return success();
}

ErrorOr<DeviceFunctionRef> CpuDeviceContext::compileFunction(
    const char *function_name,
    const DeviceFunction::SignatureBuilder &signature, const void *data,
    int32_t max_registers, int32_t threads_per_block, int32_t cache_mode,
    int32_t cache_config, int32_t max_dynamic_shared_bytes,
    const char *debug_level, int32_t optimization_level) const {
  // Currently not implemented. Maybe never to be implemented.
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOrSuccess CpuDeviceContext::enqueueFunctionExec(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    const DeviceFunction::Arguments &args) const {
  // Currently not implemented. Maybe never to be implemented.
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOrSuccess CpuDeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t num_attrs, void **args) const {
  // Currently not implemented. Maybe never to be implemented.
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOrSuccess CpuDeviceContext::synchronize() const {
  // Currently we do not schedule asynchronous work.
  return success();
}

ErrorOr<DeviceTimerRef> CpuDeviceContext::startTimer() const {
  return CpuDeviceTimerRef::create(*this, std::chrono::steady_clock::now());
}

ErrorOr<std::chrono::nanoseconds>
CpuDeviceContext::stopTimer(const DeviceTimer &timer) const {
  return std::chrono::steady_clock::now() -
         CpuDeviceTimer::cast(timer).cpu_start();
}

ErrorOr<DeviceMemInfo> CpuDeviceContext::getMemoryInfo() const {
  // TODO: Implement get memory info for CPU device context.
  return DeviceMemInfo();
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
