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
#include "AsyncRT/Runtime/Globals/Globals.h"
#include "Support/AlignedAlloc.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/Twine.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <sys/resource.h>
#include <unistd.h>

namespace M::AsyncRT {
namespace {

// Allow printing of detailed error messages with C++ source information when
// debugging low-level implementation.
static constexpr bool detailed_err_msg = false;

Error formatError(const char *file, int line, const char *func,
                  const char *msg = nullptr) {
  if (detailed_err_msg || (msg == nullptr)) {
    return Error(llvm::Twine(file) + ":" + llvm::Twine(line) + " - " + func +
                 " failed" + ((msg == nullptr) ? "." : ": ") +
                 ((msg == nullptr) ? "" : msg));
  }
  return Error(msg);
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
CpuDevice::CpuDevice() : Device(api, "cpu", 0, M::kPreferredMemoryAlignment) {}

ErrorOr<int> CpuDevice::getDriverVersion() const { return 0; };

ErrorOrSuccess CpuDevice::isCompatibleWithMAX() const { return success(); }

ErrorOr<int> CpuDevice::computeCapability() const {
  // RUNP-494 Remove this, compute capability is a CUDA-specific concept.
  return 0;
}

ErrorOr<bool> CpuDevice::canAccess(const Device &peer) const { return false; }

ErrorOr<int> CpuDevice::getAttribute(Attribute attr) const { return 0; }

ErrorOr<DeviceContextRef>
CpuDevice::createContext(size_t bufferCacheSize) const {
  return CpuDeviceContextRef::create(*this);
}

int CpuDevice::deviceCount() const {
  // Only a single CpuDevice is supported.
  return 1;
}

ErrorOr<const CpuDevice &> CpuDevice::get(int id) {
  if (id != 0) {
    RAISE_ERROR_MESSAGE("CPU device only supports id == 0.");
  }

  static CpuDevice *device = new CpuDevice();
  return *device;
}

//
// ===== CpuDeviceBuffer === //
//
CpuDeviceBuffer::CpuDeviceBuffer(const CpuDeviceContext &context,
                                 size_t byteSize, void *ptr)
    : DeviceBuffer(context, byteSize), ptr_(ptr) {}
CpuDeviceBuffer::CpuDeviceBuffer(const CpuDeviceContext &context,
                                 const CpuDeviceBuffer &parent, size_t byteSize,
                                 void *ptr)
    : DeviceBuffer(context, parent, byteSize), ptr_(ptr) {}
// TODO: Remove the V1 compatibility.
CpuDeviceBuffer::CpuDeviceBuffer(const CpuDeviceContext &context,
                                 size_t byteSize, void *ptr, bool owning)
    : DeviceBuffer(context, byteSize), ptr_(ptr), owning_(owning) {}

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
  TCMallocGlobals::tc_delete(ptr_);
}

void *CpuDeviceBuffer::cpuPtr() const { return ptr_; }
void *CpuDeviceBuffer::ptr() const { return cpuPtr(); }

// TODO: Remove the V1 compatibility.
void *CpuDeviceBuffer::releasePtr() const {
  auto tmp = ptr_;
  ptr_ = nullptr;
  owning_ = false;
  return tmp;
}

ErrorOr<DeviceBufferRef>
CpuDeviceBuffer::createSubBuffer(size_t byteOffset, size_t byteSize) const {
  char *orig_ptr = static_cast<char *>(this->cpuPtr());
  size_t origByteSize = this->byteSize();

  if ((byteSize > origByteSize) || (origByteSize - byteSize < byteOffset)) {
    RAISE_ERROR_MESSAGE("offset, len and elem_size are out of range");
  }
  return CpuDeviceBufferRef::create(CpuDeviceContext::cast(context()), *this,
                                    byteSize, orig_ptr + byteOffset);
}

const CpuDeviceBuffer &CpuDeviceBuffer::cast(const DeviceBuffer &buf) {
  return static_cast<const CpuDeviceBuffer &>(buf);
}

// ===== CpuDeviceStream ===== //
CpuDeviceStream::CpuDeviceStream(const CpuDeviceContext &context)
    : DeviceStream(context) {}

ErrorOrSuccess CpuDeviceStream::synchronize() const { return success(); }

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

ErrorOr<void *> CpuDeviceContext::mallocHost(size_t size) const {
  void *ptr = TCMallocGlobals::tc_new(size, 1);
  if (ptr == nullptr) {
    RAISE_ERROR();
  }
  return ptr;
}

ErrorOrSuccess CpuDeviceContext::freeHost(void *ptr) const {
  TCMallocGlobals::tc_delete(ptr);
  return success();
}

ErrorOr<DeviceBufferRef> CpuDeviceContext::createBuffer(size_t byteSize,
                                                        SyncMode mode) const {
  void *ptr = nullptr;
  if (byteSize != 0) {
    ptr = TCMallocGlobals::tc_new(byteSize, 1);
    if (ptr == nullptr) {
      RAISE_ERROR();
    }
  }
  return CpuDeviceBufferRef::create(*this, byteSize, ptr);
}

DeviceBufferRef CpuDeviceContext::createBufferOwning(void *ptr, size_t byteSize,
                                                     bool owning) const {
  return CpuDeviceBufferRef::create(*this, byteSize, ptr, owning);
}

ErrorOrSuccess CpuDeviceContext::copyToDevice(const DeviceBuffer &dst,
                                              const void *src,
                                              SyncMode mode) const {
  void *dst_ptr = CpuDeviceBuffer::cast(dst).cpuPtr();
  memcpy(dst_ptr, src, dst.byteSize());
  return success();
}

ErrorOrSuccess CpuDeviceContext::copyFromDevice(void *dst,
                                                const DeviceBuffer &src,
                                                SyncMode mode) const {
  void *src_ptr = CpuDeviceBuffer::cast(src).cpuPtr();
  memcpy(dst, src_ptr, src.byteSize());
  return success();
}

ErrorOrSuccess CpuDeviceContext::copyDeviceToDevice(const DeviceBuffer &dst,
                                                    const DeviceBuffer &src,
                                                    SyncMode mode) const {
  void *src_ptr = CpuDeviceBuffer::cast(src).cpuPtr();
  void *dst_ptr = CpuDeviceBuffer::cast(dst).cpuPtr();
  if (src.byteSize() < dst.byteSize()) {
    RAISE_ERROR_MESSAGE("not enough data in src");
  }
  memcpy(dst_ptr, src_ptr, dst.byteSize());
  return success();
}

ErrorOrSuccess CpuDeviceContext::setMemory(const DeviceBuffer &dst,
                                           uint32_t val, size_t valSize,
                                           SyncMode mode) const {
  void *ptr = CpuDeviceBuffer::cast(dst).cpuPtr();
  size_t byteSize = dst.byteSize();
  if (valSize == 1) {
    memset(ptr, static_cast<uint8_t>(val), byteSize);
  } else if (valSize == 2) {
    if ((byteSize & 1) != 0) {
      return Error("byteSize of buffer is misaliged");
    }
    size_t len = byteSize >> 1;
    uint16_t *cur_ptr = static_cast<uint16_t *>(ptr);
    for (size_t i = 0; i < len; i++) {
      *cur_ptr++ = static_cast<uint16_t>(val);
    }
  } else {
    if (valSize != 4) {
      return Error("unsupported valSize");
    }
    if ((byteSize & 3) != 0) {
      return Error("byteSize of buffer is misaliged");
    }
    size_t len = byteSize >> 2;
    uint32_t *cur_ptr = static_cast<uint32_t *>(ptr);
    for (size_t i = 0; i < len; i++) {
      *cur_ptr++ = val;
    }
  }
  return success();
}

ErrorOr<DeviceFunctionRef> CpuDeviceContext::loadFunction(
    const char *moduleName, const char *functionName,
    const DeviceFunction::SignatureBuilder &signature, const void *data,
    int32_t maxRegisters, int32_t threadsPerBlock, int32_t cacheMode,
    int32_t cacheConfig, int32_t maxDynamicSharedBytes, const char *debugLevel,
    int32_t optimizationLevel) const {
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
    void *attrs, uint32_t numAttrs, void **args) const {
  // Currently not implemented. Maybe never to be implemented.
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOr<DeviceStreamRef> CpuDeviceContext::stream() const {
  return CpuDeviceStreamRef::create(*this);
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

ErrorOrSuccess
CpuDeviceContext::enablePeerAccess(const DeviceContext &peer) const {
  // TODO(iposva): Verify that peer is a CpuDeviceContext.
  return success();
}

// Call sysconf() and check for errors.
static ErrorOr<size_t> getSystemValue(int name) {
  errno = 0;
  auto value = sysconf(name);
  if (errno != 0)
    RAISE_ERROR_MESSAGE(strerror(errno));
  return static_cast<size_t>(value);
}

ErrorOr<DeviceContext::MemInfo> CpuDeviceContext::getMemoryInfo() const {
  // TODO support platforms besides UNIX.
  MemInfo info;
  auto numPages = getSystemValue(_SC_PHYS_PAGES);
  if (numPages.isError())
    return numPages.takeError();
  auto pageSize = getSystemValue(_SC_PAGE_SIZE);
  if (pageSize.isError())
    return pageSize.takeError();

  rusage usage;
  if (-1 == getrusage(RUSAGE_SELF, &usage))
    RAISE_ERROR_MESSAGE("Failed to get current memory usage via rusage");
  // This is the maximum resident set size used (in kilobytes).
  // Current memory use may be lower, this is the "high water mark".
  size_t used = usage.ru_maxrss * 1024;
  // Total memory = number of pages times the page size.
  info.total = numPages.takeValue() * pageSize.takeValue();
  // Free memory = total memory minus memory in use.
  info.free = info.total - used;
  return info;
}

const CpuDeviceContext &CpuDeviceContext::cast(const DeviceContext &ctx) {
  return static_cast<const CpuDeviceContext &>(ctx);
}

} // namespace M::AsyncRT
