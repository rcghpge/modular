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
#include "AsyncRT/lib/DeviceContext/BufferCache.h"
#include "Support/AlignedAlloc.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/Twine.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <sys/resource.h>
#include <unistd.h>

#define RAISE_ERROR()                                                          \
  do {                                                                         \
    return DeviceContext::formatError(__FILE__, __LINE__, __FUNCTION__, "");   \
  } while (0)

#define RAISE_ERROR_MESSAGE(msg)                                               \
  do {                                                                         \
    return DeviceContext::formatError(__FILE__, __LINE__, __FUNCTION__, msg);  \
  } while (0)

#define CHECK_IS_CPU_ERR(dev)                                                  \
  do {                                                                         \
    if ((dev).api() != Device::cpuAPI) {                                       \
      RAISE_ERROR_MESSAGE("Not a CPU device.");                                \
    }                                                                          \
  } while (0)

namespace M::AsyncRT {

//
// ===== CpuDevice ===== //
//
// TODO(iposva): Consider using the actual device name.
CpuDevice::CpuDevice() : Device(api, "cpu", 0, M::kPreferredMemoryAlignment) {}

ErrorOr<int> CpuDevice::getDriverVersion() const { return 0; };

ErrorOrSuccess CpuDevice::isCompatible() const { return success(); }

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

ErrorOr<DeviceStreamRef>
CpuDevice::createExternalStream(void *externalStream) const {
  return CpuDeviceStreamRef::create(*this);
}

int CpuDevice::deviceCount() const {
  // Only a single CpuDevice is supported.
  return 1;
}

const CpuDevice &CpuDevice::cast(const Device &device) {
  return static_cast<const CpuDevice &>(device);
};

ErrorOr<const CpuDevice &> CpuDevice::get(int id) {
  if (id != 0) {
    RAISE_ERROR_MESSAGE("CPU device only supports id == 0.");
  }

  static CpuDevice *device = new CpuDevice();
  return *device;
}

// ===== CpuDeviceStream ===== //
CpuDeviceStream::CpuDeviceStream(const CpuDevice &device)
    : DeviceStream(device) {}

ErrorOrSuccess CpuDeviceStream::synchronize() const { return success(); }

ErrorOrSuccess
CpuDeviceStream::enqueueWaitForEvent(const DeviceEvent &event) const {
  RAISE_ERROR_MESSAGE("Unimplemented");
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
    : DeviceContext(device, std::make_unique<BufferCache>(
                                *this, 0, 0, device.allocationAlignment())) {}

ErrorOrSuccess CpuDeviceContext::memAlloc(void **ptr, size_t byteSize,
                                          AllocMode mode) const {
  assert(byteSize != 0);
  *ptr = TCMallocGlobals::tc_new(byteSize, 1);
  if (*ptr == nullptr) {
    RAISE_ERROR();
  }
  return success();
}

ErrorOrSuccess CpuDeviceContext::memFree(void *ptr, bool isHost) const {
  TCMallocGlobals::tc_delete(ptr);
  return success();
}

ErrorOrSuccess CpuDeviceContext::copyToDevice(const DeviceBuffer &dst,
                                              const void *src,
                                              SyncMode mode) const {
  CHECK_IS_CPU_ERR(dst.context().device());
  void *dstPtr = dst.ptr();
  memcpy(dstPtr, src, dst.byteSize());
  return success();
}

ErrorOrSuccess CpuDeviceContext::copyFromDevice(void *dst,
                                                const DeviceBuffer &src,
                                                SyncMode mode) const {
  CHECK_IS_CPU_ERR(src.context().device());
  void *srcPtr = src.ptr();
  memcpy(dst, srcPtr, src.byteSize());
  return success();
}

ErrorOrSuccess CpuDeviceContext::copyDeviceToDevice(const DeviceBuffer &dst,
                                                    const DeviceBuffer &src,
                                                    SyncMode mode) const {
  CHECK_IS_CPU_ERR(src.context().device());
  CHECK_IS_CPU_ERR(dst.context().device());
  void *srcPtr = src.ptr();
  void *dstPtr = dst.ptr();
  if (src.byteSize() < dst.byteSize()) {
    RAISE_ERROR_MESSAGE("not enough data in src");
  }
  memcpy(dstPtr, srcPtr, dst.byteSize());
  return success();
}

ErrorOrSuccess CpuDeviceContext::setMemory(const DeviceBuffer &dst,
                                           uint32_t val, size_t valSize,
                                           SyncMode mode) const {
  CHECK_IS_CPU_ERR(dst.context().device());
  void *ptr = dst.ptr();
  size_t byteSize = dst.byteSize();
  if (valSize == 1) {
    memset(ptr, static_cast<uint8_t>(val), byteSize);
  } else if (valSize == 2) {
    if ((byteSize & 1) != 0) {
      return Error("byteSize of buffer is misaliged");
    }
    size_t len = byteSize >> 1;
    uint16_t *curPtr = static_cast<uint16_t *>(ptr);
    for (size_t i = 0; i < len; i++) {
      *curPtr++ = static_cast<uint16_t>(val);
    }
  } else {
    if (valSize != 4) {
      return Error("unsupported valSize");
    }
    if ((byteSize & 3) != 0) {
      return Error("byteSize of buffer is misaliged");
    }
    size_t len = byteSize >> 2;
    uint32_t *curPtr = static_cast<uint32_t *>(ptr);
    for (size_t i = 0; i < len; i++) {
      *curPtr++ = val;
    }
  }
  return success();
}

ErrorOr<DeviceFunctionRef> CpuDeviceContext::loadFunction(
    StringRef moduleName, StringRef functionName,
    const DeviceFunction::SignatureBuilder &signature, const void *data,
    int32_t maxDynamicSharedBytes, StringRef debugLevel,
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
  return CpuDeviceStreamRef::create(CpuDevice::cast(this->device()));
}

ErrorOrSuccess CpuDeviceContext::synchronize() const {
  // Currently we do not schedule asynchronous work.
  return success();
}

ErrorOr<DeviceEventRef> CpuDeviceContext::enqueueEvent() const {
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOrSuccess
CpuDeviceContext::enqueueWaitForEvent(const DeviceEvent &event) const {
  RAISE_ERROR_MESSAGE("Unimplemented");
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

// The unit for the _SC_PAGE_SIZE is bytes. The unit for the ru_maxrss is
// kilobytes on Linux and bytes on MacOS.
#ifdef MODULAR_LINUX
  const size_t maxrssMultiple = 1024;
#else // macOS
  const size_t maxrssMultiple = 1;
#endif

  // This is the maximum resident set size used.
  // Current memory use may be lower, this is the "high water mark".
  size_t used = usage.ru_maxrss * maxrssMultiple;
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
