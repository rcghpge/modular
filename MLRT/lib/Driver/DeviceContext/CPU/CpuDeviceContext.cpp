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

#include "MLRT/AsyncRT/Runtime/Globals/Globals.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "Support/AlignedAlloc.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/ML/Fill.h"
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

namespace M::Driver {

//
// ===== CpuDevice ===== //
//
// TODO(iposva): Consider using the actual device name.
CpuDevice::CpuDevice(Device::MemInfo memInfo, int64_t deviceMemoryLimit)
    : Device(api, "cpu", 0, memInfo, M::kPreferredMemoryAlignment,
             deviceMemoryLimit) {}

CpuDevice::~CpuDevice() { memoryManager().clear(); }

ErrorOr<int> CpuDevice::getApiVersion() const { return 0; }

ErrorOr<DriverVersion &> CpuDevice::getDriverVersion() const {
  static DriverVersion version("0.0.0");
  return version;
}

ErrorOrSuccess CpuDevice::isCompatible() const { return success(); }

ErrorOr<int> CpuDevice::computeCapability() const {
  // RUNP-494 Remove this, compute capability is a CUDA-specific concept.
  return 0;
}

ErrorOr<bool> CpuDevice::canAccess(const Device &peer) const { return false; }

ErrorOr<int> CpuDevice::getAttribute(Attribute attr) const { return 0; }

ErrorOr<DeviceContextRef> CpuDevice::createContext() const {
  return CpuDeviceContextRef::create(*this);
}

int CpuDevice::deviceCount() const {
  // Only a single CpuDevice is supported.
  return 1;
}

ErrorOrSuccess CpuDevice::chunkAlloc(void **ptr, size_t byteSize,
                                     bool isHost) const {
  assert(byteSize != 0);
  *ptr = M::AsyncRT::TCMallocGlobals::tc_new(1, byteSize);
  return success();
}

ErrorOrSuccess CpuDevice::chunkFree(void *ptr, bool isHost) const {
  M::AsyncRT::TCMallocGlobals::tc_delete(ptr);
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

ErrorOr<Device::MemInfo> CpuDevice::getMemoryInfo() const {
  Device::MemInfo info;
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

ErrorOr<void *> CpuDevice::getCurrentDriverContext() const { return nullptr; }
ErrorOrSuccess CpuDevice::setCurrentDriverContext(void *ctx) const {
  // Not tracking a context for CPU.
  return success();
}

const CpuDevice &CpuDevice::cast(const Device &device) {
  return static_cast<const CpuDevice &>(device);
}

ErrorOr<const CpuDevice &> CpuDevice::get(int id, int64_t deviceMemoryLimit) {
  if (id != 0) {
    RAISE_ERROR_MESSAGE("CPU device only supports id == 0.");
  }

  static CpuDevice *device =
      new CpuDevice(Device::MemInfo(), deviceMemoryLimit);
  return *device;
}

// ===== CpuDeviceStream ===== //
CpuDeviceStream::CpuDeviceStream(const CpuDeviceContext &ctx)
    : DeviceStream(ctx) {}

ErrorOrSuccess CpuDeviceStream::synchronize() const { return success(); }

ErrorOr<DeviceEventRef> CpuDeviceStream::enqueueEvent() const {
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOrSuccess
CpuDeviceStream::enqueueWaitForEvent(const DeviceEvent &event) const {
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOrSuccess CpuDeviceStream::eventRecord(const DeviceEvent &event) const {
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOrSuccess CpuDeviceStream::waitEvent(const DeviceEvent &event,
                                          unsigned int flags) const {
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOrSuccess CpuDeviceStream::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args) const {
  RAISE_ERROR_MESSAGE("Unimplemented");
}

void *CpuDeviceStream::ptr() const { return nullptr; }

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

ErrorOrSuccess CpuDeviceContext::memAlloc(void **ptr, size_t byteSize,
                                          Device::AllocMode mode) const {
  assert(byteSize != 0);
  *ptr = M::AsyncRT::TCMallocGlobals::tc_new(1, byteSize);
  if (*ptr == nullptr) {
    RAISE_ERROR();
  }
  return success();
}

ErrorOrSuccess CpuDeviceContext::memFree(void *ptr, bool isHost) const {
  M::AsyncRT::TCMallocGlobals::tc_delete(ptr);
  return success();
}

void *CpuDeviceContext::driverContext() const { return nullptr; }

ErrorOrSuccess CpuDeviceContext::enqueueCopyToDevice(const DeviceBuffer &dst,
                                                     const void *src) const {
  CHECK_IS_CPU_ERR(dst.context().device());
  void *dstPtr = dst.ptr();
  memcpy(dstPtr, src, dst.byteSize());
  return success();
}

ErrorOrSuccess
CpuDeviceContext::enqueueCopyFromDevice(void *dst,
                                        const DeviceBuffer &src) const {
  CHECK_IS_CPU_ERR(src.context().device());
  void *srcPtr = src.ptr();
  memcpy(dst, srcPtr, src.byteSize());
  return success();
}

ErrorOrSuccess
CpuDeviceContext::enqueueCopyFromDeviceSized(void *dst, void *src,
                                             size_t byteSize) const {
  void *srcPtr = src;
  memcpy(dst, srcPtr, byteSize);
  return success();
}

ErrorOrSuccess
CpuDeviceContext::enqueueCopyToDeviceSized(void *dst, void *src,
                                           size_t byteSize) const {
  void *dstPtr = dst;
  memcpy(dstPtr, src, byteSize);
  return success();
}

ErrorOrSuccess CpuDeviceContext::memCopyDtoD(const DeviceBuffer &dst,
                                             const DeviceBuffer &src) const {
  CHECK_IS_CPU_ERR(src.context().device());
  CHECK_IS_CPU_ERR(dst.context().device());
  void *srcPtr = src.ptr();
  void *dstPtr = dst.ptr();
  assert(src.byteSize() >= dst.byteSize());
  memcpy(dstPtr, srcPtr, dst.byteSize());
  return success();
}

ErrorOrSuccess CpuDeviceContext::enqueueSetMemory(const DeviceBuffer &dst,
                                                  uint64_t val,
                                                  size_t valSize) const {
  CHECK_IS_CPU_ERR(dst.context().device());

  if (valSize > sizeof(uint64_t))
    return Error("unsupported val size");

  size_t byteSize = dst.byteSize();

  if (valSize != 1 && (byteSize & (1 << valSize) - 1))
    return Error("byteSize of buffer is misaligned");

  void *ptr = dst.ptr();
  size_t len = byteSize / valSize;

  switch (valSize) {
  case 1: {
    uint8_t u8Val = static_cast<uint8_t>(val);
    return M::fillHomogeneous(ptr, len, DType::ui8, &u8Val);
  }
  case 2: {
    uint16_t ui16Val = static_cast<uint16_t>(val);
    return M::fillHomogeneous(ptr, len, DType::ui16, &ui16Val);
  }
  case 4: {
    uint32_t ui32Val = static_cast<uint32_t>(val);
    return M::fillHomogeneous(ptr, len, DType::ui32, &ui32Val);
  }
  case 8: {
    uint64_t ui64Val = static_cast<uint64_t>(val);
    return M::fillHomogeneous(ptr, len, DType::ui64, &ui64Val);
  }
  default:
    llvm_unreachable("unsupported val size");
    return Error("unsupported val size");
  }
}

DeviceContext::MMapFileImplResult
CpuDeviceContext::memmapFileImpl(const std::filesystem::path &filename) const {
  auto result = openFile(filename, /*readOnly*/ true);
  if (result)
    return result.takeError();
  auto [fd, status] = *result;

  auto size = status.getSize();

  std::error_code ec;
  llvm::sys::fs::mapped_file_region mappedFile(
      fd, llvm::sys::fs::mapped_file_region::readonly, size, 0, ec);
  if (ec) {
    llvm::sys::fs::closeFile(fd);
    return Error(ec.message());
  }

  // Close the file, mmap will hold a ref to the descriptor.
  llvm::sys::fs::closeFile(fd);

  return std::tuple{
      mappedFile.data(),
      size,
      std::optional{std::move(mappedFile)},
  };
}

ErrorOr<DeviceFunctionRef> CpuDeviceContext::loadFunction(
    StringRef moduleName, StringRef functionName,
    const DeviceFunction::SignatureBuilder &signature, StringRef code,
    int32_t maxDynamicSharedBytes, StringRef debugLevel,
    int32_t optimizationLevel) const {
  // Currently not implemented. Maybe never to be implemented.
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOrSuccess CpuDeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args, const size_t *argSizes) const {
  // Currently not implemented. Maybe never to be implemented.
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOr<DeviceStreamRef> CpuDeviceContext::stream() const {
  return CpuDeviceStreamRef::create(*this);
}

ErrorOr<DeviceStreamRef> CpuDeviceContext::streamCreate(int priority) const {
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOrSuccess
CpuDeviceContext::streamPriorityRange(int *leastPriority,
                                      int *greatestPriority) const {
  *leastPriority = 0;
  *greatestPriority = 0;
  return success();
}

ErrorOr<DeviceStreamRef>
CpuDeviceContext::createExternalStream(void *externalStream) const {
  return CpuDeviceStreamRef::create(*this);
}

ErrorOrSuccess CpuDeviceContext::synchronize() const {
  // Currently we do not schedule asynchronous work.
  return success();
}

ErrorOr<DeviceEventRef> CpuDeviceContext::enqueueEvent() const {
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOr<DeviceEventRef>
CpuDeviceContext::eventCreate(unsigned int flags) const {
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOrSuccess
CpuDeviceContext::enqueueWaitFor(const DeviceEvent &event) const {
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOrSuccess
CpuDeviceContext::enqueueWaitFor(const DeviceContext &ctx) const {
  RAISE_ERROR_MESSAGE("Unimplemented");
}

ErrorOrSuccess CpuDeviceContext::occupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, const DeviceFunction &func, int blockSize,
    size_t dynamicSharedMemSize) const {
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

ErrorOr<Device::MemInfo> CpuDeviceContext::getDeviceMemoryInfo() const {
  return device().getMemoryInfo();
}

const CpuDeviceContext &CpuDeviceContext::cast(const DeviceContext &ctx) {
  return static_cast<const CpuDeviceContext &>(ctx);
}

} // namespace M::Driver
