//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Metal device context.
//
//===----------------------------------------------------------------------===//

#include "MetalDeviceContext.h"
#ifdef MODULAR_MACOS

#include "AsyncRT/Runtime/Globals/Globals.h"
#include "MetalImpl.hpp"
#include "Support/AlignedAlloc.h"
#include "Support/ErrorOr.h"

#include "llvm/ADT/Twine.h"
#include <chrono>
#include <cstdint>
#include <cstring>
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

namespace M::AsyncRT {

//
//
// ===== MetalDevice ===== //
//
MetalDevice::MetalDevice(int id, MTL::Device *device, std::string name,
                         Device::MemInfo memInfo, size_t allocAlignment)
    : Device(api, name, id, memInfo, allocAlignment), device_(device),
      memInfo_(memInfo), driverVersion_("0.0.0") {}

MetalDevice::~MetalDevice() {
  if (device_) {
    device_->release();
    device_ = nullptr;
  }
}

ErrorOr<int> MetalDevice::getApiVersion() const { return 0; };

ErrorOr<DriverVersion &> MetalDevice::getDriverVersion() const {
  return driverVersion_;
}

static ErrorOr<size_t> getSystemValue(int name) {
  errno = 0;
  auto value = sysconf(name);
  if (errno != 0)
    RAISE_ERROR_MESSAGE(strerror(errno));
  return static_cast<size_t>(value);
}

ErrorOr<Device::MemInfo> MetalDevice::getMemInfo() {
  Device::MemInfo info;
  auto numPages = getSystemValue(_SC_PHYS_PAGES);
  if (numPages.isError())
    return numPages.takeError();

  auto pageSize = getSystemValue(_SC_PAGE_SIZE);
  if (pageSize.isError())
    return pageSize.takeError();

  auto pages = numPages.takeValue();
  auto size = pageSize.takeValue();

  rusage usage;
  if (-1 == getrusage(RUSAGE_SELF, &usage))
    RAISE_ERROR_MESSAGE("Failed to get current memory usage via rusage");

#ifdef MODULAR_LINUX
  const size_t maxrssMultiple = 1024;
#else
  const size_t maxrssMultiple = 1;
#endif

  size_t used = usage.ru_maxrss * maxrssMultiple;

  info.total = pages * size;
  info.free = info.total > used ? info.total - used : 0;
  return info;
}

ErrorOr<Device::MemInfo> MetalDevice::getMemoryInfo() const {
  return getMemInfo();
}

ErrorOrSuccess MetalDevice::isCompatible() const { return success(); }

ErrorOr<int> MetalDevice::computeCapability() const {
  // TODO(amir): this is a placeholder value. We need to get the actual compute
  return 30;
}

ErrorOr<bool> MetalDevice::canAccess(const Device &peer) const { return false; }

ErrorOr<int> MetalDevice::getAttribute(Attribute attr) const {
  switch (attr) {
  case Device::Attribute::COMPUTE_CAPABILITY_MAJOR: {
    std::string deviceName;
    if (device_ && device_->name()) {
      deviceName = device_->name()->cString(NS::UTF8StringEncoding);
    }

    std::string nameToCheck;
    if (!deviceName.empty()) {
      nameToCheck = deviceName;
    } else {
      return 3;
    }

    if (nameToCheck.find("M4") != std::string::npos ||
        nameToCheck.find("m4") != std::string::npos) {
      return 4;
    }

    if (nameToCheck.find("M3") != std::string::npos ||
        nameToCheck.find("m3") != std::string::npos) {
      return 3;
    }

    if (nameToCheck.find("M2") != std::string::npos ||
        nameToCheck.find("m2") != std::string::npos) {
      return 2;
    }

    if (nameToCheck.find("M1") != std::string::npos ||
        nameToCheck.find("m1") != std::string::npos) {
      return 1;
    }

    return Error("Unknown device name");
  }
  case Device::Attribute::COMPUTE_CAPABILITY_MINOR:
    return 0;
  default:
    return 0;
  }
}

int MetalDevice::deviceCount() const { return 1; }

ErrorOrSuccess MetalDevice::chunkAlloc(void **ptr, size_t byteSize,
                                       bool isHost) const {
  return Error("Metal chunkAlloc not implemented");
}

ErrorOrSuccess MetalDevice::chunkFree(void *ptr, bool isHost) const {
  return Error("Metal chunkFree not implemented");
}

ErrorOr<void *> MetalDevice::getCurrentDriverContext() const {
  RAISE_ERROR_MESSAGE("Metal not supported");
}

ErrorOrSuccess MetalDevice::setCurrentDriverContext(void *ctx) const {
  RAISE_ERROR_MESSAGE("Metal not supported");
}

ErrorOr<DeviceContextRef> MetalDevice::createContext() const {
  MTL::CommandQueue *commandQueue = device_->newCommandQueue();
  if (commandQueue == nullptr)
    RAISE_ERROR_MESSAGE("Failed to create Metal command queue for context.");

  return MetalDeviceContextRef::create(*this, commandQueue);
}

ErrorOr<DeviceStreamRef>
MetalDevice::createExternalStream(void *externalStream) const {
  RAISE_ERROR_MESSAGE("Metal not supported");
}

const MetalDevice &MetalDevice::cast(const Device &device) {
  return static_cast<const MetalDevice &>(device);
}

ErrorOr<const MetalDevice &> MetalDevice::get(int id) {
  if (id != 0)
    RAISE_ERROR_MESSAGE("Apple Metal device only supports id == 0.");

  static RCRef<MetalDevice> metalDeviceInstance;

  if (!metalDeviceInstance) {
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    if (device == nullptr)
      RAISE_ERROR_MESSAGE("Failed to create Metal device.");

    std::string name = device->name()->cString(NS::UTF8StringEncoding);

    Device::MemInfo memInfo;
    bool useBufferCache = false;
    if (useBufferCache) {
      ErrorOr<Device::MemInfo> memInfoOrError = getMemInfo();
      if (memInfoOrError.isError())
        RAISE_ERROR_MESSAGE("Failed to get memory info from system.");
      memInfo = memInfoOrError.takeValue();
    }

    size_t allocAlignment = 64;

    metalDeviceInstance =
        RCRef<MetalDevice>::create(id, device, name, memInfo, allocAlignment);
  }

  return *metalDeviceInstance;
}

ErrorOr<int> MetalDevice::numberOfGPUs() {
  MTL::Device *device = MTL::CreateSystemDefaultDevice();
  if (device == nullptr) {
    return 0;
  }

  device->release();
  return 1;
}

//
//
// ===== MetalDeviceContext ===== //
//

MetalDeviceContext::MetalDeviceContext(const MetalDevice &device,
                                       MTL::CommandQueue *commandQueue)
    : DeviceContext(device), commandQueue_(commandQueue) {
  if (commandQueue_)
    commandQueue_->retain();
}

MetalDeviceContext::~MetalDeviceContext() {
  if (commandQueue_) {
    commandQueue_->release();
    commandQueue_ = nullptr;
  }
}

ErrorOrSuccess MetalDeviceContext::enqueueCopyToDevice(const DeviceBuffer &dst,
                                                       const void *src) const {
  return Error("Metal enqueueCopyToDevice not implemented");
}

ErrorOrSuccess
MetalDeviceContext::enqueueCopyFromDevice(void *dst,
                                          const DeviceBuffer &src) const {
  return Error("Metal enqueueCopyFromDevice not implemented");
}

ErrorOrSuccess
MetalDeviceContext::enqueueCopyFromDeviceSized(void *dst, void *src,
                                               size_t byteSize) const {
  return Error("Metal enqueueCopyFromDeviceSized not implemented");
}

ErrorOrSuccess MetalDeviceContext::memCopyDtoD(const DeviceBuffer &dst,
                                               const DeviceBuffer &src) const {
  return Error("Metal memCopyDtoD not implemented");
}

ErrorOrSuccess MetalDeviceContext::enqueueSetMemory(const DeviceBuffer &dst,
                                                    uint64_t val,
                                                    size_t valSize) const {
  return Error("Metal enqueueSetMemory not implemented");
}

ErrorOr<DeviceFunctionRef> MetalDeviceContext::loadFunction(
    StringRef moduleName, StringRef functionName,
    const DeviceFunction::SignatureBuilder &signature, StringRef code,
    int32_t maxDynamicSharedBytes, StringRef debugLevel,
    int32_t optimizationLevel) const {
  return Error("Metal loadFunction not implemented");
}

ErrorOrSuccess MetalDeviceContext::enqueueFunctionExec(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    const DeviceFunction::Arguments &args) const {
  return Error("Metal enqueueFunctionExec not implemented");
}

ErrorOrSuccess MetalDeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args) const {
  return Error("Metal enqueueFunctionExecDirect not implemented");
}

ErrorOr<DeviceStreamRef> MetalDeviceContext::stream() const {
  return Error("Metal stream not implemented");
}

ErrorOrSuccess MetalDeviceContext::synchronize() const {
  return Error("Metal synchronize not implemented");
}

ErrorOr<DeviceEventRef> MetalDeviceContext::enqueueEvent() const {
  return Error("Metal enqueueEvent not implemented");
}

ErrorOrSuccess
MetalDeviceContext::enqueueWaitFor(const DeviceEvent &event) const {
  return Error("Metal enqueueWaitFor(event) not implemented");
}

ErrorOrSuccess
MetalDeviceContext::enqueueWaitFor(const DeviceContext &ctx) const {
  return Error("Metal enqueueWaitFor(ctx) not implemented");
}

ErrorOr<DeviceTimerRef> MetalDeviceContext::startTimer() const {
  return Error("Metal startTimer not implemented");
}

ErrorOr<std::chrono::nanoseconds>
MetalDeviceContext::stopTimer(const DeviceTimer &timer) const {
  return Error("Metal stopTimer not implemented");
}

ErrorOrSuccess
MetalDeviceContext::enablePeerAccess(const DeviceContext &peer) const {
  return Error("Metal enablePeerAccess not implemented");
}

ErrorOr<Device::MemInfo> MetalDeviceContext::getMemoryInfo() const {
  return Error("Metal getMemoryInfo not implemented");
}

ErrorOrSuccess MetalDeviceContext::memAlloc(void **ptr, size_t byteSize,
                                            Device::AllocMode mode) const {
  return Error("Metal memAlloc not implemented");
}

ErrorOrSuccess MetalDeviceContext::memFree(void *ptr, bool isHost) const {
  return Error("Metal memFree not implemented");
}

ErrorOr<std::tuple<void *, size_t,
                   std::optional<llvm::sys::fs::mapped_file_region>>>
MetalDeviceContext::memmapFileImpl(
    const std::filesystem::path &filename) const {
  return Error("Metal memmapFileImpl not implemented");
}

void *MetalDeviceContext::driverContext() const {
  return const_cast<MetalDeviceContext *>(this);
}

ErrorOrSuccess MetalDeviceContext::enqueueMetalCopy(MTL::Buffer *src,
                                                    MTL::Buffer *dst,
                                                    size_t size) const {
  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr) {
    return Error("Failed to create command buffer");
  }

  MTL::BlitCommandEncoder *blit = cmdBuffer->blitCommandEncoder();
  if (blit == nullptr) {
    cmdBuffer->release();
    return Error("Failed to create blit encoder");
  }

  blit->copyFromBuffer(src, 0, dst, 0, size);
  blit->endEncoding();

  cmdBuffer->commit();
  cmdBuffer->release();
  return ErrorOrSuccess();
}

} // namespace M::AsyncRT

#endif // MODULAR_MACOS
