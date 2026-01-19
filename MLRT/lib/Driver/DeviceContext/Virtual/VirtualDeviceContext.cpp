//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Virtual device context for compile-only mode.
//
//===----------------------------------------------------------------------===//

#include "VirtualDeviceContext.h"

#include "MLRT/Driver/DeviceContext/ErrorMacros.h"
#include "MLRT/lib/Driver/DeviceContext/CUDA/CUDACompilationDevice.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/Profiling/Tracy.h"

#include <cassert>
#include <string>

namespace M::Driver {

//===----------------------------------------------------------------------===//
// VirtualDevice
//===----------------------------------------------------------------------===//

VirtualDevice::VirtualDevice(
    int id, std::string name,
    std::unique_ptr<CompilationDevice> compilationDevice,
    int64_t deviceMemoryLimit)
    : Device(compilationDevice->api(), std::move(name), id, Device::MemInfo(),
             M::kPreferredMemoryAlignment, deviceMemoryLimit),
      compilationDevice_(std::move(compilationDevice)),
      deviceMemoryLimit_(deviceMemoryLimit) {
  assert(compilationDevice_ &&
         "VirtualDevice requires a non-null compilation device");
}

VirtualDevice::~VirtualDevice() = default;

ErrorOr<int> VirtualDevice::getApiVersion() const {
  return compilationDevice_->getApiVersion();
}

ErrorOr<DriverVersion &> VirtualDevice::getDriverVersion() const {
  return compilationDevice_->getDriverVersion();
}

ErrorOrSuccess VirtualDevice::isCompatible() const { return success(); }

ErrorOr<int> VirtualDevice::computeCapability() const {
  RETURN_ERROR("VirtualDevice does not support compute capability");
}

ErrorOr<const char *> VirtualDevice::getArchName() const {
  return compilationDevice_->getArchName();
}

ErrorOr<bool> VirtualDevice::canAccess(const Device &peer) const {
  // Virtual devices don't support peer access
  return false;
}

ErrorOrSuccess VirtualDevice::enablePeerAccess(const Device &peer) const {
  // Virtual devices don't support peer access
  RETURN_ERROR("VirtualDevice does not support peer access");
}

ErrorOr<int> VirtualDevice::getAttribute(Attribute attr) const {
  RETURN_ERROR("VirtualDevice does not support getAttribute");
}

int VirtualDevice::deviceCount() const {
  // Virtual devices don't have real device count
  return 1;
}

ErrorOrSuccess VirtualDevice::chunkAlloc(void **ptr, size_t byteSize,
                                         bool isHost) const {
  assert(false && "VirtualDevice::chunkAlloc not implemented");
  RETURN_ERROR("VirtualDevice does not support memory allocation");
}

ErrorOrSuccess VirtualDevice::chunkFree(void *ptr, bool isHost) const {
  assert(false && "VirtualDevice::chunkFree not implemented");
  RETURN_ERROR("VirtualDevice does not support memory deallocation");
}

ErrorOr<Device::MemInfo> VirtualDevice::getMemoryInfo() const {
  // Return simulated memory info
  Device::MemInfo info;
  if (deviceMemoryLimit_ > 0) {
    // Use explicitly specified limit
    info.free = deviceMemoryLimit_;
    info.total = deviceMemoryLimit_;
  } else {
    // Return a reasonable default (192GB, typical for B200)
    info.free = 192ULL * 1024 * 1024 * 1024;
    info.total = 192ULL * 1024 * 1024 * 1024;
  }
  return info;
}

ErrorOr<void *> VirtualDevice::getCurrentDriverContext() const {
  assert(false && "VirtualDevice::getCurrentDriverContext not implemented");
  RETURN_ERROR("VirtualDevice does not have a driver context");
}

ErrorOrSuccess VirtualDevice::setCurrentDriverContext(void *ctx) const {
  assert(false && "VirtualDevice::setCurrentDriverContext not implemented");
  RETURN_ERROR("VirtualDevice does not support setting driver context");
}

ErrorOr<DeviceContextRef> VirtualDevice::createContext() const {
  // Create a virtual context that uses the compilation device
  return VirtualDeviceContextRef::create(*this);
}

const VirtualDevice &VirtualDevice::cast(const Device &device) {
  // Virtual devices can have any API based on their backing device.
  // The cast is valid as long as the device is actually a VirtualDevice.
  return static_cast<const VirtualDevice &>(device);
}

//===----------------------------------------------------------------------===//
// VirtualDeviceEvent
//===----------------------------------------------------------------------===//

VirtualDeviceEvent::VirtualDeviceEvent(const VirtualDevice &device)
    : DeviceEvent(device) {}

ErrorOrSuccess VirtualDeviceEvent::synchronize() const {
  // No-op for virtual devices
  return success();
}

ErrorOr<bool> VirtualDeviceEvent::isReady() const {
  // Virtual events are always ready
  return true;
}

const VirtualDeviceEvent &VirtualDeviceEvent::cast(const DeviceEvent &event) {
  // Virtual device events can have any API based on their backing device.
  return static_cast<const VirtualDeviceEvent &>(event);
}

//===----------------------------------------------------------------------===//
// VirtualDeviceStream
//===----------------------------------------------------------------------===//

VirtualDeviceStream::VirtualDeviceStream(const VirtualDeviceContext &ctx)
    : DeviceStream(ctx) {}

ErrorOrSuccess VirtualDeviceStream::synchronize() const {
  // No-op for virtual devices
  return success();
}

ErrorOr<DeviceEventRef> VirtualDeviceStream::enqueueEvent() const {
  return VirtualDeviceEventRef::create(VirtualDevice::cast(device()));
}

ErrorOrSuccess
VirtualDeviceStream::eventRecord(const DeviceEvent &event) const {
  // No-op for virtual devices
  return success();
}

ErrorOrSuccess VirtualDeviceStream::waitEvent(const DeviceEvent &event,
                                              unsigned int flags) const {
  // No-op for virtual devices
  return success();
}

ErrorOrSuccess
VirtualDeviceStream::enqueueWaitForEvent(const DeviceEvent &event) const {
  // No-op for virtual devices
  return success();
}

ErrorOrSuccess VirtualDeviceStream::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args) const {
  assert(false &&
         "VirtualDeviceStream::enqueueFunctionExecDirect not implemented");
  RETURN_ERROR("VirtualDevice does not support function execution");
}

void *VirtualDeviceStream::ptr() const {
  // Return nullptr for virtual streams
  return nullptr;
}

const VirtualDeviceStream *
VirtualDeviceStream::verified_cast(const DeviceStream *stream) {
  if (!stream)
    return nullptr;
  // Virtual device streams can have any API based on their backing device.
  return static_cast<const VirtualDeviceStream *>(stream);
}

//===----------------------------------------------------------------------===//
// VirtualDeviceTimer
//===----------------------------------------------------------------------===//

VirtualDeviceTimer::VirtualDeviceTimer(const VirtualDeviceContext &context)
    : DeviceTimer(context) {}

const VirtualDeviceTimer &VirtualDeviceTimer::cast(const DeviceTimer &timer) {
  // Virtual device timers can have any API based on their backing device.
  return static_cast<const VirtualDeviceTimer &>(timer);
}

//===----------------------------------------------------------------------===//
// VirtualDeviceContext
//===----------------------------------------------------------------------===//

VirtualDeviceContext::VirtualDeviceContext(const VirtualDevice &device)
    : DeviceContext(device) {}

VirtualDeviceContext::~VirtualDeviceContext() = default;

ErrorOrSuccess
VirtualDeviceContext::enqueueCopyToDevice(void *dst, const void *src,
                                          size_t byteSize) const {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::enqueueCopyToDevice",
                       TRACY_COLOR_PURPLE);
  assert(false && "VirtualDeviceContext::enqueueCopyToDevice not implemented");
  RETURN_ERROR("VirtualDevice does not support memory operations");
}

ErrorOrSuccess
VirtualDeviceContext::enqueueCopyFromDevice(void *dst, const void *src,
                                            size_t byteSize) const {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::enqueueCopyFromDevice",
                       TRACY_COLOR_PURPLE);
  assert(false &&
         "VirtualDeviceContext::enqueueCopyFromDevice not implemented");
  RETURN_ERROR("VirtualDevice does not support memory operations");
}

ErrorOrSuccess
VirtualDeviceContext::memCopyDtoD(const DeviceBuffer &dst,
                                  const DeviceBuffer &src) const {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::memCopyDtoD", TRACY_COLOR_PURPLE);
  assert(false && "VirtualDeviceContext::memCopyDtoD not implemented");
  RETURN_ERROR("VirtualDevice does not support memory operations");
}

ErrorOrSuccess VirtualDeviceContext::enqueueSetMemory(void *dst,
                                                      size_t byteSize,
                                                      uint64_t val,
                                                      size_t valSize) const {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::enqueueSetMemory",
                       TRACY_COLOR_PURPLE);
  assert(false && "VirtualDeviceContext::enqueueSetMemory not implemented");
  RETURN_ERROR("VirtualDevice does not support memory operations");
}

ErrorOr<DeviceFunctionRef> VirtualDeviceContext::loadFunction(
    StringRef moduleName, StringRef functionName,
    const DeviceFunction::SignatureBuilder &signature, StringRef code,
    int32_t maxDynamicSharedBytes, StringRef debugLevel,
    int32_t optimizationLevel) const {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::loadFunction",
                       TRACY_COLOR_PURPLE);
  assert(false && "VirtualDeviceContext::loadFunction not implemented");
  RETURN_ERROR("VirtualDevice does not support function loading");
}

ErrorOrSuccess VirtualDeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args, const size_t *argSizes) const {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::enqueueFunctionExecDirect",
                       TRACY_COLOR_PURPLE);
  assert(false &&
         "VirtualDeviceContext::enqueueFunctionExecDirect not implemented");
  RETURN_ERROR("VirtualDevice does not support function execution");
}

ErrorOr<DeviceStreamRef> VirtualDeviceContext::stream() const {
  // Return a virtual stream for compile-only mode
  return VirtualDeviceStreamRef::create(*this);
}

ErrorOr<DeviceStreamRef>
VirtualDeviceContext::createStream(int priority) const {
  // Return a virtual stream for compile-only mode (ignore priority)
  return VirtualDeviceStreamRef::create(*this);
}

ErrorOrSuccess
VirtualDeviceContext::streamPriorityRange(int *leastPriority,
                                          int *greatestPriority) const {
  // Virtual devices support a single priority level
  if (leastPriority)
    *leastPriority = 0;
  if (greatestPriority)
    *greatestPriority = 0;
  return success();
}

ErrorOr<DeviceStreamRef>
VirtualDeviceContext::createExternalStream(void *externalStream) const {
  assert(false && "VirtualDeviceContext::createExternalStream not implemented");
  RETURN_ERROR("VirtualDevice does not support external streams");
}

ErrorOrSuccess VirtualDeviceContext::synchronize() const {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::synchronize", TRACY_COLOR_PURPLE);
  // No-op for virtual devices
  return success();
}

ErrorOr<DeviceEventRef> VirtualDeviceContext::enqueueEvent() const {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::enqueueEvent",
                       TRACY_COLOR_PURPLE);
  return VirtualDeviceEventRef::create(VirtualDevice::cast(device()));
}

ErrorOrSuccess
VirtualDeviceContext::enqueueEvent(const DeviceEvent &event) const {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::enqueueEvent",
                       TRACY_COLOR_PURPLE);
  // No-op for virtual devices
  return success();
}

ErrorOr<DeviceEventRef>
VirtualDeviceContext::eventCreate(unsigned int flags) const {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::eventCreate", TRACY_COLOR_PURPLE);
  return VirtualDeviceEventRef::create(VirtualDevice::cast(device()));
}

ErrorOrSuccess
VirtualDeviceContext::enqueueWaitFor(const DeviceEvent &event) const {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::enqueueWaitFor",
                       TRACY_COLOR_PURPLE);
  // No-op for virtual devices
  return success();
}

ErrorOrSuccess
VirtualDeviceContext::enqueueWaitFor(const DeviceContext &ctx) const {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::enqueueWaitFor",
                       TRACY_COLOR_PURPLE);
  // No-op for virtual devices
  return success();
}

ErrorOr<DeviceTimerRef> VirtualDeviceContext::startTimer() const {
  return VirtualDeviceTimerRef::create(*this);
}

ErrorOr<std::chrono::nanoseconds>
VirtualDeviceContext::stopTimer(const DeviceTimer &timer) const {
  // Return 0 duration for virtual timers
  return std::chrono::nanoseconds(0);
}

ErrorOrSuccess
VirtualDeviceContext::enablePeerAccess(const DeviceContext &peer) const {
  // No-op for virtual devices
  return success();
}

ErrorOr<Device::MemInfo> VirtualDeviceContext::getDeviceMemoryInfo() const {
  return VirtualDevice::cast(device()).getMemoryInfo();
}

ErrorOrSuccess VirtualDeviceContext::occupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, const DeviceFunction &func, int blockSize,
    size_t dynamicSharedMemSize) const {
  assert(false &&
         "VirtualDeviceContext::occupancyMaxActiveBlocksPerMultiprocessor "
         "not implemented");
  RETURN_ERROR("VirtualDevice does not support occupancy queries");
}

const VirtualDeviceContext &
VirtualDeviceContext::cast(const DeviceContext &ctx) {
  // Virtual device contexts can have any API based on their backing device.
  return static_cast<const VirtualDeviceContext &>(ctx);
}

const VirtualDeviceContext *
VirtualDeviceContext::verified_cast(const DeviceContext *ctx) {
  if (!ctx)
    return nullptr;
  // Virtual device contexts can have any API based on their backing device.
  return static_cast<const VirtualDeviceContext *>(ctx);
}

ErrorOrSuccess VirtualDeviceContext::memAlloc(void **ptr, size_t byteSize,
                                              Device::AllocMode mode) const {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::memAlloc", TRACY_COLOR_PURPLE);
  assert(false && "VirtualDeviceContext::memAlloc not implemented");
  RETURN_ERROR("VirtualDevice does not support memory allocation");
}

ErrorOrSuccess VirtualDeviceContext::memFree(void *ptr, bool isHost) const {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::memFree", TRACY_COLOR_PURPLE);
  assert(false && "VirtualDeviceContext::memFree not implemented");
  RETURN_ERROR("VirtualDevice does not support memory deallocation");
}

void *VirtualDeviceContext::driverContext() const {
  // Return nullptr for virtual contexts
  return nullptr;
}

DeviceContext::MMapFileImplResult VirtualDeviceContext::memmapFileImpl(
    const std::filesystem::path &filename) const {
  assert(false && "VirtualDeviceContext::memmapFileImpl not implemented");
  RETURN_ERROR("VirtualDevice does not support memory mapping");
}

} // namespace M::Driver
