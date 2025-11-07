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

#include "Support/Buffer.h"
#include "Support/ErrorOr.h"

#include <cassert>
#include <string>

namespace M::AsyncRT {

//===----------------------------------------------------------------------===//
// VirtualDevice
//===----------------------------------------------------------------------===//

VirtualDevice::VirtualDevice(int id, std::string name, const Device *realDevice,
                             int64_t deviceMemoryLimit)
    : Device(realDevice->api(), std::move(name), id, Device::MemInfo(),
             M::kPreferredMemoryAlignment, deviceMemoryLimit),
      realDevice_(realDevice), deviceMemoryLimit_(deviceMemoryLimit) {
  assert(realDevice_ && "VirtualDevice requires a non-null real device");
}

VirtualDevice::~VirtualDevice() = default;

ErrorOr<int> VirtualDevice::getApiVersion() const {
  return realDevice_->getApiVersion();
}

ErrorOr<DriverVersion &> VirtualDevice::getDriverVersion() const {
  return realDevice_->getDriverVersion();
}

ErrorOrSuccess VirtualDevice::isCompatible() const {
  return realDevice_->isCompatible();
}

ErrorOr<int> VirtualDevice::computeCapability() const {
  return realDevice_->computeCapability();
}

ErrorOr<std::string> VirtualDevice::getArchName() const {
  return realDevice_->getArchName();
}

ErrorOr<bool> VirtualDevice::canAccess(const Device &peer) const {
  return realDevice_->canAccess(peer);
}

ErrorOrSuccess VirtualDevice::enablePeerAccess(const Device &peer) const {
  return realDevice_->enablePeerAccess(peer);
}

ErrorOr<int> VirtualDevice::getAttribute(Attribute attr) const {
  return realDevice_->getAttribute(attr);
}

int VirtualDevice::deviceCount() const {
  // Virtual devices don't have real device count
  return 1;
}

ErrorOrSuccess VirtualDevice::chunkAlloc(void **ptr, size_t byteSize,
                                         bool isHost) const {
  assert(false && "VirtualDevice::chunkAlloc not implemented");
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not support memory allocation");
}

ErrorOrSuccess VirtualDevice::chunkFree(void *ptr, bool isHost) const {
  assert(false && "VirtualDevice::chunkFree not implemented");
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not support memory deallocation");
}

ErrorOr<Device::MemInfo> VirtualDevice::getMemoryInfo() const {
  // If a memory limit was explicitly specified, use it
  if (deviceMemoryLimit_ > 0) {
    Device::MemInfo info;
    info.free = deviceMemoryLimit_;
    info.total = deviceMemoryLimit_;
    return info;
  }

  // Otherwise, query the real device's memory
  return realDevice_->getMemoryInfo();
}

ErrorOr<void *> VirtualDevice::getCurrentDriverContext() const {
  assert(false && "VirtualDevice::getCurrentDriverContext not implemented");
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not have a driver context");
}

ErrorOrSuccess VirtualDevice::setCurrentDriverContext(void *ctx) const {
  assert(false && "VirtualDevice::setCurrentDriverContext not implemented");
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not support setting driver context");
}

ErrorOr<DeviceContextRef> VirtualDevice::createContext() const {
  // Create a context from the real device for compilation operations
  ErrorOr<DeviceContextRef> realContextErr = realDevice_->createContext();
  if (realContextErr.isError()) {
    return realContextErr.takeError();
  }

  DeviceContextRef realContext = realContextErr.takeValue();
  return VirtualDeviceContextRef::create(*this, std::move(realContext));
}

ErrorOr<DeviceStreamRef>
VirtualDevice::createExternalStream(void *externalStream) const {
  assert(false && "VirtualDevice::createExternalStream not implemented");
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not support external streams");
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

VirtualDeviceStream::VirtualDeviceStream(const VirtualDevice &device)
    : DeviceStream(device) {}

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
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not support function execution");
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

VirtualDeviceContext::VirtualDeviceContext(const VirtualDevice &device,
                                           DeviceContextRef realContext)
    : DeviceContext(device), realContext_(std::move(realContext)) {
  assert(realContext_ &&
         "VirtualDeviceContext requires a non-null real context");
}

VirtualDeviceContext::~VirtualDeviceContext() = default;

ErrorOrSuccess
VirtualDeviceContext::enqueueCopyToDevice(const DeviceBuffer &dst,
                                          const void *src) const {
  assert(false && "VirtualDeviceContext::enqueueCopyToDevice not implemented");
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not support memory operations");
}

ErrorOrSuccess
VirtualDeviceContext::enqueueCopyFromDevice(void *dst,
                                            const DeviceBuffer &src) const {
  assert(false &&
         "VirtualDeviceContext::enqueueCopyFromDevice not implemented");
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not support memory operations");
}

ErrorOrSuccess
VirtualDeviceContext::enqueueCopyFromDeviceSized(void *dst, void *src,
                                                 size_t byteSize) const {
  assert(false &&
         "VirtualDeviceContext::enqueueCopyFromDeviceSized not implemented");
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not support memory operations");
}

ErrorOrSuccess
VirtualDeviceContext::enqueueCopyToDeviceSized(void *dst, void *src,
                                               size_t byteSize) const {
  assert(false &&
         "VirtualDeviceContext::enqueueCopyToDeviceSized not implemented");
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not support memory operations");
}

ErrorOrSuccess
VirtualDeviceContext::memCopyDtoD(const DeviceBuffer &dst,
                                  const DeviceBuffer &src) const {
  assert(false && "VirtualDeviceContext::memCopyDtoD not implemented");
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not support memory operations");
}

ErrorOrSuccess VirtualDeviceContext::enqueueSetMemory(const DeviceBuffer &dst,
                                                      uint64_t val,
                                                      size_t valSize) const {
  assert(false && "VirtualDeviceContext::enqueueSetMemory not implemented");
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not support memory operations");
}

ErrorOr<M::BufferRef>
VirtualDeviceContext::compileFunction(StringRef ir, StringRef debugLevel,
                                      int32_t optimizationLevel,
                                      StringRef targetAccelerator) const {
  // Delegate compilation to the real device context
  return realContext_->compileFunction(ir, debugLevel, optimizationLevel,
                                       targetAccelerator);
}

ErrorOr<DeviceFunctionRef> VirtualDeviceContext::loadFunction(
    StringRef moduleName, StringRef functionName,
    const DeviceFunction::SignatureBuilder &signature, StringRef code,
    int32_t maxDynamicSharedBytes, StringRef debugLevel,
    int32_t optimizationLevel) const {
  assert(false && "VirtualDeviceContext::loadFunction not implemented");
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not support function loading");
}

ErrorOrSuccess VirtualDeviceContext::enqueueFunctionExec(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    const DeviceFunction::Arguments &args) const {
  assert(false && "VirtualDeviceContext::enqueueFunctionExec not implemented");
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not support function execution");
}

ErrorOrSuccess VirtualDeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args, const size_t *argSizes) const {
  assert(false &&
         "VirtualDeviceContext::enqueueFunctionExecDirect not implemented");
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not support function execution");
}

ErrorOr<DeviceStreamRef> VirtualDeviceContext::stream() const {
  // Return a virtual stream for compile-only mode
  return VirtualDeviceStreamRef::create(VirtualDevice::cast(device()));
}

ErrorOr<DeviceStreamRef>
VirtualDeviceContext::streamCreate(unsigned int flags) const {
  // Return a virtual stream for compile-only mode
  return VirtualDeviceStreamRef::create(VirtualDevice::cast(device()));
}

ErrorOr<DeviceStreamRef>
VirtualDeviceContext::streamCreateWithPriority(unsigned int flags,
                                               int priority) const {
  // Return a virtual stream for compile-only mode (ignore priority)
  return VirtualDeviceStreamRef::create(VirtualDevice::cast(device()));
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

ErrorOrSuccess VirtualDeviceContext::synchronize() const {
  // No-op for virtual devices
  return success();
}

ErrorOr<DeviceEventRef> VirtualDeviceContext::enqueueEvent() const {
  return VirtualDeviceEventRef::create(VirtualDevice::cast(device()));
}

ErrorOr<DeviceEventRef>
VirtualDeviceContext::eventCreate(unsigned int flags) const {
  return VirtualDeviceEventRef::create(VirtualDevice::cast(device()));
}

ErrorOrSuccess
VirtualDeviceContext::enqueueWaitFor(const DeviceEvent &event) const {
  // No-op for virtual devices
  return success();
}

ErrorOrSuccess
VirtualDeviceContext::enqueueWaitFor(const DeviceContext &ctx) const {
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
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not support occupancy queries");
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
  assert(false && "VirtualDeviceContext::memAlloc not implemented");
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not support memory allocation");
}

ErrorOrSuccess VirtualDeviceContext::memFree(void *ptr, bool isHost) const {
  assert(false && "VirtualDeviceContext::memFree not implemented");
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not support memory deallocation");
}

void *VirtualDeviceContext::driverContext() const {
  // Return nullptr for virtual contexts
  return nullptr;
}

DeviceContext::MMapFileImplResult VirtualDeviceContext::memmapFileImpl(
    const std::filesystem::path &filename) const {
  assert(false && "VirtualDeviceContext::memmapFileImpl not implemented");
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "VirtualDevice does not support memory mapping");
}

} // namespace M::AsyncRT
