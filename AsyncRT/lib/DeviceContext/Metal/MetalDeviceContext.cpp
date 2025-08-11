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
  // TODO(Amir): this is a placeholder value. We need to get the actual compute
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
  assert(byteSize != 0);

  MTL::Device *metalDev = metalDevice();
  if (metalDev == nullptr)
    return Error("Invalid Metal device: null pointer");

  // Create Metal buffer with shared storage mode for all allocations
  MTL::ResourceOptions options = MTL::ResourceStorageModeShared;

  MTL::Buffer *buf = metalDev->newBuffer(byteSize, options);
  if (buf == nullptr)
    return Error("Failed to create Metal buffer in chunkAlloc");
  buf->retain();

  // For both host and device allocations, return the contents pointer
  // The buffer cache will manage the memory layout
  void *contents = buf->contents();
  if (contents == nullptr) {
    buf->release();
    return Error("Failed to get contents from Metal buffer");
  }

  *ptr = contents;

  // Track this allocation
  trackBuffer(contents, buf);

  return success();
}

ErrorOrSuccess MetalDevice::chunkFree(void *ptr, bool isHost) const {
  if (ptr == nullptr)
    return success();

  // Look up the Metal buffer for this pointer
  MTL::Buffer *bufferToRelease = getTrackedBuffer(ptr);

  if (bufferToRelease == nullptr) {
    // This shouldn't happen if all allocations go through chunkAlloc
    return Error("Attempt to free untracked Metal buffer in chunkFree");
  }

  // Release the buffer and remove from tracking
  bufferToRelease->release();
  untrackBuffer(ptr);

  return success();
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
// Buffer tracking implementation
//
void MetalDevice::trackBuffer(void *ptr, MTL::Buffer *buffer) const {
  std::lock_guard<std::mutex> lock(bufferMapMutex_);
  ptrToBuffer_[ptr] = buffer;
}

void MetalDevice::untrackBuffer(void *ptr) const {
  std::lock_guard<std::mutex> lock(bufferMapMutex_);
  ptrToBuffer_.erase(ptr);
}

MTL::Buffer *MetalDevice::getTrackedBuffer(const void *ptr) const {
  std::lock_guard<std::mutex> lock(bufferMapMutex_);
  auto it = ptrToBuffer_.find(const_cast<void *>(ptr));
  return (it != ptrToBuffer_.end()) ? it->second : nullptr;
}

bool MetalDevice::isTrackedBuffer(const void *ptr) const {
  std::lock_guard<std::mutex> lock(bufferMapMutex_);
  return ptrToBuffer_.find(const_cast<void *>(ptr)) != ptrToBuffer_.end();
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
  // Validate source
  if (src == nullptr)
    RAISE_ERROR_MESSAGE("Source pointer is null");

  size_t byteSize = dst.byteSize();
  if (byteSize == 0)
    return success();

  // For host destination buffers, we can do a direct memcpy
  if (dst.isHost()) {
    // Host buffer - do direct memcpy
    void *dstPtr = dst.ptr();

    // create a empty command buffer
    MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
    if (cmdBuffer == nullptr)
      return Error("Failed to create command buffer");

    // add completion handler to the command buffer

    cmdBuffer->addCompletedHandler(^(MTL::CommandBuffer *buffer) {
      std::memcpy(dstPtr, src, byteSize);
    });

    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();
    cmdBuffer->release();

    return success();
  }

  // Get destination Metal buffer for device buffers
  MTL::Buffer *dstBuf = getMetalBuffer(dst.ptr());
  if (dstBuf == nullptr)
    RAISE_ERROR_MESSAGE("Invalid Metal buffer pointer");

  NS::UInteger bufferLength = dstBuf->length();
  if (bufferLength == 0)
    RAISE_ERROR_MESSAGE("Metal buffer length is 0 - buffer may be invalid");

  if (byteSize > bufferLength)
    RAISE_ERROR_MESSAGE("Copy size exceeds Metal buffer length");

  // Check if source is a Metal buffer allocated by our context
  MTL::Buffer *srcBuf = getMetalBuffer(src);
  if (srcBuf) {
    // Source is a Metal buffer - use blit encoder for copy
    return enqueueMetalCopy(srcBuf, dstBuf, byteSize);
  }

  // Source is regular host memory - use memcpy approach

  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr)
    return Error("Failed to create command buffer");

  // For memcpy operations, we need to wait for completion even though we commit
  // to queue

  __block size_t capturedByteSize = byteSize;

  // Add completion handler to perform the memcpy when GPU is done
  cmdBuffer->addCompletedHandler(^(MTL::CommandBuffer *buffer) {
    void *dstPtr = dstBuf->contents();
    if (dstPtr) {
      std::memcpy(dstPtr, src, capturedByteSize);

      // If Managed mode, mark the range as modified
    }
  });

  // Commit to queue but wait for memcpy completion
  cmdBuffer->commit();
  cmdBuffer->waitUntilCompleted();
  cmdBuffer->release();

  return success();
}

ErrorOrSuccess
MetalDeviceContext::enqueueCopyFromDevice(void *dst,
                                          const DeviceBuffer &src) const {
  if (dst == nullptr)
    RAISE_ERROR_MESSAGE("Destination pointer cannot be null");
  if (src.isHost())
    RAISE_ERROR_MESSAGE("Source buffer must be a device buffer");
  if (src.context().device().api() != Device::metalAPI)
    RAISE_ERROR_MESSAGE("Source buffer must be a Metal buffer");

  size_t byteSize = src.byteSize();
  if (byteSize == 0)
    return success();

  // Get source Metal buffer
  MTL::Buffer *srcBuf = getMetalBuffer(src.ptr());
  if (srcBuf == nullptr)
    RAISE_ERROR_MESSAGE("Source buffer is null");

  // When we have 2 Device buffers we can do an async copy via
  // enqueueMetalCopy() similar to what the CUDA implementation does
  // However, when we are using raw pointers we need memcpy and wait for
  // completion otherwise, the next command buffer will be executed before the
  // current one is completed

  // Check if destination is a Metal buffer allocated by our context
  MTL::Buffer *dstBuf = getMetalBuffer(dst);
  if (dstBuf) {
    // Destination is a Metal buffer - use blit encoder for copy
    return enqueueMetalCopy(srcBuf, dstBuf, byteSize);
  }

  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr)
    return Error("Failed to create command buffer");

  // For memcpy operations, we need to wait for completion even though we commit
  // to queue Add completion handler to perform the memcpy when GPU is done
  cmdBuffer->addCompletedHandler(^(MTL::CommandBuffer *buffer) {
    void *srcPtr = srcBuf->contents();
    if (srcPtr) {
      std::memcpy(dst, srcPtr, byteSize);
    }
  });

  // Commit to queue but wait for memcpy completion
  // When doing cpu memcopy, we need to wait for the command buffer to complete
  // before committing the next command buffer otherwise, the next command
  // buffer will be executed before the current one is completed
  cmdBuffer->commit();
  cmdBuffer->waitUntilCompleted();
  cmdBuffer->release();

  return success();
}

ErrorOrSuccess
MetalDeviceContext::enqueueCopyFromDeviceSized(void *dst, void *src,
                                               size_t byteSize) const {
  return Error("Unimplemented for Metal devices");
}

ErrorOrSuccess MetalDeviceContext::memCopyDtoD(const DeviceBuffer &dst,
                                               const DeviceBuffer &src) const {
  // 1) Basic validation
  if (src.byteSize() == 0 || dst.byteSize() == 0)
    return success(); // no-op for zero-size

  if (dst.byteSize() < src.byteSize())
    RAISE_ERROR_MESSAGE(
        "Destination buffer size must be >= source buffer size");

  // 2) Handle Metal buffer to Metal buffer copies (both host and device are
  // Metal buffers now)
  MTL::Buffer *srcBuf = getMetalBuffer(src.ptr());
  MTL::Buffer *dstBuf = getMetalBuffer(dst.ptr());

  if (srcBuf == nullptr || dstBuf == nullptr)
    RAISE_ERROR_MESSAGE("Invalid Metal buffer pointer");

  // 3) Delegate to enqueueMetalCopy for all Metal-to-Metal copies
  return enqueueMetalCopy(srcBuf, dstBuf, dst.byteSize());
}

ErrorOrSuccess MetalDeviceContext::enqueueSetMemory(const DeviceBuffer &dst,
                                                    uint64_t val,
                                                    size_t valSize) const {
  // 1) Validation
  if (dst.context().device().api() != Device::metalAPI)
    RAISE_ERROR_MESSAGE("Destination buffer must be a Metal buffer");

  size_t byteSize = dst.byteSize();
  if (byteSize == 0)
    return success(); // no-op

  // Get the Metal buffer for the destination
  MTL::Buffer *dstBuf = getMetalBuffer(dst.ptr());
  if (dstBuf == nullptr)
    RAISE_ERROR_MESSAGE("Destination buffer is null");

  void *dstPtr = dstBuf->contents();
  if (dstPtr == nullptr)
    RAISE_ERROR_MESSAGE("Shared buffer contents() returned null");

  size_t count = byteSize / valSize;
  if (reinterpret_cast<uintptr_t>(dstPtr) % valSize != 0)
    return Error("destination buffer not properly aligned");

  // Push  an empty command buffer to the queue
  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr)
    return Error("Failed to create command buffer");

  // add completion handler to the command buffer
  cmdBuffer->addCompletedHandler(^(MTL::CommandBuffer *buffer) {
    // do nothing

    switch (valSize) {
    case 1:
      std::memset(dstPtr, static_cast<uint8_t>(val), byteSize);
      break;
    case 2: {
      uint16_t v16 = static_cast<uint16_t>(val);
      auto p16 = static_cast<uint16_t *>(dstPtr);
      for (size_t i = 0; i < count; ++i)
        p16[i] = v16;
      break;
    }
    case 4: {
      uint32_t v32 = static_cast<uint32_t>(val);
      auto p32 = static_cast<uint32_t *>(dstPtr);
      for (size_t i = 0; i < count; ++i)
        p32[i] = v32;
      break;
    }
    case 8: {
      auto p64 = static_cast<uint64_t *>(dstPtr);
      for (size_t i = 0; i < count; ++i)
        p64[i] = val;
      break;
    }
    }
  });

  // commit the command buffer
  cmdBuffer->commit();
  // When doing cpu memcopy, we need to wait for the command buffer to complete
  // before committing the next command buffer otherwise, the next command
  // buffer will be executed before the current one is completed
  cmdBuffer->waitUntilCompleted();

  return success();
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
  // Check for any pending errors first, similar to CUDA's behavior
  ErrorOrSuccess pendingErr = checkPendingWork();
  if (pendingErr.isError()) {
    return pendingErr.takeError();
  }

  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr)
    return Error("Failed to create command buffer for sync");

  // Submit an empty command buffer and wait for it to complete
  // This acts as a fence, ensuring all previously submitted work completes
  cmdBuffer->commit();
  cmdBuffer->waitUntilCompleted();
  cmdBuffer->release();

  return success();
}

ErrorOr<DeviceEventRef> MetalDeviceContext::enqueueEvent() const {
  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr)
    return Error("Failed to create command buffer for event");

  // Commit the command buffer - this creates the event
  cmdBuffer->commit();

  return MetalDeviceEventRef::create(MetalDevice::cast(device()), cmdBuffer);
}

ErrorOrSuccess
MetalDeviceContext::enqueueWaitFor(const DeviceEvent &event) const {

  // This is a synchronous implementation of the enqueueWaitFor().
  // Cast to MetalDeviceEvent to get the command buffer
  const MetalDeviceEvent &metalEvent = MetalDeviceEvent::cast(event);

  // For Metal, we need to wait for the event's command buffer to complete
  // before continuing with subsequent operations on our command queue
  ErrorOrSuccess syncResult = metalEvent.synchronize();
  if (syncResult.isError()) {
    return syncResult.takeError();
  }

  return success();
}

ErrorOrSuccess
MetalDeviceContext::enqueueWaitFor(const DeviceContext &ctx) const {
  // For context-based waiting, we synchronize with the other context
  // This ensures all operations on that context complete before we continue
  // This is a synchronous implementation of the enqueueWaitFor().
  return ctx.synchronize();
}

ErrorOr<DeviceTimerRef> MetalDeviceContext::startTimer() const {
  return MetalDeviceTimerRef::create(*this, std::chrono::steady_clock::now());
}

ErrorOr<std::chrono::nanoseconds>
MetalDeviceContext::stopTimer(const DeviceTimer &timer) const {
  return std::chrono::steady_clock::now() -
         MetalDeviceTimer::cast(timer).cpu_start();
}

ErrorOrSuccess
MetalDeviceContext::enablePeerAccess(const DeviceContext &peer) const {
  return Error("Metal enablePeerAccess not implemented");
}

ErrorOr<Device::MemInfo> MetalDeviceContext::getMemoryInfo() const {

  ErrorOrSuccess syncResult = synchronize();
  if (syncResult.isError()) {
    return syncResult.takeError();
  }

  return device().getMemoryInfo();
}

ErrorOrSuccess MetalDeviceContext::memAlloc(void **ptr, size_t byteSize,
                                            Device::AllocMode mode) const {
  if (ptr == nullptr)
    return Error("Invalid allocation parameters: ptr is null");
  if (byteSize == 0) {
    *ptr = nullptr;
    return success();
  }

  MTL::Device *metalDev = MetalDevice::cast(device()).metalDevice();
  if (metalDev == nullptr)
    return Error("Invalid Metal device: null pointer");

  // create the Metal buffer
  MTL::ResourceOptions options = getResourceOptions(mode);
  MTL::Buffer *buf = metalDev->newBuffer(byteSize, options);
  if (buf == nullptr)
    return Error("Failed to create Metal buffer");
  buf->retain();

  // Based on mode of allocation, return the appropriate pointer
  switch (mode) {
  case Device::AllocMode::HOST: {
    // For host buffers, return the contents pointer
    void *contents = buf->contents();
    if (contents == nullptr) {
      buf->release();
      return Error("Failed to get contents from Metal buffer");
    }
    // TODO: Update ptr() to return from MTL::Buffer
    *ptr = contents;

    // Track this allocation at device level
    MetalDevice::cast(device()).trackBuffer(contents, buf);
    return success();
  }
  case Device::AllocMode::SYNC:
  case Device::AllocMode::ASYNC: {
    // For device buffers, return the GPU address
    uint64_t gpuAddress = buf->gpuAddress();
    if (gpuAddress == 0) {
      buf->release();
      return Error("Failed to get GPU address from Metal buffer");
    }
    *ptr = reinterpret_cast<void *>(gpuAddress);

    // Track this allocation at device level
    // TODO: Remove tracking of device buffers and add a new field to the
    // deviceBuffer class to track the MTL::buffer
    MetalDevice::cast(device()).trackBuffer(
        reinterpret_cast<void *>(gpuAddress), buf);
    return success();
  }
  }
}

ErrorOrSuccess MetalDeviceContext::memFree(void *ptr, bool isHost) const {
  if (ptr == nullptr)
    return success();

  // Check if this pointer was allocated by us
  MTL::Buffer *bufferToRelease = getMetalBuffer(ptr);

  if (bufferToRelease == nullptr) {
    // Not a Metal buffer allocated by us - this is an error
    return Error("Attempt to free non-Metal memory - operation failed");
  }

  // Release the buffer and remove from tracking
  bufferToRelease->release();
  MetalDevice::cast(device()).untrackBuffer(ptr);

  return success();
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
  return success();
}

//
// Helper methods for buffer management
//
bool MetalDeviceContext::isMetalBuffer(const void *ptr) const {
  return MetalDevice::cast(device()).isTrackedBuffer(ptr);
}

MTL::Buffer *MetalDeviceContext::getMetalBuffer(const void *ptr) const {
  return MetalDevice::cast(device()).getTrackedBuffer(ptr);
}

MTL::StorageMode
MetalDeviceContext::getStorageMode(Device::AllocMode mode) const {
  // Always use Shared storage mode for all allocations
  // Todo(Amir): Make this configurable in the future via an Environment
  // variable
  return MTL::StorageModeShared;
}

MTL::ResourceOptions
MetalDeviceContext::getResourceOptions(Device::AllocMode mode) const {
  // Pick the storage mode
  MTL::StorageMode storage = getStorageMode(mode);
  // We always want write-combined CPU cache for GPU buffers
  MTL::CPUCacheMode cache = MTL::CPUCacheModeWriteCombined;
  // Combine into a ResourceOptions bitmask
  return static_cast<MTL::ResourceOptions>(storage) |
         static_cast<MTL::ResourceOptions>(cache);
}

//
//
// ===== MetalDeviceEvent ===== //
//

MetalDeviceEvent::MetalDeviceEvent(const MetalDevice &device,
                                   MTL::CommandBuffer *buffer)
    : DeviceEvent(device), buffer_(buffer) {
  if (buffer_)
    buffer_->retain();
}

MetalDeviceEvent::~MetalDeviceEvent() {
  if (buffer_) {
    buffer_->release();
    buffer_ = nullptr;
  }
}

ErrorOrSuccess MetalDeviceEvent::synchronize() const {
  if (buffer_ == nullptr)
    return Error("MetalDeviceEvent: null command buffer");

  buffer_->waitUntilCompleted();
  return success();
}

ErrorOr<bool> MetalDeviceEvent::isReady() const {
  if (buffer_ == nullptr)
    return Error("MetalDeviceEvent: null command buffer");

  MTL::CommandBufferStatus status = buffer_->status();
  return status == MTL::CommandBufferStatusCompleted;
}

const MetalDeviceEvent &MetalDeviceEvent::cast(const DeviceEvent &event) {
  return static_cast<const MetalDeviceEvent &>(event);
}

//
//
// ===== MetalDeviceTimer ===== //
//

MetalDeviceTimer::MetalDeviceTimer(
    const MetalDeviceContext &context,
    std::chrono::time_point<std::chrono::steady_clock> start)
    : DeviceTimer(context), start_(start) {}

std::chrono::time_point<std::chrono::steady_clock>
MetalDeviceTimer::cpu_start() const {
  return start_;
}

const MetalDeviceTimer &MetalDeviceTimer::cast(const DeviceTimer &timer) {
  return static_cast<const MetalDeviceTimer &>(timer);
}

} // namespace M::AsyncRT

#endif // MODULAR_MACOS
