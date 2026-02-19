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
#include <limits>
#if MODULAR_MACOS

#include "MLRT/AsyncRT/Runtime/Globals/Globals.h"
#include "MLRT/Driver/DeviceContext/ErrorMacros.h"
#include "MetalDriver.h"
#include "Support/AlignedAlloc.h"
#include "Support/ErrorOr.h"
#include "Support/Profiling/Tracy.h"

#include "llvm/ADT/Twine.h"
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <unistd.h>

#define CHECK_IS_METAL_ERR(dev)                                                \
  do {                                                                         \
    if ((dev).api() != Device::metalAPI)                                       \
      RETURN_ERROR("Not a Metal device.");                                     \
  } while (0)

#define METAL_MAYBE_SYNC(ctx)                                                  \
  do {                                                                         \
    const DeviceContext &metal_ctx = (ctx);                                    \
    if (metal_ctx.syncMode()) {                                                \
      RETURN_IF_ERROR(metal_ctx.synchronize());                                \
    }                                                                          \
  } while (0)

namespace M::Driver {

void ensureThreadlocalAutoreleasePoolExists() {
  thread_local NS::SharedPtr<NS::AutoreleasePool> autoreleasePool;
  if (!autoreleasePool) {
    autoreleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  }
}

static ErrorOr<int> getComputeCapability(StringRef name) {
  ensureThreadlocalAutoreleasePoolExists();

  if (name.empty())
    return 3;

  if (name.contains_insensitive("M5"))
    return 5;

  if (name.contains_insensitive("M4"))
    return 4;

  if (name.contains_insensitive("M3"))
    return 3;

  if (name.contains_insensitive("M2"))
    return 2;

  if (name.contains_insensitive("M1"))
    return 1;

  RETURN_ERROR("Unknown device name");
}

//
//
// ===== MetalDevice ===== //
//
MetalDevice::MetalDevice(int id, NS::SharedPtr<MTL::Device> device,
                         std::string name, Device::MemInfo memInfo,
                         size_t allocAlignment)
    : Device(api, name, id, memInfo, allocAlignment), device_(device),
      memInfo_(memInfo), driverVersion_([]() -> std::string {
        // Metal drivers ship with macOS, Darwin release is the driver version
        struct utsname systemInfo;
        uname(&systemInfo);
        return systemInfo.release;
      }()) {}

ErrorOr<int> MetalDevice::getApiVersion() const { return 0; }

ErrorOr<DriverVersion &> MetalDevice::getDriverVersion() const {
  return driverVersion_;
}

static ErrorOr<size_t> getSystemValue(int name) {
  errno = 0;
  long value = sysconf(name);
  if (errno != 0)
    RETURN_ERROR(strerror(errno));
  return static_cast<size_t>(value);
}

ErrorOr<Device::MemInfo> MetalDevice::getMemInfo(MTL::Device *metalDev) {
  Device::MemInfo info;
  DEFINE_OR_RETURN_ERROR(size_t, pages, getSystemValue(_SC_PHYS_PAGES));
  DEFINE_OR_RETURN_ERROR(size_t, size, getSystemValue(_SC_PAGE_SIZE));

  rusage usage;
  if (-1 == getrusage(RUSAGE_SELF, &usage))
    RETURN_ERROR("Failed to get current memory usage via rusage");

#if MODULAR_LINUX
  const size_t maxrssMultiple = 1024;
#else
  const size_t maxrssMultiple = 1;
#endif

  size_t used = usage.ru_maxrss * maxrssMultiple;

  size_t systemTotal = pages * size;

  // On Apple Silicon, Metal has a maxBufferLength limit that restricts the
  // maximum size of a single buffer allocation. This is typically much smaller
  // than total system memory (e.g., ~4GB vs 36GB). We need to report this limit
  // as our effective memory ceiling to prevent buffer allocation failures.
  size_t maxBufferLength = systemTotal;
  if (metalDev != nullptr) {
    maxBufferLength = metalDev->maxBufferLength();
  }

  info.total = std::min(systemTotal, maxBufferLength);
  info.free = info.total > used ? info.total - used : 0;
  return info;
}

ErrorOr<Device::MemInfo> MetalDevice::getMemoryInfo() const {
  return getMemInfo(device_.get());
}

ErrorOrSuccess MetalDevice::isCompatible() const { return success(); }

ErrorOr<int> MetalDevice::computeCapability() const {
  // Use the major compute capability as the compute capability.
  // This allows to get something like apple-m1 as arch name.
  return getAttribute(Device::Attribute::COMPUTE_CAPABILITY_MAJOR);
}

ErrorOr<bool> MetalDevice::canAccess(const Device &peer) const { return false; }

ErrorOr<int> MetalDevice::getAttribute(Attribute attr) const {
  // Constants pulled from the Metal feature set tables available in
  // PDF/Numbers spreadsheet at:
  // https://developer.apple.com/documentation/metal/detecting-gpu-features-and-metal-software-versions
  //
  // The families we care about:
  // - M1 -> Apple7
  // - M2 -> Apple8
  // - M3 -> Apple9
  // - M4 -> Apple9
  // - M5 -> Apple10

#define ATTRIBUTE_UNSUPPORTED(ATTR)                                            \
  RETURN_ERROR("Attribute \"" #ATTR "\" not supported for Apple GPU.")

  switch (attr) {
  // 'Maximum threads per threadgroup' from 'Function arguments' section in the
  // 'GPU implementation limits by family', same limit for Apple7 - Apple10.
  case Device::Attribute::MAX_THREADS_PER_BLOCK: {
    return 1024;
  }

  // Block dimensions map to max threads per threadgroup.
  case Device::Attribute::MAX_BLOCK_DIM_X:
    return device_->maxThreadsPerThreadgroup().width;
  case Device::Attribute::MAX_BLOCK_DIM_Y:
    return device_->maxThreadsPerThreadgroup().height;
  case Device::Attribute::MAX_BLOCK_DIM_Z:
    return device_->maxThreadsPerThreadgroup().depth;

  // There is no limit to the size of a grid.
  case Device::Attribute::MAX_GRID_DIM_X:
    return std::numeric_limits<int>::max();
  case Device::Attribute::MAX_GRID_DIM_Y:
    return std::numeric_limits<int>::max();
  case Device::Attribute::MAX_GRID_DIM_Z:
    return std::numeric_limits<int>::max();

  // 'Maximum total threadgroup memory allocation' from 'Function arguments'
  // section in the 'GPU implementation limits by family', same limit for
  // Apple7 - Apple10.
  case Device::Attribute::MAX_SHARED_MEMORY_PER_BLOCK:
    return 33554432; // 32KB

  // The SIMD-group size is a dynamic property which is accessble through the
  // computePipeline->threadExecutionWidth property, i.e. it's tied to a
  // specific kernel invocation rather available as a static/dynamic device
  // property.
  case Device::Attribute::WARP_SIZE:
    ATTRIBUTE_UNSUPPORTED(WARP_SIZE);

  // Max registers per block is not documented.
  case Device::Attribute::MAX_REGISTERS_PER_BLOCK:
    ATTRIBUTE_UNSUPPORTED(MAX_REGISTERS_PER_BLOCK);

  // Clock rate is exposed through performance counters not as a property of
  // the device.
  case Device::Attribute::CLOCK_RATE:
    ATTRIBUTE_UNSUPPORTED(CLOCK_RATE);

  case Device::Attribute::MULTIPROCESSOR_COUNT: {
// Host is built for MacOS 11.0, but `kIOMainPortDefault` is only avilable on
// MacOS >= 12.0. However, we support Apple GPU only for MacOS >= 15, so this
// warning can be ignored.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability-new"
    io_iterator_t iterator;
    CFMutableDictionaryRef matchDict = IOServiceMatching("AGXAccelerator");

    if (IOServiceGetMatchingServices(kIOMainPortDefault, matchDict,
                                     &iterator) != KERN_SUCCESS)
      ATTRIBUTE_UNSUPPORTED(MULTIPROCESSOR_COUNT);

    io_service_t service;
    while ((service = IOIteratorNext(iterator)) != 0) {
      CFMutableDictionaryRef properties = nullptr;

      if (IORegistryEntryCreateCFProperties(
              service, &properties, kCFAllocatorDefault, 0) == KERN_SUCCESS) {
        CFNumberRef coreCountRef = (CFNumberRef)CFDictionaryGetValue(
            properties, CFSTR("gpu-core-count"));

        if (coreCountRef && CFGetTypeID(coreCountRef) == CFNumberGetTypeID()) {
          int coreCount = 0;
          CFNumberGetValue(coreCountRef, kCFNumberIntType, &coreCount);
          CFRelease(properties);
          IOObjectRelease(service);
          IOObjectRelease(iterator);
          return coreCount;
        }

        if (properties)
          CFRelease(properties);
      }

      IOObjectRelease(service);
    }

    IOObjectRelease(iterator);
    ATTRIBUTE_UNSUPPORTED(MULTIPROCESSOR_COUNT);
#pragma clang diagnostic pop
  }

  // The max total threads per threadgroup is a dynamic property which is
  // accessible through the computePipeline->maxTotalThreadsPerThreadgroup size
  // property, i.e. it's tied to a specific kernel invocation rather than
  // available as a static/dynamic device property.
  case Device::Attribute::MAX_THREADS_PER_MULTIPROCESSOR:
    ATTRIBUTE_UNSUPPORTED(MAX_THREADS_PER_MULTIPROCESSOR);

  // TODO: Would it make more sense to map the compute capability major/minor
  // to the Apple GPU family instead?
  // Note that M3 and M4 both map to the Apple9 family.
  case Device::Attribute::COMPUTE_CAPABILITY_MAJOR: {
    std::string deviceName;
    if (device_ && device_->name()) {
      deviceName = device_->name()->cString(NS::UTF8StringEncoding);
    }
    return getComputeCapability(deviceName);
  }
  case Device::Attribute::COMPUTE_CAPABILITY_MINOR:
    return 0;

  // Max shared memory per GPU core is not documented.
  case Device::Attribute::MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
    ATTRIBUTE_UNSUPPORTED(MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);

  // Max registers per GPU core is not documented.
  case Device::Attribute::MAX_REGISTERS_PER_MULTIPROCESSOR:
    ATTRIBUTE_UNSUPPORTED(MAX_REGISTERS_PER_MULTIPROCESSOR);

  // Cooperative launch is not supported.
  case Device::Attribute::COOPERATIVE_LAUNCH:
    ATTRIBUTE_UNSUPPORTED(COOPERATIVE_LAUNCH);

  // Max threadgroups per GPU core is not documented.
  case Device::Attribute::MAX_BLOCKS_PER_MULTIPROCESSOR:
    ATTRIBUTE_UNSUPPORTED(MAX_BLOCKS_PER_MULTIPROCESSOR);

  // Max access policy window size is not supported.
  case Device::Attribute::MAX_ACCESS_POLICY_WINDOW_SIZE:
    ATTRIBUTE_UNSUPPORTED(MAX_ACCESS_POLICY_WINDOW_SIZE);
  }

#undef ATTRIBUTE_UNSUPPORTED
}

ErrorOr<std::string> MetalDevice::getPciBusId() const {
  RETURN_ERROR("PCI bus address not available for Metal devices");
}

int MetalDevice::deviceCount() const { return 1; }

ErrorOrSuccess MetalDevice::chunkAlloc(void **ptr, size_t byteSize,
                                       bool isHost) const {
  assert(byteSize != 0);

  ensureThreadlocalAutoreleasePoolExists();

  MTL::Device *metalDev = metalDevice();
  if (metalDev == nullptr)
    RETURN_ERROR("Invalid Metal device: null pointer");

  // Create Metal buffer with shared storage mode for all allocations
  MTL::ResourceOptions options = MTL::ResourceStorageModeShared;

  NS::SharedPtr<MTL::Buffer> buf =
      NS::TransferPtr(metalDev->newBuffer(byteSize, options));
  if (!buf)
    RETURN_ERROR("Failed to create Metal buffer in chunkAlloc");

  // For both host and device allocations, return the contents pointer
  // The memory manager will manage the memory layout
  void *contents = buf->contents();
  if (contents == nullptr)
    RETURN_ERROR("Failed to get contents from Metal buffer");

  *ptr = contents;

  // Track this allocation
  trackBuffer(contents, buf.get());
  buf.detach();

  return success();
}

ErrorOrSuccess MetalDevice::chunkFree(void *ptr, bool isHost) const {
  if (ptr == nullptr)
    return success();

  // Look up the Metal buffer for this pointer
  MTL::Buffer *bufferToRelease = getTrackedBuffer(ptr);

  if (bufferToRelease == nullptr) {
    // This shouldn't happen if all allocations go through chunkAlloc
    RETURN_ERROR("Attempt to free untracked Metal buffer in chunkFree");
  }

  // Release the buffer and remove from tracking
  bufferToRelease->release();
  untrackBuffer(ptr);

  return success();
}

ErrorOr<void *> MetalDevice::getCurrentDriverContext() const {
  RETURN_ERROR("Metal not supported");
}

ErrorOrSuccess MetalDevice::setCurrentDriverContext(void *ctx) const {
  RETURN_ERROR("Metal not supported");
}

ErrorOr<DeviceContextRef> MetalDevice::createContext() const {
  ensureThreadlocalAutoreleasePoolExists();

  NS::SharedPtr<MTL::CommandQueue> commandQueue =
      NS::TransferPtr(device_->newCommandQueue());
  if (!commandQueue)
    RETURN_ERROR("Failed to create Metal command queue for context.");

  return MetalDeviceContextRef::create(*this, commandQueue);
}

ErrorOr<DeviceStreamRef> MetalDeviceContext::createStream(int priority) const {
  RETURN_ERROR("Unimplemented for Metal devices");
}

ErrorOrSuccess
MetalDeviceContext::streamPriorityRange(int *leastPriority,
                                        int *greatestPriority) const {
  *leastPriority = 0;
  *greatestPriority = 0;
  return success();
}

ErrorOr<DeviceStreamRef>
MetalDeviceContext::createExternalStream(void *externalStream) const {
  RETURN_ERROR("Metal not supported");
}

const MetalDevice &MetalDevice::cast(const Device &device) {
  return static_cast<const MetalDevice &>(device);
}

ErrorOr<const MetalDevice &> MetalDevice::get(int id) {
  if (id != 0)
    RETURN_ERROR("Apple Metal device only supports id == 0.");

  static RCRef<MetalDevice> metalDeviceInstance;

  if (!metalDeviceInstance) {
    ensureThreadlocalAutoreleasePoolExists();

    NS::SharedPtr<MTL::Device> device =
        NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!device)
      RETURN_ERROR("Failed to create Metal device.");

    std::string name = device->name()->cString(NS::UTF8StringEncoding);

    Device::MemInfo memInfo;
    bool useMemoryManager = false;
    if (useMemoryManager) {
      DEFINE_OR_RETURN_ERROR(Device::MemInfo, memInfoResult,
                             getMemInfo(device.get()));
      memInfo = memInfoResult;
    }

    size_t allocAlignment = 64;

    metalDeviceInstance =
        RCRef<MetalDevice>::create(id, device, name, memInfo, allocAlignment);
  }

  return *metalDeviceInstance;
}

ErrorOr<int> MetalDevice::numberOfGPUs() {
  ensureThreadlocalAutoreleasePoolExists();
  int count = 0;
  NS::SharedPtr<NS::Array> devices = NS::TransferPtr(MTL::CopyAllDevices());
  if (devices) {
    count = devices->count();
  }
  return count;
}

//
// Buffer tracking implementation
//
void MetalDevice::trackBuffer(void *ptr, MTL::Buffer *buffer) const {
  std::lock_guard<std::mutex> lock(bufferMapMutex_);
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  allocations_[addr] = BufferAllocation{addr, buffer->length(), buffer};
}

void MetalDevice::untrackBuffer(void *ptr) const {
  std::lock_guard<std::mutex> lock(bufferMapMutex_);
  allocations_.erase(reinterpret_cast<uintptr_t>(ptr));
}

MetalDevice::BufferLookupResult
MetalDevice::findContainingBuffer(const void *ptr) const {
  std::lock_guard<std::mutex> lock(bufferMapMutex_);
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);

  // Find first allocation with base address > addr
  auto it = allocations_.upper_bound(addr);
  if (it != allocations_.begin()) {
    --it; // Step back to candidate allocation
    const BufferAllocation &alloc = it->second;
    if (addr >= alloc.baseAddress && addr < alloc.baseAddress + alloc.size) {
      return {alloc.buffer, static_cast<size_t>(addr - alloc.baseAddress)};
    }
  }
  return {nullptr, 0};
}

MTL::Buffer *MetalDevice::getTrackedBuffer(const void *ptr) const {
  return findContainingBuffer(ptr).buffer;
}

bool MetalDevice::isTrackedBuffer(const void *ptr) const {
  return findContainingBuffer(ptr).isValid();
}

std::map<uintptr_t, MetalDevice::BufferAllocation>
MetalDevice::getTrackedBuffers() const {
  std::lock_guard<std::mutex> lock(bufferMapMutex_);
  // return current copy of the tracked buffers.
  return allocations_;
}

//
//
// ===== MetalDeviceContext ===== //
//

MetalDeviceContext::MetalDeviceContext(
    const MetalDevice &device, NS::SharedPtr<MTL::CommandQueue> commandQueue)
    : DeviceContext(device), commandQueue_(commandQueue) {}

ErrorOrSuccess MetalDeviceContext::enqueueCopyToDevice(void *dst,
                                                       const void *src,
                                                       size_t byteSize) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueCopyToDevice",
                       TRACY_COLOR_PURPLE);

  ensureThreadlocalAutoreleasePoolExists();

  // Validate source
  if (src == nullptr)
    RETURN_ERROR("Source pointer is null");

  if (byteSize == 0)
    return success();

  // For host destination buffers, we can do a direct memcpy
  if (!isMetalBuffer(dst)) {
    // Host buffer - do direct memcpy

    // create a empty command buffer
    MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
    if (cmdBuffer == nullptr)
      RETURN_ERROR("Failed to create command buffer");

    // add completion handler to the command buffer

    cmdBuffer->addCompletedHandler(^(MTL::CommandBuffer *buffer) {
      std::memcpy(dst, src, byteSize);
    });

    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    return success();
  }

  // Get destination Metal buffer with offset for device buffers (handles
  // sub-buffers)
  MetalDevice::BufferLookupResult dstResult = getMetalBufferWithOffset(dst);
  if (!dstResult.isValid())
    RETURN_ERROR("Invalid Metal buffer pointer");

  NS::UInteger bufferLength = dstResult.buffer->length();
  if (bufferLength == 0)
    RETURN_ERROR("Metal buffer length is 0 - buffer may be invalid");

  // Check that copy fits within the buffer from the offset
  if (dstResult.offset + byteSize > bufferLength)
    RETURN_ERROR("Copy size exceeds Metal buffer length");

  // Check if source is a Metal buffer allocated by our context
  MetalDevice::BufferLookupResult srcResult = getMetalBufferWithOffset(src);
  if (srcResult.isValid()) {
    // Source is a Metal buffer - use blit encoder for copy with offsets
    return enqueueMetalCopy(srcResult.buffer, dstResult.buffer,
                            srcResult.offset, dstResult.offset, byteSize);
  }

  // Source is regular host memory - use memcpy approach

  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr)
    RETURN_ERROR("Failed to create command buffer");

  // For memcpy operations, we need to wait for completion even though we commit
  // to queue

  __block size_t capturedByteSize = byteSize;
  __block size_t capturedOffset = dstResult.offset;
  MTL::Buffer *dstBuf = dstResult.buffer;

  // Add completion handler to perform the memcpy when GPU is done
  cmdBuffer->addCompletedHandler(^(MTL::CommandBuffer *buffer) {
    void *dstContents = dstBuf->contents();
    if (dstContents) {
      // Apply offset to destination pointer
      void *dstPtr = static_cast<char *>(dstContents) + capturedOffset;
      std::memcpy(dstPtr, src, capturedByteSize);

      // If Managed mode, mark the range as modified
    }
  });

  // Commit to queue but wait for memcpy completion
  cmdBuffer->commit();
  cmdBuffer->waitUntilCompleted();

  return success();
}

ErrorOrSuccess
MetalDeviceContext::enqueueCopyFromDevice(void *dst, const void *src,
                                          size_t byteSize) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueCopyFromDevice",
                       TRACY_COLOR_PURPLE);

  ensureThreadlocalAutoreleasePoolExists();

  if (dst == nullptr)
    RETURN_ERROR("Destination pointer cannot be null");

  if (byteSize == 0)
    return success(); // no-op

  // Get source Metal buffer with offset (handles sub-buffers)
  MetalDevice::BufferLookupResult srcResult = getMetalBufferWithOffset(src);
  if (!srcResult.isValid())
    RETURN_ERROR("Invalid source buffer");

  // When we have 2 Device buffers we can do an async copy via
  // enqueueMetalCopy() similar to what the CUDA implementation does
  // However, when we are using raw pointers we need memcpy and wait for
  // completion otherwise, the next command buffer will be executed before the
  // current one is completed

  // Check if destination is a Metal buffer allocated by our context
  MetalDevice::BufferLookupResult dstResult = getMetalBufferWithOffset(dst);
  if (dstResult.isValid()) {
    // Destination is a Metal buffer - use blit encoder for copy with offsets
    return enqueueMetalCopy(srcResult.buffer, dstResult.buffer,
                            srcResult.offset, dstResult.offset, byteSize);
  }

  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr)
    RETURN_ERROR("Failed to create command buffer");

  // For memcpy operations, we need to wait for completion even though we commit
  // to queue Add completion handler to perform the memcpy when GPU is done
  __block size_t capturedOffset = srcResult.offset;
  MTL::Buffer *srcBuf = srcResult.buffer;

  cmdBuffer->addCompletedHandler(^(MTL::CommandBuffer *buffer) {
    void *srcContents = srcBuf->contents();
    if (srcContents) {
      // Apply offset to source pointer
      void *srcPtr = static_cast<char *>(srcContents) + capturedOffset;
      std::memcpy(dst, srcPtr, byteSize);
    }
  });

  // Commit to queue but wait for memcpy completion
  // When doing cpu memcopy, we need to wait for the command buffer to complete
  // before committing the next command buffer otherwise, the next command
  // buffer will be executed before the current one is completed
  cmdBuffer->commit();
  cmdBuffer->waitUntilCompleted();

  return success();
}

ErrorOrSuccess MetalDeviceContext::memCopyDtoD(const DeviceBuffer &dst,
                                               const DeviceBuffer &src,
                                               size_t srcByteOffset,
                                               size_t srcByteSize) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::memCopyDtoD", TRACY_COLOR_PURPLE);

  // 1) Basic validation
  if (src.byteSize() == 0 || dst.byteSize() == 0)
    return success(); // no-op for zero-size

  if (dst.byteSize() < src.byteSize())
    RETURN_ERROR("Destination buffer size must be >= source buffer size");

  if (srcByteOffset + srcByteSize < src.byteSize())
    RETURN_ERROR("Requested source slice larger than source buffer");

  // 2) Handle Metal buffer to Metal buffer copies (both host and device are
  // Metal buffers now) with offset support
  // This allows sub-buffers and slices to work correctly
  MetalDevice::BufferLookupResult srcResult =
      getMetalBufferWithOffset(src.ptr());
  MetalDevice::BufferLookupResult dstResult =
      getMetalBufferWithOffset(dst.ptr());

  if (!srcResult.isValid() || !dstResult.isValid())
    RETURN_ERROR("Invalid Metal buffer pointer");

  // 3) Delegate to enqueueMetalCopy with offsets for all Metal-to-Metal copies
  return enqueueMetalCopy(srcResult.buffer, dstResult.buffer, srcResult.offset,
                          dstResult.offset, src.byteSize());
}

ErrorOrSuccess MetalDeviceContext::enqueueSetMemory(void *dst, size_t byteSize,
                                                    uint64_t val,
                                                    size_t valSize) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueSetMemory",
                       TRACY_COLOR_PURPLE);

  ensureThreadlocalAutoreleasePoolExists();

  if (byteSize == 0)
    return success(); // no-op

  // Get the Metal buffer for the destination with offset (handles sub-buffers)
  MetalDevice::BufferLookupResult dstResult = getMetalBufferWithOffset(dst);
  if (!dstResult.isValid())
    RETURN_ERROR("Destination buffer is null");

  void *dstContents = dstResult.buffer->contents();
  if (dstContents == nullptr)
    RETURN_ERROR("Shared buffer contents() returned null");

  // Apply offset to get actual destination pointer
  void *dstPtr = static_cast<char *>(dstContents) + dstResult.offset;

  size_t count = byteSize / valSize;
  if (reinterpret_cast<uintptr_t>(dstPtr) % valSize != 0)
    RETURN_ERROR("destination buffer not properly aligned");

  // Push  an empty command buffer to the queue
  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr)
    RETURN_ERROR("Failed to create command buffer");

  // add completion handler to the command buffer
  cmdBuffer->addCompletedHandler(^(MTL::CommandBuffer *buffer) {
    switch (valSize) {
    case 1:
      std::memset(dstPtr, static_cast<uint8_t>(val), byteSize);
      break;
    case 2: {
      uint16_t v16 = static_cast<uint16_t>(val);
      uint16_t *p16 = static_cast<uint16_t *>(dstPtr);
      for (size_t i = 0; i < count; ++i)
        p16[i] = v16;
      break;
    }
    case 4: {
      uint32_t v32 = static_cast<uint32_t>(val);
      uint32_t *p32 = static_cast<uint32_t *>(dstPtr);
      for (size_t i = 0; i < count; ++i)
        p32[i] = v32;
      break;
    }
    case 8: {
      uint64_t *p64 = static_cast<uint64_t *>(dstPtr);
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

ErrorOr<DeviceFunctionRef>
MetalDeviceContext::loadFunction(StringRef moduleName, StringRef functionName,
                                 StringRef code, int32_t maxDynamicSharedBytes,
                                 StringRef debugLevel,
                                 int32_t optimizationLevel) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::loadFunction", TRACY_COLOR_PURPLE);
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  // Get the Metal device
  MTL::Device *mtlDevice = MetalDevice::cast(device()).metalDevice();
  if (mtlDevice == nullptr)
    RETURN_ERROR("Failed to get Metal device");

  NS::SharedPtr<MTL::Library> library;

  // Detect if input is binary or MSL source code
  bool isBinary = false;
  if (code.size() >= 4) {
    // Check if it's NOT MSL text (which would start with valid ASCII/UTF-8)
    // Look for non-printable characters in the first few bytes
    isBinary = false;
    for (size_t i = 0; i < std::min(size_t(8), code.size()); ++i) {
      unsigned char c = static_cast<unsigned char>(code[i]);
      if (c < 32 && c != '\n' && c != '\r' && c != '\t') {
        isBinary = true;
        break;
      }
    }
  }

  if (!isBinary) {
    // Compile from MSL source code
    NS::String *sourceString =
        NS::String::string(code.data(), NS::UTF8StringEncoding);
    NS::SharedPtr<MTL::CompileOptions> options =
        NS::TransferPtr(MTL::CompileOptions::alloc()->init());

    NS::Error *error = nullptr;
    library = NS::TransferPtr(
        mtlDevice->newLibrary(sourceString, options.get(), &error));
    if (error) {
      RETURN_ERROR(
          Twine("Failed to create Metal library (compiling MSL to AIR): ") +
          error->localizedDescription()->cString(NS::UTF8StringEncoding));
    }
  } else {
    // Load from binary metallib data
    dispatch_data_t binaryData = dispatch_data_create(
        code.data(), code.size(), nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);

    NS::Error *error = nullptr;
    library = NS::TransferPtr(mtlDevice->newLibrary(binaryData, &error));
    if (error) {
      RETURN_ERROR(
          Twine("Failed to create Metal library (loading binary/metallib): ") +
          error->localizedDescription()->cString(NS::UTF8StringEncoding));
    }

    // Release the dispatch data
    dispatch_release(binaryData);
  }

  // Get the function from the library
  NS::String *funcName =
      NS::String::string(functionName.data(), NS::UTF8StringEncoding);
  NS::SharedPtr<MTL::Function> function =
      NS::TransferPtr(library->newFunction(funcName));
  if (!function) {
    RETURN_ERROR("Failed to create Metal function: " + functionName.str());
  }

  // Create a compute pipeline state with reflection to get argument info
  NS::SharedPtr<MTL::ComputePipelineDescriptor> pipelineDesc =
      NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  pipelineDesc->setComputeFunction(function.get());
  pipelineDesc->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);

  // Create pipeline with reflection to get argument count
  MTL::ComputePipelineReflection *reflection = nullptr;
  NS::Error *error = nullptr;
  NS::SharedPtr<MTL::ComputePipelineState> pipelineState =
      NS::TransferPtr(mtlDevice->newComputePipelineState(
          pipelineDesc.get(), MTL::PipelineOptionArgumentInfo, &reflection,
          &error));

  if (error) {
    RETURN_ERROR(
        Twine("Failed to create compute pipeline state (GPU machine code "
              "generation): ") +
        error->localizedDescription()->cString(NS::UTF8StringEncoding));
  }

  if (!pipelineState)
    RETURN_ERROR("Failed to create compute pipeline state (no error returned)");

  if (!reflection)
    RETURN_ERROR("Failed to extract pipeline argument information");

  // Extract argument count from reflection, defaulting to 0 if unavailable
  NS::Array *arguments = reflection->arguments();
  size_t argCount = arguments ? arguments->count() : 0;

  // Create and return a MetalDeviceFunction
  MetalDeviceFunctionRef func = MetalDeviceFunctionRef::create(
      *this, moduleName, functionName, pipelineState, argCount);

  return func;
}

ErrorOrSuccess MetalDeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void * /*attrs*/, uint32_t numAttrs, void **args,
    const size_t *argSizes) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueFunctionExecDirect",
                       TRACY_COLOR_PURPLE);

  ensureThreadlocalAutoreleasePoolExists();

  // 1) Validate pipeline
  const MetalDeviceFunction &metalFunc = MetalDeviceFunction::cast(func);
  MTL::ComputePipelineState *pipeline = metalFunc.getPipelineState();

  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr)
    RETURN_ERROR("Failed to create command buffer");

  // 3) Create compute encoder
  MTL::ComputeCommandEncoder *encoder = cmdBuffer->computeCommandEncoder();
  if (encoder == nullptr)
    RETURN_ERROR("Failed to create compute command encoder");

  encoder->setComputePipelineState(pipeline);
  size_t numArgs = func.argumentCount();
  const MetalDevice &metalDevice = MetalDevice::cast(device());
  MTL::Device *mtlDevice = metalDevice.metalDevice();
  std::vector<NS::SharedPtr<MTL::Buffer>> newBuffers(numArgs);

  // 4) Bind all buffer arguments
  for (uint32_t i = 0; i < numArgs; ++i) {
    if (args[i] == nullptr) {
      encoder->endEncoding();
      RETURN_ERROR("Null argument at index " + std::to_string(i));
    }

    const size_t argSize = argSizes[i];

    if (argSize == 0) {
      RETURN_ERROR("Unknown argument size at index " + std::to_string(i));
    }

    MTL::Buffer *buf = nullptr;
    if (argSize == sizeof(void *)) {
      // For simple pointers argument, try to either get a cached MTLBuffer or
      // just set an argument using given bytes
      uint64_t *argData = static_cast<uint64_t *>(args[i]);
      uint64_t originalPtr = argData[0];
      void *ptrValue = reinterpret_cast<void *>(originalPtr);
      // FIXME: There's a possible bug here: if the argument is an integer that
      // has a value of a cached buffer, we will use that buffer instead of
      // encoding the integer.
      // That can result to flaky behavior. To address it, we should be able to
      // distinguish integer vs pointer arguments here.
      // Because integer arguments are currently converted to pointers (Metal
      // only accepts its `uint` type by value`), the issue is not likely to
      // happen.
      buf = getMetalBuffer(ptrValue);
      if (buf != nullptr)
        encoder->setBuffer(buf, 0, i);
      else
        encoder->setBytes(args[i], argSize, i);
    } else {
      // We have more complex argument, such as pointer to a struct. Allocate
      // new MTLBuffer and set it.
      MTL::ResourceOptions options =
          getResourceOptions(Device::AllocMode::ASYNC);
      NS::SharedPtr<MTL::Buffer> buf =
          NS::TransferPtr(mtlDevice->newBuffer(args[i], argSize, options));
      encoder->setBuffer(buf.get(), 0, i);
      newBuffers.push_back(buf);
    }
  }

  // 5) Set resources that might be used by the kernel
  // FIXME: only set resources that are used by the kernel
  for (const auto &it : metalDevice.getTrackedBuffers())
    encoder->useResource(it.second.buffer, MTL::ResourceUsageWrite);

  // 6) Dispatch
  MTL::Size gridSize{gridX, gridY, gridZ};
  MTL::Size threadgroupSize{blockX, blockY, blockZ};

  encoder->dispatchThreadgroups(gridSize, threadgroupSize);

  // 7) End encoding
  encoder->endEncoding();

  // 8) Commit and release
  cmdBuffer->commit();

  // 9) Wait until completion. This is a blocking call therefore we have no pure
  // asynchronous execution. Remove it with GEX-2503 fixed.
  cmdBuffer->waitUntilCompleted();

  return success();
}

ErrorOr<DeviceStreamRef> MetalDeviceContext::stream() const {
  RETURN_ERROR("Metal stream not implemented");
}

ErrorOrSuccess MetalDeviceContext::synchronize() const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::synchronize", TRACY_COLOR_PURPLE);

  ensureThreadlocalAutoreleasePoolExists();

  // Check for any pending errors first, similar to CUDA's behavior
  RETURN_IF_ERROR(checkPendingError());

  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr)
    RETURN_ERROR("Failed to create command buffer for sync");

  // Submit an empty command buffer and wait for it to complete
  // This acts as a fence, ensuring all previously submitted work completes
  cmdBuffer->commit();
  cmdBuffer->waitUntilCompleted();

  return success();
}

ErrorOr<DeviceEventRef> MetalDeviceContext::enqueueEvent() const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueEvent", TRACY_COLOR_PURPLE);

  ensureThreadlocalAutoreleasePoolExists();

  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (!cmdBuffer)
    RETURN_ERROR("Failed to create command buffer for event");

  // Commit the command buffer - this creates the event
  cmdBuffer->commit();

  return MetalDeviceEventRef::create(MetalDevice::cast(device()), cmdBuffer);
}

ErrorOrSuccess
MetalDeviceContext::enqueueEvent(const DeviceEvent &event) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueEvent", TRACY_COLOR_PURPLE);
  RETURN_ERROR("Unimplemented for Metal devices");
}

ErrorOr<DeviceEventRef>
MetalDeviceContext::eventCreate(unsigned int flags) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::eventCreate", TRACY_COLOR_PURPLE);
  RETURN_ERROR("Unimplemented for Metal devices");
}

ErrorOrSuccess
MetalDeviceContext::enqueueWaitFor(const DeviceEvent &event) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueWaitFor",
                       TRACY_COLOR_PURPLE);

  // This is a synchronous implementation of the enqueueWaitFor().
  // Cast to MetalDeviceEvent to get the command buffer
  const MetalDeviceEvent &metalEvent = MetalDeviceEvent::cast(event);

  // For Metal, we need to wait for the event's command buffer to complete
  // before continuing with subsequent operations on our command queue
  RETURN_IF_ERROR(metalEvent.synchronize());

  return success();
}

ErrorOrSuccess
MetalDeviceContext::enqueueWaitFor(const DeviceContext &ctx) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueWaitFor",
                       TRACY_COLOR_PURPLE);

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

ErrorOr<Device::MemInfo> MetalDeviceContext::getDeviceMemoryInfo() const {
  RETURN_IF_ERROR(synchronize());
  return device().getMemoryInfo();
}

ErrorOrSuccess MetalDeviceContext::occupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, const DeviceFunction &func, int blockSize,
    size_t dynamicSharedMemSize) const {
  RETURN_ERROR("Unimplemented for Metal devices");
}

ErrorOrSuccess MetalDeviceContext::memAlloc(void **ptr, size_t byteSize,
                                            Device::AllocMode mode) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::memAlloc", TRACY_COLOR_PURPLE);

  ensureThreadlocalAutoreleasePoolExists();

  if (ptr == nullptr)
    RETURN_ERROR("Invalid allocation parameters: ptr is null");
  if (byteSize == 0) {
    *ptr = nullptr;
    return success();
  }

  MTL::Device *metalDev = MetalDevice::cast(device()).metalDevice();
  if (metalDev == nullptr)
    RETURN_ERROR("Invalid Metal device: null pointer");

  // create the Metal buffer
  MTL::ResourceOptions options = getResourceOptions(mode);
  // TODO: Ideally this should be stored in an NS::SharedPtr for RAII refcount
  // management, however this causes "Trace/breakpoint trap" crashes in some
  // tests. This points to a subtle interaction with how MTL::Buffer* objects
  // are currently managed by MetalDevice::trackBuffer() and
  // MetalDevice::untrackBuffer().
  MTL::Buffer *buf = metalDev->newBuffer(byteSize, options);
  if (buf == nullptr)
    RETURN_ERROR("Failed to create Metal buffer");
  buf->retain();

  // Based on mode of allocation, return the appropriate pointer
  switch (mode) {
  case Device::AllocMode::HOST: {
    // For host buffers, return the contents pointer
    void *contents = buf->contents();
    if (contents == nullptr) {
      buf->release();
      RETURN_ERROR("Failed to get contents from Metal buffer");
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
      RETURN_ERROR("Failed to get GPU address from Metal buffer");
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
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::memFree", TRACY_COLOR_PURPLE);

  if (ptr == nullptr)
    return success();

  // Check if this pointer was allocated by us
  MTL::Buffer *bufferToRelease = getMetalBuffer(ptr);

  if (bufferToRelease == nullptr) {
    // Not a Metal buffer allocated by us - this is an error
    RETURN_ERROR("Attempt to free non-Metal memory - operation failed");
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
  RETURN_ERROR("Metal memmapFileImpl not implemented");
}

void *MetalDeviceContext::driverContext() const {
  return const_cast<MetalDeviceContext *>(this);
}

ErrorOrSuccess MetalDeviceContext::enqueueMetalCopy(MTL::Buffer *src,
                                                    MTL::Buffer *dst,
                                                    size_t srcOffset,
                                                    size_t dstOffset,
                                                    size_t size) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueMetalCopy",
                       TRACY_COLOR_PURPLE);

  ensureThreadlocalAutoreleasePoolExists();

  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr) {
    RETURN_ERROR("Failed to create command buffer");
  }

  MTL::BlitCommandEncoder *blit = cmdBuffer->blitCommandEncoder();
  if (blit == nullptr)
    RETURN_ERROR("Failed to create blit encoder");

  blit->copyFromBuffer(src, srcOffset, dst, dstOffset, size);
  blit->endEncoding();

  cmdBuffer->commit();
  return success();
}

MetalDevice::BufferLookupResult
MetalDeviceContext::getMetalBufferWithOffset(const void *ptr) const {
  return MetalDevice::cast(device()).findContainingBuffer(ptr);
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

const MetalDeviceContext *
MetalDeviceContext::verified_cast(const DeviceContext *ctx) {
  if (ctx->device().api() != MetalDevice::api)
    return nullptr;
  return static_cast<const MetalDeviceContext *>(ctx);
}

ErrorOr<std::string> MetalDevice::getArchName(MTL::Device *d) {
  ensureThreadlocalAutoreleasePoolExists();
  MTL::Device *device = d ? d : MTL::CreateSystemDefaultDevice();
  if (device == nullptr)
    RETURN_ERROR("Failed to create Metal device.");
  std::string name = device->name()->cString(NS::UTF8StringEncoding);

  DEFINE_OR_RETURN_ERROR(int, cc, getComputeCapability(name));
  return std::to_string(cc);
}

//
//
// ===== MetalDeviceEvent ===== //
//

MetalDeviceEvent::MetalDeviceEvent(const MetalDevice &device,
                                   MTL::CommandBuffer *cmdBuffer)
    : DeviceEvent(device), cmdBuffer_(NS::TransferPtr(cmdBuffer)) {
  if (cmdBuffer_)
    cmdBuffer_->retain(); // Take ownership of autoreleased MTL::CommandBuffer
}

ErrorOrSuccess MetalDeviceEvent::synchronize() const {
  if (!cmdBuffer_)
    RETURN_ERROR("MetalDeviceEvent: null command buffer");
  cmdBuffer_->waitUntilCompleted();
  return success();
}

ErrorOr<bool> MetalDeviceEvent::isReady() const {
  if (!cmdBuffer_)
    RETURN_ERROR("MetalDeviceEvent: null command buffer");

  MTL::CommandBufferStatus status = cmdBuffer_->status();
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

//
//
// ===== MetalDeviceFunction ===== //
//

MetalDeviceFunction::MetalDeviceFunction(
    const MetalDeviceContext &context, StringRef moduleName,
    StringRef functionName,
    NS::SharedPtr<MTL::ComputePipelineState> pipelineState,
    size_t argumentCount)
    : DeviceFunction(context), name_(functionName.str()),
      pipelineState_(pipelineState), argumentCount_(argumentCount) {}

size_t MetalDeviceFunction::argumentCount() const { return argumentCount_; }

const std::string &MetalDeviceFunction::name() const { return name_; }

ErrorOrSuccess MetalDeviceFunction::copyToNamedConstant(const char *name,
                                                        const void *data,
                                                        size_t byteSize) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceFunction::copyToNamedConstant",
                       TRACY_COLOR_PURPLE);
  // Metal doesn't support named constants in the same way as CUDA/HIP
  RETURN_ERROR("Named constants not supported in Metal");
}

ErrorOr<int32_t> MetalDeviceFunction::getAttribute(int32_t code) const {
  // Metal doesn't support the same attributes as CUDA/HIP
  RETURN_ERROR("Attributes not supported in Metal");
}

const MetalDeviceFunction &
MetalDeviceFunction::cast(const DeviceFunction &func) {
  return static_cast<const MetalDeviceFunction &>(func);
}

} // namespace M::Driver

#endif // MODULAR_MACOS
