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
#ifdef MODULAR_MACOS

#include "MLRT/AsyncRT/Runtime/Globals/Globals.h"
#include "MetalDriver.hpp"
#include "Support/AlignedAlloc.h"
#include "Support/ErrorOr.h"
#include "Support/Profiling/Tracy.h"

#include "llvm/ADT/Twine.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <unistd.h>

#define RAISE_ERROR()                                                          \
  do {                                                                         \
    return DeviceContext::formatError(__FILE__, __LINE__, __FUNCTION__, "");   \
  } while (0)

#define RAISE_ERROR_MESSAGE(msg)                                               \
  do {                                                                         \
    return DeviceContext::formatError(__FILE__, __LINE__, __FUNCTION__, msg);  \
  } while (0)

#define CHECK_IS_METAL_ERR(dev)                                                \
  do {                                                                         \
    if ((dev).api() != Device::metalAPI)                                       \
      RAISE_ERROR_MESSAGE("Not a Metal device.");                              \
  } while (0)

#define METAL_MAYBE_SYNC(ctx)                                                  \
  do {                                                                         \
    const DeviceContext &metal_ctx = (ctx);                                    \
    if (metal_ctx.syncMode()) {                                                \
      ErrorOrSuccess _sync_result = metal_ctx.synchronize();                   \
      if (_sync_result.isError())                                              \
        return M::Driver::DeviceContext::formatError(                          \
            __FILE__, __LINE__, __FUNCTION__, _sync_result.getError());        \
    }                                                                          \
  } while (0)

namespace M::Driver {

static ErrorOr<int> getComputeCapability(StringRef name) {
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

  RAISE_ERROR_MESSAGE("Unknown device name");
}

//
//
// ===== MetalDevice ===== //
//
MetalDevice::MetalDevice(int id, NS::SharedPtr<MTL::Device> device,
                         std::string name, Device::MemInfo memInfo,
                         size_t allocAlignment, int64_t deviceMemoryLimit)
    : Device(api, name, id, memInfo, allocAlignment, deviceMemoryLimit),
      device_(device), memInfo_(memInfo), driverVersion_([]() -> std::string {
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
    RAISE_ERROR_MESSAGE(strerror(errno));
  return static_cast<size_t>(value);
}

ErrorOr<Device::MemInfo> MetalDevice::getMemInfo() {
  Device::MemInfo info;
  ErrorOr<size_t> numPages = getSystemValue(_SC_PHYS_PAGES);
  if (numPages.isError())
    return numPages.takeError();

  ErrorOr<size_t> pageSize = getSystemValue(_SC_PAGE_SIZE);
  if (pageSize.isError())
    return pageSize.takeError();

  size_t pages = numPages.takeValue();
  size_t size = pageSize.takeValue();

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
  RAISE_ERROR_MESSAGE("Attribute \"" #ATTR "\" not supported for Apple GPU.")

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

  // TODO: The GPU core count can be inferred from the device name.
  case Device::Attribute::MULTIPROCESSOR_COUNT:
    ATTRIBUTE_UNSUPPORTED(MULTIPROCESSOR_COUNT);

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

int MetalDevice::deviceCount() const { return 1; }

ErrorOrSuccess MetalDevice::chunkAlloc(void **ptr, size_t byteSize,
                                       bool isHost) const {
  assert(byteSize != 0);

  MTL::Device *metalDev = metalDevice();
  if (metalDev == nullptr)
    RAISE_ERROR_MESSAGE("Invalid Metal device: null pointer");

  // Create Metal buffer with shared storage mode for all allocations
  MTL::ResourceOptions options = MTL::ResourceStorageModeShared;

  NS::SharedPtr<MTL::Buffer> buf =
      NS::TransferPtr(metalDev->newBuffer(byteSize, options));
  if (!buf)
    RAISE_ERROR_MESSAGE("Failed to create Metal buffer in chunkAlloc");

  // For both host and device allocations, return the contents pointer
  // The memory manager will manage the memory layout
  void *contents = buf->contents();
  if (contents == nullptr)
    RAISE_ERROR_MESSAGE("Failed to get contents from Metal buffer");

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
    RAISE_ERROR_MESSAGE("Attempt to free untracked Metal buffer in chunkFree");
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
  NS::SharedPtr<MTL::CommandQueue> commandQueue =
      NS::TransferPtr(device_->newCommandQueue());
  if (!commandQueue)
    RAISE_ERROR_MESSAGE("Failed to create Metal command queue for context.");

  return MetalDeviceContextRef::create(*this, commandQueue);
}

ErrorOr<DeviceStreamRef> MetalDeviceContext::streamCreate(int priority) const {
  RAISE_ERROR_MESSAGE("Unimplemented for Metal devices");
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
  RAISE_ERROR_MESSAGE("Metal not supported");
}

const MetalDevice &MetalDevice::cast(const Device &device) {
  return static_cast<const MetalDevice &>(device);
}

ErrorOr<const MetalDevice &> MetalDevice::get(int id,
                                              int64_t deviceMemoryLimit) {
  if (id != 0)
    RAISE_ERROR_MESSAGE("Apple Metal device only supports id == 0.");

  static RCRef<MetalDevice> metalDeviceInstance;

  if (!metalDeviceInstance) {
    NS::SharedPtr<MTL::Device> device =
        NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!device)
      RAISE_ERROR_MESSAGE("Failed to create Metal device.");

    std::string name = device->name()->cString(NS::UTF8StringEncoding);

    Device::MemInfo memInfo;
    bool useMemoryManager = false;
    if (useMemoryManager) {
      ErrorOr<Device::MemInfo> memInfoOrError = getMemInfo();
      if (memInfoOrError.isError())
        RAISE_ERROR_MESSAGE("Failed to get memory info from system.");
      memInfo = memInfoOrError.takeValue();
    }

    size_t allocAlignment = 64;

    metalDeviceInstance = RCRef<MetalDevice>::create(
        id, device, name, memInfo, allocAlignment, deviceMemoryLimit);
  }

  return *metalDeviceInstance;
}

ErrorOr<int> MetalDevice::numberOfGPUs() {
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

std::unordered_map<void *, MTL::Buffer *>
MetalDevice::getTrackedBuffers() const {
  std::lock_guard<std::mutex> lock(bufferMapMutex_);
  // return current copy of the tracked buffers.
  return ptrToBuffer_;
}

//
//
// ===== MetalDeviceContext ===== //
//

MetalDeviceContext::MetalDeviceContext(
    const MetalDevice &device, NS::SharedPtr<MTL::CommandQueue> commandQueue)
    : DeviceContext(device), commandQueue_(commandQueue) {}

ErrorOrSuccess MetalDeviceContext::enqueueCopyToDevice(const DeviceBuffer &dst,
                                                       const void *src) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueCopyToDevice",
                       TRACY_COLOR_PURPLE);

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
      RAISE_ERROR_MESSAGE("Failed to create command buffer");

    // add completion handler to the command buffer

    cmdBuffer->addCompletedHandler(^(MTL::CommandBuffer *buffer) {
      std::memcpy(dstPtr, src, byteSize);
    });

    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

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
    RAISE_ERROR_MESSAGE("Failed to create command buffer");

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

  return success();
}

ErrorOrSuccess
MetalDeviceContext::enqueueCopyFromDevice(void *dst,
                                          const DeviceBuffer &src) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueCopyFromDevice",
                       TRACY_COLOR_PURPLE);

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
    RAISE_ERROR_MESSAGE("Failed to create command buffer");

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

  return success();
}

ErrorOrSuccess
MetalDeviceContext::enqueueCopyToDeviceSized(void *dst, void *src,
                                             size_t byteSize) const {
  RAISE_ERROR_MESSAGE("Unimplemented for Metal devices");
}

ErrorOrSuccess
MetalDeviceContext::enqueueCopyFromDeviceSized(void *dst, void *src,
                                               size_t byteSize) const {
  RAISE_ERROR_MESSAGE("Unimplemented for Metal devices");
}

ErrorOrSuccess MetalDeviceContext::memCopyDtoD(const DeviceBuffer &dst,
                                               const DeviceBuffer &src) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::memCopyDtoD", TRACY_COLOR_PURPLE);

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
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueSetMemory",
                       TRACY_COLOR_PURPLE);

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
    RAISE_ERROR_MESSAGE("destination buffer not properly aligned");

  // Push  an empty command buffer to the queue
  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr)
    RAISE_ERROR_MESSAGE("Failed to create command buffer");

  // add completion handler to the command buffer
  cmdBuffer->addCompletedHandler(^(MTL::CommandBuffer *buffer) {
    // do nothing

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
ErrorOr<DeviceFunctionRef> MetalDeviceContext::loadFunction(
    StringRef moduleName, StringRef functionName,
    const DeviceFunction::SignatureBuilder &signature, StringRef code,
    int32_t maxDynamicSharedBytes, StringRef debugLevel,
    int32_t optimizationLevel) const {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  // Get the Metal device
  MTL::Device *mtlDevice = MetalDevice::cast(device()).metalDevice();
  if (mtlDevice == nullptr)
    RAISE_ERROR_MESSAGE("Failed to get Metal device");

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
      RAISE_ERROR_MESSAGE(
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
      RAISE_ERROR_MESSAGE(
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
    RAISE_ERROR_MESSAGE("Failed to create Metal function: " +
                        functionName.str());
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
    RAISE_ERROR_MESSAGE(
        Twine("Failed to create compute pipeline state (GPU machine code "
              "generation): ") +
        error->localizedDescription()->cString(NS::UTF8StringEncoding));
  }

  if (!pipelineState)
    RAISE_ERROR_MESSAGE(
        "Failed to create compute pipeline state (no error returned)");

  // Build a proper signature from reflection if needed
  DeviceFunction::SignatureBuilder reflectedSignature;
  bool useReflectedSignature = false;

  // Check if we need to use reflection (when signature is empty from Mojo)
  if (reflection) {
    // Get the arguments from reflection
    NS::Array *arguments = reflection->arguments();
    if (arguments && arguments->count() > 0) {
      uint32_t argCount = arguments->count();

      for (uint32_t i = 0; i < argCount; ++i) {
        MTL::Argument *arg = static_cast<MTL::Argument *>(arguments->object(i));
        if (arg->type() == MTL::ArgumentTypeBuffer) {
          reflectedSignature.addBuffer();
        } else {
          reflectedSignature.addUInt64();
        }
      }
      useReflectedSignature = true;
    }
  }

  // Create and return a MetalDeviceFunction
  // Use reflected signature if we built one, otherwise use the provided
  // signature
  MetalDeviceFunctionRef func = MetalDeviceFunctionRef::create(
      *this, moduleName, functionName,
      useReflectedSignature ? reflectedSignature : signature, pipelineState);

  return func;
}

ErrorOrSuccess MetalDeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void * /*attrs*/, uint32_t numAttrs, void **args,
    const size_t *argSizes) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueFunctionExecDirect",
                       TRACY_COLOR_PURPLE);

  // 1) Validate pipeline
  const MetalDeviceFunction &metalFunc = MetalDeviceFunction::cast(func);
  MTL::ComputePipelineState *pipeline = metalFunc.getPipelineState();

  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr)
    RAISE_ERROR_MESSAGE("Failed to create command buffer");

  // 3) Create compute encoder
  MTL::ComputeCommandEncoder *encoder = cmdBuffer->computeCommandEncoder();
  if (encoder == nullptr)
    RAISE_ERROR_MESSAGE("Failed to create compute command encoder");

  encoder->setComputePipelineState(pipeline);
  size_t numArgs = func.argumentCount();
  const MetalDevice &metalDevice = MetalDevice::cast(device());
  MTL::Device *mtlDevice = metalDevice.metalDevice();

  // 4) Bind all buffer arguments
  for (uint32_t i = 0; i < numArgs; ++i) {
    if (args[i] == nullptr) {
      encoder->endEncoding();
      RAISE_ERROR_MESSAGE("Null argument at index " + std::to_string(i));
    }

    const size_t argSize = argSizes[i];

    if (argSize == 0) {
      RAISE_ERROR_MESSAGE("Unknown argument size at index " +
                          std::to_string(i));
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
      MTL::Buffer *buf = mtlDevice->newBuffer(args[i], argSize, options);
      buf->retain();
      metalDevice.trackBuffer(buf->contents(), buf);
      encoder->setBuffer(buf, 0, i);
    }
  }

  // 5) Set resources that might be used by the kernel
  // FIXME: only set resources that are used by the kernel
  for (auto it : metalDevice.getTrackedBuffers())
    encoder->useResource(it.second, MTL::ResourceUsageWrite);

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
  RAISE_ERROR_MESSAGE("Metal stream not implemented");
}

ErrorOrSuccess MetalDeviceContext::synchronize() const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::synchronize", TRACY_COLOR_PURPLE);

  // Check for any pending errors first, similar to CUDA's behavior
  ErrorOrSuccess pendingErr = checkPendingWork();
  if (pendingErr.isError()) {
    return pendingErr.takeError();
  }

  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr)
    RAISE_ERROR_MESSAGE("Failed to create command buffer for sync");

  // Submit an empty command buffer and wait for it to complete
  // This acts as a fence, ensuring all previously submitted work completes
  cmdBuffer->commit();
  cmdBuffer->waitUntilCompleted();

  return success();
}

ErrorOr<DeviceEventRef> MetalDeviceContext::enqueueEvent() const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueEvent", TRACY_COLOR_PURPLE);

  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (!cmdBuffer)
    RAISE_ERROR_MESSAGE("Failed to create command buffer for event");

  // Commit the command buffer - this creates the event
  cmdBuffer->commit();

  return MetalDeviceEventRef::create(MetalDevice::cast(device()), cmdBuffer);
}

ErrorOr<DeviceEventRef>
MetalDeviceContext::eventCreate(unsigned int flags) const {
  RAISE_ERROR_MESSAGE("Unimplemented for Metal devices");
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
  ErrorOrSuccess syncResult = metalEvent.synchronize();
  if (syncResult.isError()) {
    return syncResult.takeError();
  }

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

ErrorOrSuccess
MetalDeviceContext::enablePeerAccess(const DeviceContext &peer) const {
  RAISE_ERROR_MESSAGE("Metal enablePeerAccess not implemented");
}

ErrorOr<Device::MemInfo> MetalDeviceContext::getDeviceMemoryInfo() const {

  ErrorOrSuccess syncResult = synchronize();
  if (syncResult.isError()) {
    return syncResult.takeError();
  }

  return device().getMemoryInfo();
}

ErrorOrSuccess MetalDeviceContext::occupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, const DeviceFunction &func, int blockSize,
    size_t dynamicSharedMemSize) const {
  RAISE_ERROR_MESSAGE("Unimplemented for Metal devices");
}

ErrorOrSuccess MetalDeviceContext::memAlloc(void **ptr, size_t byteSize,
                                            Device::AllocMode mode) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::memAlloc", TRACY_COLOR_PURPLE);

  if (ptr == nullptr)
    RAISE_ERROR_MESSAGE("Invalid allocation parameters: ptr is null");
  if (byteSize == 0) {
    *ptr = nullptr;
    return success();
  }

  MTL::Device *metalDev = MetalDevice::cast(device()).metalDevice();
  if (metalDev == nullptr)
    RAISE_ERROR_MESSAGE("Invalid Metal device: null pointer");

  // create the Metal buffer
  MTL::ResourceOptions options = getResourceOptions(mode);
  // TODO: Ideally this should be stored in an NS::SharedPtr for RAII refcount
  // management, however this causes "Trace/breakpoint trap" crashes in some
  // tests. This points to a subtle interaction with how MTL::Buffer* objects
  // are currently managed by MetalDevice::trackBuffer() and
  // MetalDevice::untrackBuffer().
  MTL::Buffer *buf = metalDev->newBuffer(byteSize, options);
  if (buf == nullptr)
    RAISE_ERROR_MESSAGE("Failed to create Metal buffer");
  buf->retain();

  // Based on mode of allocation, return the appropriate pointer
  switch (mode) {
  case Device::AllocMode::HOST: {
    // For host buffers, return the contents pointer
    void *contents = buf->contents();
    if (contents == nullptr) {
      buf->release();
      RAISE_ERROR_MESSAGE("Failed to get contents from Metal buffer");
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
      RAISE_ERROR_MESSAGE("Failed to get GPU address from Metal buffer");
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
    RAISE_ERROR_MESSAGE("Attempt to free non-Metal memory - operation failed");
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
  RAISE_ERROR_MESSAGE("Metal memmapFileImpl not implemented");
}

void *MetalDeviceContext::driverContext() const {
  return const_cast<MetalDeviceContext *>(this);
}

ErrorOrSuccess MetalDeviceContext::enqueueMetalCopy(MTL::Buffer *src,
                                                    MTL::Buffer *dst,
                                                    size_t size) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueMetalCopy",
                       TRACY_COLOR_PURPLE);

  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr) {
    RAISE_ERROR_MESSAGE("Failed to create command buffer");
  }

  MTL::BlitCommandEncoder *blit = cmdBuffer->blitCommandEncoder();
  if (blit == nullptr)
    RAISE_ERROR_MESSAGE("Failed to create blit encoder");

  blit->copyFromBuffer(src, 0, dst, 0, size);
  blit->endEncoding();

  cmdBuffer->commit();
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

const MetalDeviceContext *
MetalDeviceContext::verified_cast(const DeviceContext *ctx) {
  if (ctx->device().api() != MetalDevice::api)
    return nullptr;
  return static_cast<const MetalDeviceContext *>(ctx);
}

ErrorOr<std::string> MetalDevice::getArchName(MTL::Device *d) {
  MTL::Device *device = d ? d : MTL::CreateSystemDefaultDevice();
  if (device == nullptr)
    RAISE_ERROR_MESSAGE("Failed to create Metal device.");
  std::string name = device->name()->cString(NS::UTF8StringEncoding);

  auto ccOrErr = getComputeCapability(name);
  if (ccOrErr.isError())
    return ccOrErr.takeError();
  return std::to_string(*ccOrErr);
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
    RAISE_ERROR_MESSAGE("MetalDeviceEvent: null command buffer");
  cmdBuffer_->waitUntilCompleted();
  return success();
}

ErrorOr<bool> MetalDeviceEvent::isReady() const {
  if (!cmdBuffer_)
    RAISE_ERROR_MESSAGE("MetalDeviceEvent: null command buffer");

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
    StringRef functionName, const DeviceFunction::SignatureBuilder &signature,
    NS::SharedPtr<MTL::ComputePipelineState> pipelineState)
    : DeviceFunction(context, signature), name_(functionName.str()),
      pipelineState_(pipelineState) {}

const std::string &MetalDeviceFunction::name() const { return name_; }

ErrorOrSuccess MetalDeviceFunction::copyToNamedConstant(const char *name,
                                                        const void *data,
                                                        size_t byteSize) const {
  // Metal doesn't support named constants in the same way as CUDA/HIP
  RAISE_ERROR_MESSAGE("Named constants not supported in Metal");
}

ErrorOr<int32_t> MetalDeviceFunction::getAttribute(int32_t code) const {
  // Metal doesn't support the same attributes as CUDA/HIP
  RAISE_ERROR_MESSAGE("Attributes not supported in Metal");
}

const MetalDeviceFunction &
MetalDeviceFunction::cast(const DeviceFunction &func) {
  return static_cast<const MetalDeviceFunction &>(func);
}

} // namespace M::Driver

#endif // MODULAR_MACOS
