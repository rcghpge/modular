//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CPU device context.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/Driver/DeviceContext/ErrorMacros.h"

#include "CPU/CpuDeviceContext.h"
#include "CUDA/CUDACompilationDevice.h"
#include "CUDA/CUDADeviceContext.h"
#include "HIP/HIPCompilationDevice.h"
#include "HIP/HIPDeviceContext.h"
#include "MemoryManager.h"
#include "Metal/MetalDeviceContext.h"
#include "Support/AlignedAlloc.h"
#include "Support/AssertStream.h"
#include "Support/ErrorOr.h"
#include "Support/SymbolExport.h"
#include "Virtual/VirtualDeviceContext.h"

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Process.h"

#include <atomic>
#include <cstdio>
#include <mutex>
#include <string.h>

namespace M::Driver {

// Thread-local pointer to the currently bound device context.
static thread_local const DeviceContext *currentlyBoundContext = nullptr;

// Global count of virtual devices for device creation.
// When set to a positive value, Device::create() will return VirtualDevice
// instead of real hardware devices for GPU APIs, and Device::numberOfDevices()
// will return this count. Using atomic for thread-safe access.
static std::atomic<int> g_virtualDeviceCount{0};

MODULAR_CXX_EXPORT void setVirtualDeviceCount(int count) {
  g_virtualDeviceCount = count;
}

MODULAR_CXX_EXPORT int getVirtualDeviceCount() { return g_virtualDeviceCount; }

MODULAR_CXX_EXPORT bool isVirtualDeviceMode() {
  return g_virtualDeviceCount > 0;
}

// Global API for virtual device compilation.
// This specifies which API to use (e.g., "cuda", "hip", "metal") for
// virtual devices created in compile-only mode.
// Using function-local static for thread-safe lazy initialization.
static std::string &getVirtualDeviceAPIImpl() {
  static std::string api;
  return api;
}
static std::mutex &getVirtualDeviceAPIMutex() {
  static std::mutex mutex;
  return mutex;
}

MODULAR_CXX_EXPORT void setVirtualDeviceAPI(StringRef api) {
  std::lock_guard<std::mutex> lock(getVirtualDeviceAPIMutex());
  getVirtualDeviceAPIImpl() = api.str();
}

MODULAR_CXX_EXPORT std::string getVirtualDeviceAPI() {
  std::lock_guard<std::mutex> lock(getVirtualDeviceAPIMutex());
  return getVirtualDeviceAPIImpl();
}

// Global target architecture for virtual device compilation.
// This specifies the GPU architecture (e.g., "sm_80", "sm_90") for
// virtual devices created in compile-only mode.
// Using function-local static for thread-safe lazy initialization.
static std::string &getVirtualDeviceTargetArchImpl() {
  static std::string arch;
  return arch;
}
static std::mutex &getVirtualDeviceTargetArchMutex() {
  static std::mutex mutex;
  return mutex;
}

MODULAR_CXX_EXPORT void setVirtualDeviceTargetArch(StringRef arch) {
  std::lock_guard<std::mutex> lock(getVirtualDeviceTargetArchMutex());
  getVirtualDeviceTargetArchImpl() = arch.str();
}

MODULAR_CXX_EXPORT std::string getVirtualDeviceTargetArch() {
  std::lock_guard<std::mutex> lock(getVirtualDeviceTargetArchMutex());
  return getVirtualDeviceTargetArchImpl();
}

DeviceContextScope::DeviceContextScope(const DeviceContext &ctx)
    : prevCtx_(nullptr), err_(success()) {
  const Device &dev = ctx.device();
  ErrorOr<void *> errOrCtx = dev.getCurrentDriverContext();
  if (errOrCtx.isError()) {
    err_ = errOrCtx.takeError();
    return;
  }

  void *prevCtx = errOrCtx.takeValue();
  void *newCtx = ctx.driverContext();

  if (prevCtx == newCtx) {
    // Desired context is already current. Nothing to be done on restore, so we
    // keep prevCtx_ and device_ uninitialized.
    return;
  }

  ErrorOrSuccess err = dev.setCurrentDriverContext(newCtx);
  if (err.isError()) {
    err_ = err.takeError();
    return;
  }

  // Remember values to be restored in destructor.
  device_ = DeviceRef::copy(&dev);
  prevCtx_ = prevCtx;
}

DeviceContextScope::~DeviceContextScope() {
  if (device_.getPointer() == nullptr) {
    assert(prevCtx_ == nullptr);
    return;
  }
  ErrorOrSuccess ignore = device_->setCurrentDriverContext(prevCtx_);
}

ErrorOrSuccess DeviceContextScope::err() const {
  if (!err_.isError())
    return success();
  return err_.copy().takeError();
}

DeviceFunctionCacheEntry::DeviceFunctionCacheEntry(const std::string name)
    : name_(std::move(name)) {}

const std::string &DeviceFunctionCacheEntry::name() const { return name_; }

Error DeviceContext::formatError(const char *file, int line, const char *func,
                                 const llvm::Twine &msg) {
  static bool syncModeEnabled = DeviceContext::configAsBool("sync-mode", false);

  llvm::StringLiteral kErrorMessage = "";

  if (!syncModeEnabled)
    kErrorMessage = "\nTo get more accurate error information, set "
                    "MODULAR_DEVICE_CONTEXT_SYNC_MODE=true.";

  static bool logVerbose = DeviceContext::configAsBool("log_verbose", false);
  if (logVerbose) {
    return Error(llvm::Twine(file) + ":" + llvm::Twine(line) + " - " + func +
                 " failed: " + msg + kErrorMessage);
  }
  return Error(msg + kErrorMessage);
}

// TODO: Remove once users have been updated to use
// `Device::preferredAlignment`.
MODULAR_CXX_EXPORT size_t getPreferredAlignment(std::string label) {
  return Device::preferredAlignment(label);
}

//
// ===== Device ===== //
//
Device::Device(API api, std::string name, int id, MemInfo memInfo,
               size_t allocAlignment, int64_t deviceMemoryLimit)
    : name_(std::move(name)), id_(id), api_(api),
      memoryManager_(MemoryManager::onDevice(*this, memInfo, allocAlignment,
                                             deviceMemoryLimit)),
      hostMemoryManager_(MemoryManager::onHost(*this, allocAlignment)) {}

Device::~Device() {
  // Destroy the function map before destroying the context.
  funcMap_.clear();
}

MODULAR_CXX_EXPORT const std::string &Device::name() const { return name_; }

ErrorOr<const char *> Device::getArchName() const {
  return Error(Twine("Not implemented for device: ") + Twine(name_));
}

MODULAR_CXX_EXPORT int Device::id() const { return id_; }

MODULAR_CXX_EXPORT Device::API Device::api() const { return api_; }

ErrorOr<bool> Device::supportsMulticast() const { return false; }

MODULAR_CXX_EXPORT size_t Device::allocationAlignment() const {
  return memoryManager_->allocationAlignment();
}

MODULAR_CXX_EXPORT size_t Device::preferredAlignment(StringRef api) {
  if (api == gpuAPI) {
    // TODO(GEX-1416): Remove special hard-coded value to pass tests.
    return 256;
  }
  ErrorOr<DeviceRef> device = create(api, 0);
  if (device.isError()) {
    return M::kPreferredMemoryAlignment;
  }
  return device.get()->allocationAlignment();
}

MODULAR_CXX_EXPORT ErrorOr<DeviceRef>
Device::create(StringRef api, int id, int64_t deviceMemoryLimit) {
  // Check if virtual device mode is enabled. If so, create a VirtualDevice
  // instead of a real hardware device.
  if (isVirtualDeviceMode() &&
      (api == gpuAPI || api == cudaAPI || api == hipAPI || api == metalAPI)) {
    // In virtual device mode, create a lightweight compilation device
    // that can generate code without requiring physical GPU hardware.

    // Get the configured API and target architecture from global config
    std::string virtualAPI = getVirtualDeviceAPI();
    std::string targetArch = getVirtualDeviceTargetArch();

    // Validate that both have been set
    if (virtualAPI.empty()) {
      return Error(
          "Virtual device API not set. "
          "Call setVirtualDeviceAPI() with target API "
          "(e.g., 'cuda', 'hip', 'metal') before creating virtual devices.");
    }
    if (targetArch.empty()) {
      return Error(
          "Virtual device target architecture not set. "
          "Call setVirtualDeviceTargetArch() with target GPU architecture "
          "(e.g., 'sm_80', 'sm_90') before creating virtual devices.");
    }

    ErrorOr<std::unique_ptr<CompilationDevice>> compilationDevice =
        CompilationDevice::create(virtualAPI, targetArch);
    if (compilationDevice.isError())
      return compilationDevice.takeError();

    // Create VirtualDevice with the compilation device
    return VirtualDeviceRef::create(id, "VirtualGPU-" + std::to_string(id),
                                    std::move(*compilationDevice),
                                    deviceMemoryLimit);
  }

  // Only zero and positive values are allowed as device ids.
  if (id < 0) {
    RETURN_ERROR("Device id must be positive. Got " + llvm::Twine(id));
  }
  // Make sure that we are not requesting a device that is out of range. If
  // numberOfDevices(api) returned a zero, then this is an unsupported API and
  // we let the code below issue the correct error.
  int numDevices = numberOfDevices(api);
  if ((numDevices != 0) && (id >= numDevices)) {
    RETURN_ERROR("Device(api: " + api + ", id: " + llvm::Twine(id) +
                 ") is not available. Maximum number of supported devices is " +
                 llvm::Twine(numDevices) + ".");
  }

  // TODO(iposva): Implement dynamic registration mechanism.
  std::string cudaError;
  if ((api == gpuAPI) || (api == cudaAPI)) {
    ErrorOr<const Device &> device = CUDADevice::get(id, deviceMemoryLimit);
    if (!device.isError()) {
      return DeviceRef::copy(&device.get());
    } else if (api == cudaAPI)
      return device.takeError();
    cudaError = device.getError();
  }

  std::string hipError;
  if ((api == gpuAPI) || (api == hipAPI)) {
    ErrorOr<const Device &> device = HIPDevice::get(id, deviceMemoryLimit);
    if (!device.isError()) {
      // This disables full buffering when piping to a file if running
      // on AMD GPU. Without setting this `FileCheck` will have CPU
      // prints and GPU kernel prints mixed together, causing the tests
      // to fail.
      setbuf(stdout, nullptr);
      return DeviceRef::copy(&device.get());
    } else if (api == hipAPI)
      return device.takeError();
    hipError = device.getError();
  }

  std::string metalError;
  if ((api == gpuAPI) || (api == metalAPI)) {
#if MODULAR_MACOS

    ErrorOr<const MetalDevice &> device =
        MetalDevice::get(id, deviceMemoryLimit);
    if (!device.isError()) {
      return DeviceRef::copy(&device.get());
    } else if (api == metalAPI)
      return device.takeError();
    metalError = device.getError();
#endif
  }

  if (api == cpuAPI) {
    ErrorOr<const CpuDevice &> device = CpuDevice::get(id, deviceMemoryLimit);
    if (device.isError())
      return device.takeError();
    return DeviceRef::copy(&device.get());
  }

  if (api == gpuAPI) {
    return Error(llvm::Twine("No supported \"gpu\" device available.\n") +
                 "  CUDA information: " + cudaError + "\n" +
                 "  HIP information:  " + hipError + "\n" +
                 "  Metal information: " + metalError);
  }
  return Error(llvm::Twine("Unsupported device api: \"") + api + "\"");
}

DeviceFunctionCacheEntryRef
Device::findInFunctionCache(const FunctionCacheKey &key) const {
  TRACY_ZONE_SCOPED_NC("Device::findInFunctionCache", TRACY_COLOR_PURPLE);
  std::scoped_lock<std::mutex> lock(mu_);
  return funcMap_.lookup(key);
}

DeviceFunctionCacheEntryRef
Device::insertInFunctionCache(const FunctionCacheKey &key,
                              const DeviceFunctionCacheEntry &entry) const {
  TRACY_ZONE_SCOPED_NC("Device::insertInFunctionCache", TRACY_COLOR_PURPLE);
  {
    std::scoped_lock<std::mutex> lock(mu_);
    auto iter = funcMap_.find(key);
    if (iter != funcMap_.end()) {
      return iter->second;
    }
    funcMap_.insert({key, DeviceFunctionCacheEntryRef::copy(&entry)});
    return DeviceFunctionCacheEntryRef::copy(&entry);
  }
}

void Device::clearFunctionCache() const {
  std::scoped_lock<std::mutex> lock(mu_);
  funcMap_.clear();
}

//
// ===== DeviceBuffer ===== //
//
DeviceBuffer::DeviceBuffer(const DeviceContext &context, void *ptr,
                           size_t byteSize, bool isHost)
    : context_(DeviceContextRef::copy(&context)), ptr_(ptr),
      byteSize_(byteSize), isHost_(isHost) {}
DeviceBuffer::DeviceBuffer(const DeviceContext &context,
                           const DeviceBuffer &parent, void *ptr,
                           size_t byteSize, bool isHost)
    : context_(DeviceContextRef::copy(&context)),
      parent_(DeviceBufferRef::copy(&parent)), ptr_(ptr), byteSize_(byteSize),
      isHost_(isHost) {}
// TODO: Remove the V1 compatibility.
DeviceBuffer::DeviceBuffer(const DeviceContext &context, void *ptr,
                           size_t byteSize, bool isHost, bool owning)
    : context_(DeviceContextRef::copy(&context)), ptr_(ptr),
      byteSize_(byteSize), isHost_(isHost), owning_(owning) {}

MODULAR_CXX_EXPORT DeviceBuffer::~DeviceBuffer() {
  if (!owning_) {
    // TODO: Remove the V1 compatibility.
    // This is a device buffer created by pointer in Mojo while ownership is
    // maintained elsewhere.
    return;
  }
  if ((parent() != nullptr) || (ptr_ == nullptr)) {
    // Parent owns the memory.
    return;
  }
  // Release the GPU memory now or enqueue an event based release for host
  // memory. Host memory can only be released once the stream of execution on
  // the corresponding device has completed all currently pending events.
  const DeviceContext &ctx = context();
  // DEBUGGING AID
  // RECORD_PENDING_ERROR(ctx, ctx.enqueueSetMemory(*this, 0xf3, 1));
  // \DEBUGGING AID
  RECORD_PENDING_ERROR(ctx, ctx.pollPendingEvents());
  if (!isHost()) {
    RECORD_PENDING_ERROR(ctx, ctx.freeList().free(ptr_));
  } else {
    RECORD_PENDING_ERROR(
        ctx, ctx.enqueueEventHandler(
                 [contextRef = DeviceContextRef::copy(&ctx), ptr = ptr_]() {
                   RECORD_PENDING_ERROR(*contextRef,
                                        contextRef->hostFreeList().free(ptr));
                 }));
  }
}

MODULAR_CXX_EXPORT const DeviceContext &DeviceBuffer::context() const {
  return *context_;
}

const DeviceBuffer *DeviceBuffer::parent() const {
  return parent_.getPointer();
}

const DeviceEvent *DeviceBuffer::manualEvent() const {
  return manualEvent_.getPointer();
}

MODULAR_CXX_EXPORT void *DeviceBuffer::ptr() const { return ptr_; }

// TODO: Remove the V1 compatibility.
MODULAR_CXX_EXPORT void *DeviceBuffer::releasePtr() const {
  auto tmp = ptr_;
  ptr_ = nullptr;
  owning_ = false;
  return tmp;
}

MODULAR_CXX_EXPORT size_t DeviceBuffer::byteSize() const { return byteSize_; }

MODULAR_CXX_EXPORT bool DeviceBuffer::isHost() const { return isHost_; }

MODULAR_CXX_EXPORT ErrorOrSuccess DeviceBuffer::disableAutoSync() const {
  if (manualEvent_)
    RETURN_ERROR("Buffer synchronization already disabled");

  DEFINE_OR_RETURN_ERROR(auto, event, context_->eventCreate(0));
  manualEvent_ = std::move(event);

  return success();
}

MODULAR_CXX_EXPORT ErrorOrSuccess DeviceBuffer::markAsReady() const {
  if (!manualEvent_)
    RETURN_ERROR(
        "Invalid use of markAsReady on buffer not with auto-sync enabled");

  return this->context().enqueueEvent(*manualEvent_);
}

MODULAR_CXX_EXPORT ErrorOrSuccess DeviceBuffer::copyTo(
    const DeviceBuffer &dst, size_t srcByteOffset, size_t srcByteSize) const {
  const DeviceBuffer &src = *this;
  return dst.context().enqueueCopyDeviceToDevice(dst, src, srcByteOffset,
                                                 srcByteSize);
}

MODULAR_CXX_EXPORT ErrorOrSuccess DeviceBuffer::copyTo(void *dst) const {
  const DeviceBuffer &src = *this;
  return src.context().enqueueCopyFromDevice(dst, src);
}

MODULAR_CXX_EXPORT ErrorOrSuccess
DeviceBuffer::copyFrom(const DeviceBuffer &src) const {
  const DeviceBuffer &dst = *this;
  return dst.context().enqueueCopyDeviceToDevice(dst, src);
}

MODULAR_CXX_EXPORT ErrorOrSuccess
DeviceBuffer::copyFrom(const void *src) const {
  const DeviceBuffer &dst = *this;
  return dst.context().enqueueCopyToDevice(dst, src);
}

MODULAR_CXX_EXPORT ErrorOrSuccess
DeviceBuffer::reassignOwnershipTo(const DeviceContext &ctx) const {
  if ((context().device().api() != ctx.device().api()) ||
      (context().device().id() != ctx.device().id())) {
    RETURN_ERROR("cannot reassign a buffer between different devices");
  }
  DEFINE_OR_RETURN_ERROR(DeviceEventRef, srcEvent, context().enqueueEvent());
  RETURN_IF_ERROR(ctx.enqueueWaitForEvent(*srcEvent));

  context_ = DeviceContextRef::copy(&ctx);
  return success();
}

MODULAR_CXX_EXPORT ErrorOr<DeviceBufferRef>
DeviceBuffer::createSubBuffer(size_t byteOffset, size_t byteSize) const {
  char *origPtr = static_cast<char *>(this->ptr());
  size_t origByteSize = this->byteSize();

  if ((byteSize > origByteSize) || (origByteSize - byteSize < byteOffset)) {
    RETURN_ERROR("offset, len and elem_size are out of range");
  }
  return DeviceBufferRef::take(new DeviceBuffer(
      context(), *this, origPtr + byteOffset, byteSize, isHost()));
}

MODULAR_CXX_EXPORT ErrorOrSuccess
DeviceBuffer::synchronize(void *external_stream) const {
  // Regular synchronization
  if (!external_stream) {
    if (manualEvent_)
      return manualEvent_->synchronize();
    else
      return context_->synchronize();
  }

  // Synchronization with an external stream
  DEFINE_OR_RETURN_ERROR(DeviceStreamRef, extStream,
                         context_->createExternalStream(external_stream));
  if (manualEvent_)
    // Enqueue a wait for the manual event on the external stream
    RETURN_IF_ERROR(extStream->enqueueWaitForEvent(*manualEvent_));
  else {
    // Create an event on the buffer's stream
    DEFINE_OR_RETURN_ERROR(DeviceEventRef, event, context_->enqueueEvent());
    // Enqueue a wait for the event on the external stream
    RETURN_IF_ERROR(extStream->enqueueWaitForEvent(*event));
  }

  return success();
}

//
// ===== DeviceMulticastBuffer ===== //
//
DeviceMulticastBuffer::DeviceMulticastBuffer(size_t byteSize)
    : byteSize_(byteSize) {}

DeviceMulticastBuffer::~DeviceMulticastBuffer() {}

size_t DeviceMulticastBuffer::byteSize() const { return byteSize_; }

ErrorOr<DeviceMulticastBufferRef>
DeviceMulticastBuffer::allocate(const std::vector<DeviceContextRef> &contexts,
                                size_t byteSize) {
  std::vector<const Device *> devices;
  for (const auto &ctx : contexts) {
    if (!devices.empty()) {
      if (devices[0]->api() != ctx->device().api()) {
        RETURN_ERROR("All contexts need to use the same API.");
      }
    }
    // Check for duplicate entries.
    for (const auto &dev : devices) {
      if (dev->id() == ctx->device().id()) {
        RETURN_ERROR("Duplicate context on a device.");
      }
    }
    // Check that multicast is supported on this particular device.
    DEFINE_OR_RETURN_ERROR(bool, multicastSupported,
                           ctx->device().supportsMulticast());
    if (!multicastSupported)
      RETURN_ERROR("Multicast buffer not supported.");

    // Add to the set of devices participating in this multicast allocation.
    devices.emplace_back(&ctx->device());
  }
  if (devices.size() < 2) {
    RETURN_ERROR("At least two devices necessary.");
  }
  if (devices[0]->api() == CUDADevice::api) {
    return CUDADeviceMulticastBuffer::allocate(devices, byteSize);
  }
  RETURN_ERROR("Multicast only supported on CUDA devices.");
}

//
// ===== DeviceEvent ===== //
//
DeviceEvent::DeviceEvent(const Device &device)
    : device_(DeviceRef::copy(&device)) {}

MODULAR_CXX_EXPORT const Device &DeviceEvent::device() const {
  return *device_;
}

//
// ===== DeviceGraphCollection ===== //
//
void DeviceGraphCollection::setOutputs(
    ArrayRef<AsyncRT::AnyAsyncValueRef> values) {
  ASSERT_STREAM(outputs.empty(), "DeviceGraphCollection already has values");
  outputs = llvm::map_to_vector(values, [](auto &ref) { return ref.copy(); });
}

SmallVector<AsyncRT::AnyAsyncValueRef>
DeviceGraphCollection::getOutputs() const {
  return llvm::map_to_vector(outputs, [](auto &ref) { return ref.copy(); });
}

ErrorOrSuccess DeviceGraphCollection::replay() const {
  for (const DeviceGraphRef &graph : graphs)
    if (auto err = graph->replay())
      return err;

  return success();
}

int64_t DeviceGraphCollection::numGraphs() const {
  return static_cast<int64_t>(graphs.size());
}

/// RAII type for managing the capture state of a stream.
class DeviceGraphCaptureContext {
public:
  DeviceGraphCaptureContext(const DeviceGraphCaptureContext &other) = delete;
  DeviceGraphCaptureContext(DeviceGraphCaptureContext &&other) = default;

  /// Initiate capture on a particular stream.
  static ErrorOr<DeviceGraphCaptureContext>
  beginCapture(const DeviceStreamRef &stream) {
    if (auto err = stream->beginCapture())
      return err.takeError();
    return DeviceGraphCaptureContext(stream);
  }

  /// End capture and return the resulting device graph.
  ErrorOr<DeviceGraphRef> endCapture() {
    ErrorOr<DeviceGraphRef> result = stream->endCapture();

    if (result.isError()) {
      stream.reset();
      return result.takeError();
    }

    result.get()->setStream(std::move(stream));
    return result;
  }

  ~DeviceGraphCaptureContext() {
    if (stream)
      (void)stream->endCapture();
  }

private:
  DeviceGraphCaptureContext(const DeviceStreamRef &ref) : stream(ref.copy()) {}

  /// The stream on which capture is occurring (can be nullptr).
  DeviceStreamRef stream;
};

//
// ===== DeviceGraph ===== //
//
ErrorOr<DeviceGraphCollection>
DeviceGraph::capture(ArrayRef<DeviceStreamRef> streams,
                     llvm::function_ref<ErrorOrSuccess()> execute) {
  SmallVector<DeviceGraphCaptureContext> captures;
  captures.reserve(streams.size());

  // Begin capture on all streams.
  for (const DeviceStreamRef &stream : streams) {
    auto capture = DeviceGraphCaptureContext::beginCapture(stream);
    if (capture.isError())
      return capture.takeError();

    captures.push_back(capture.takeValue());
  }

  ErrorOrSuccess execResult = execute();

  if (execResult.isError())
    return execResult.takeError();

  SmallVector<DeviceGraphRef> results;
  results.reserve(streams.size());

  for (DeviceGraphCaptureContext &context : captures) {
    ErrorOr<DeviceGraphRef> graphOr = context.endCapture();
    if (graphOr.isError())
      return graphOr.takeError();

    results.push_back(graphOr.takeValue());
  }

  return DeviceGraphCollection(std::move(results));
}

void DeviceGraph::setStream(DeviceStreamRef stream) {
  captureStream_ = std::move(stream);
}

//
// ===== DeviceStream ===== //
//
DeviceStream::DeviceStream(const DeviceContext &context)
    : ctx_(DeviceContextRef::copy(&context)) {}

const Device &DeviceStream::device() const { return ctx_->device(); }

ErrorOrSuccess DeviceStream::beginCapture() const {
  RETURN_ERROR("graph capture not supported on this device");
}

ErrorOr<DeviceGraphRef> DeviceStream::endCapture() const {
  RETURN_ERROR("graph capture not supported on this device");
}

//
// ===== DeviceFunction ===== //
//
enum DeviceFunction::ArgumentType : uint8_t {
  UNSET = 0, // This entry must be zero and first.
  UINT8,
  UINT16,
  UINT32,
  UINT64,
  INT8,
  INT16,
  INT32,
  INT64,
  FP4,
  FP8,
  FP16,
  FP32,
  FP64,
  BUFFER,
  NUM_ENTRIES_ // This entry must be last.
};

DeviceFunction::DeviceFunction(
    const DeviceContext &context,
    const DeviceFunction::SignatureBuilder &signature)
    : context_(DeviceContextRef::copy(&context)), signature_(signature.types_) {
}

const DeviceContext &DeviceFunction::context() const { return *context_; }

size_t DeviceFunction::argumentCount() const { return signature_.size(); }

//
// ===== DeviceFunction::SignatureBuilder ===== //
//
MODULAR_CXX_EXPORT DeviceFunction::SignatureBuilder &
DeviceFunction::SignatureBuilder::addUInt64() {
  types_.push_back(UINT64);
  return *this;
}

MODULAR_CXX_EXPORT DeviceFunction::SignatureBuilder &
DeviceFunction::SignatureBuilder::addBuffer() {
  types_.push_back(BUFFER);
  return *this;
}

//
// ===== DeviceTimer ===== //
//
DeviceTimer::DeviceTimer(const DeviceContext &context)
    : context_(DeviceContextRef::copy(&context)) {}

const DeviceContext &DeviceTimer::context() const { return *context_; }

//
// ===== DeviceContext ===== //
//

// Mutex to protect shared, static data. E.g. config dictionary
static std::mutex &getGlobalMutex() {
  static std::mutex mu;
  return mu;
}

DeviceContext::DeviceContext(const Device &device)
    : device_(DeviceRef::copy(&device)),
      freeList_(std::make_unique<MemoryManagerFreeList>(
          this, device.memoryManager(), device.memoryManager().freeList())),
      hostFreeList_(std::make_unique<MemoryManagerFreeList>(
          this, device.hostMemoryManager(),
          device.hostMemoryManager().freeList())),
      syncMode_(0) {
  if (configAsBool("sync-mode", false)) {
    incSyncMode();
  }
}

DeviceContext::~DeviceContext() {
  // Key function for vtable generation.
}

MODULAR_CXX_EXPORT const Device &DeviceContext::device() const {
  return *device_;
}

ErrorOrSuccess DeviceContext::setAsCurrent() const {
  // If this context is already bound, the driver context is already set.
  if (currentlyBoundContext == this)
    return success();
  // If a different context is currently bound, unbind it since we're
  // explicitly changing the current context.
  if (currentlyBoundContext != nullptr)
    unbindFromCurrentThread();
  return device().setCurrentDriverContext(driverContext());
}

MODULAR_CXX_EXPORT bool DeviceContext::isBoundToCurrentThread() const {
  return currentlyBoundContext == this;
}

MODULAR_CXX_EXPORT ErrorOrSuccess DeviceContext::bindToCurrentThread() const {
  // No-op if device context is already bound to current thread.
  if (currentlyBoundContext == this) {
    return success();
  }

  // Check if the driver context is already the current context.
  ErrorOr<void *> currentCtx = device().getCurrentDriverContext();
  if (currentCtx.isError()) {
    return currentCtx.takeError();
  }
  if (*currentCtx != driverContext()) {
    // Context is not the current context, so set it.
    ErrorOrSuccess err = device().setCurrentDriverContext(driverContext());
    if (err.isError()) {
      return err;
    }
  }

  currentlyBoundContext = this;
  return success();
}

MODULAR_CXX_EXPORT void DeviceContext::unbindFromCurrentThread() {
  currentlyBoundContext = nullptr;
}

MODULAR_CXX_EXPORT ErrorOr<Device::MemInfo>
DeviceContext::getMemoryInfo() const {
  // If we are using the memory manager, get memory information from it.
  MemoryManager &cache = device().memoryManager();
  if (cache.isEnabled()) {
    return cache.getMemoryInfo();
  }

  // Otherwise, go directly to the device.
  return getDeviceMemoryInfo();
}

void DeviceContext::setPendingError(Error err) const {
  {
    std::lock_guard<std::mutex> lock(pendingErrorMutex_);
    // Do not overwrite an already pending error.
    if (pendingError_.isError())
      return;

    pendingError_ = std::move(err);
  }
  // After releasing pendingErrorMutex_ also set the atomic value to true.
  hasPendingError_.store(1, std::memory_order_release);
}

class DeviceContext::PendingEventHandler {
public:
  PendingEventHandler(DeviceContextRef ctx, DeviceEventRef event,
                      std::function<void()> handler)
      : next_(nullptr), ctx_(ctx), event_(std::move(event)),
        handler_(std::move(handler)) {}

  // Returns true if the event is ready (or if there's no event).
  ErrorOr<bool> isReady() {
    if (!event_)
      return true; // No event means immediately ready
    return event_->isReady();
  }

  void runHandler() { handler_(); }

private:
  friend class DeviceContext;

  PendingEventHandler *next_;
  // We keep a reference to the context to ensure that we do not release the
  // context while we still have pending events in the queue.
  DeviceContextRef ctx_;
  DeviceEventRef event_;
  std::function<void()> handler_;
};

void DeviceContext::setPendingEventHandler(PendingEventHandler *h) const {
  {
    std::lock_guard<std::mutex> lock(pendingWorkMutex_);
    if (pendingEventsLast_ != nullptr) {
      // Enqueue at end.
      assert(pendingEventsLast_->next_ == nullptr);
      pendingEventsLast_->next_ = h;
      pendingEventsLast_ = h;
    } else {
      // Enqueue as first item.
      pendingEvents_ = h;
      pendingEventsLast_ = h;
    }
  }
  // After releasing pendingWorkMutex_ also set the atomic value to true.
  hasPendingWork_.store(1, std::memory_order_release);
}

ErrorOrSuccess DeviceContext::checkPendingError() const {
  TRACY_ZONE_SCOPED_NC("DeviceContext::checkPendingError", TRACY_COLOR_PURPLE);

  // Quickly check if we should check for a pending error.
  if (hasPendingError_.load(std::memory_order_acquire) == 0) {
    return success();
  }

  // We can safely remove the flag indicating that the pending error should be
  // checked. We will unconditionally acquire the lock below and check the
  // pending error while protected by the lock.
  hasPendingError_.store(0, std::memory_order_release);

  {
    std::lock_guard<std::mutex> lock(pendingErrorMutex_);

    // Check if we have a pending error recorded.
    if (!pendingError_.isError())
      return success();

    // Return the error and reset.
    Error result = pendingError_.takeError();
    pendingError_ = success();
    return result;
  }
}

ErrorOrSuccess DeviceContext::pollPendingEvents() const {
  TRACY_ZONE_SCOPED_NC("DeviceContext::pollPendingEvents", TRACY_COLOR_PURPLE);

  // During graph capture, events aren't executing - they're graph nodes.
  // cuEventQuery is prohibited during capture, so skip polling. Handlers
  // will run after capture ends and the graph executes.
  if (isGraphCapturing())
    return success();

  // Quickly check if we should check for a pending work.
  if (hasPendingWork_.load(std::memory_order_acquire) == 0) {
    return success();
  }

  // We can safely remove the flag indicating that the pending work should be
  // checked. We will unconditionally acquire the lock below and check the
  // pending event handlers while protected by the lock.
  hasPendingWork_.store(0, std::memory_order_release);

  {
    std::lock_guard<std::mutex> lock(pendingWorkMutex_);

    // Check if the head of the queue of pending events exists and its event is
    // ready.
    while (pendingEvents_ != nullptr) {
      ErrorOr<bool> isReadyOr = pendingEvents_->isReady();
      if (isReadyOr.isError()) {
        // Since there is at least one pending event handler in the queue we
        // also ensure that we do continue checking for the events to be ready.
        hasPendingWork_.store(1, std::memory_order_release);
        return isReadyOr.takeError();
      }
      if (!*isReadyOr) {
        // First element in the queue is not ready, so we can abort the
        // walk through the pending events.
        // Since there is at least one pending event handler in the queue we
        // also ensure that we do continue checking for the events to be ready.
        hasPendingWork_.store(1, std::memory_order_release);
        break;
      }
      // The handler at the head of the queue is ready to run. Remove the head
      // of the queue from the list of pending handlers.
      // Important: The head of the queue needs to be removed before the lock is
      // released below to avoid racing threads triggering the current handler
      // multiple times.
      PendingEventHandler *curHandler = pendingEvents_;
      pendingEvents_ = pendingEvents_->next_;
      if (pendingEvents_ == nullptr) {
        // We reached the end of the queue.
        pendingEventsLast_ = nullptr;
      }

      // Before calling the handler we need to release the lock in case there
      // are pending errors to be enqueued or there are other effects by the
      // handler code (e.g. causing dead lock or contention).
      pendingWorkMutex_.unlock();
      curHandler->runHandler();
      delete curHandler;
      pendingWorkMutex_.lock();
    }
  }

  return success();
}

MODULAR_CXX_EXPORT ErrorOr<DeviceBufferRef>
DeviceContext::enqueueCreateBuffer(size_t byteSize) const {
  if (byteSize == 0)
    return this->createBufferOwning(nullptr, 0, /*owning=*/true);
  return allocBuffer(byteSize, Device::AllocMode::ASYNC);
}

MODULAR_CXX_EXPORT ErrorOr<DeviceBufferRef>
DeviceContext::createHostBuffer(size_t byteSize) const {
  if (byteSize == 0)
    return this->createBufferOwning(nullptr, 0, true);
  return allocBuffer(byteSize, Device::AllocMode::HOST);
}

MODULAR_CXX_EXPORT DeviceBufferRef DeviceContext::createBufferOwning(
    void *ptr, size_t byteSize, bool owning) const {
  return DeviceBufferRef::take(
      new DeviceBuffer(*this, ptr, byteSize, /* isHost = */ false, owning));
}

MODULAR_CXX_EXPORT DeviceBufferRef DeviceContext::createHostBufferOwning(
    void *ptr, size_t byteSize, bool owning) const {
  return DeviceBufferRef::take(
      new DeviceBuffer(*this, ptr, byteSize, /* isHost = */ true, owning));
}

ErrorOr<DeviceBufferRef>
DeviceContext::allocBuffer(size_t byteSize, Device::AllocMode mode) const {
  assert(byteSize != 0);
  void *ptr = nullptr;

  // When in graph capture mode, we bypass the MemoryManager and allow the
  // graph capture mechanism handle memory allocations.
  if (isGraphCapturing()) {
    if (ErrorOrSuccess err = memAlloc(&ptr, byteSize, mode))
      return err.takeError();
  } else {
    // Allocate from the correct memory manager.
    ErrorOr<void *> errPtr =
        ((mode == Device::AllocMode::HOST) ? hostFreeList() : freeList())
            .allocate(byteSize, mode);
    if (errPtr.isError()) {
      return errPtr.takeError();
    }
    ptr = errPtr.takeValue();
  }
  return DeviceBufferRef::take(
      new DeviceBuffer(*this, ptr, byteSize, mode == Device::AllocMode::HOST));
}

MODULAR_CXX_EXPORT ErrorOrSuccess DeviceContext::enqueueCopyDeviceToDevice(
    const DeviceBuffer &dst, const DeviceBuffer &src, size_t srcByteOffset,
    size_t srcByteSize) const {
  const Device::API srcApi = src.context().device().api();
  const Device::API dstApi = dst.context().device().api();
  if ((srcByteOffset + srcByteSize) < dst.byteSize())
    RETURN_ERROR("not enough data in src");

  if (src.isHost() || (srcApi == Device::cpuAPI)) {
    return dst.context().enqueueCopyToDevice(
        dst.ptr(), (uint8_t *)src.ptr() + srcByteOffset, srcByteSize);
  } else if (dst.isHost() || (dstApi == Device::cpuAPI)) {
    return src.context().enqueueCopyFromDevice(
        dst.ptr(), (uint8_t *)src.ptr() + srcByteOffset, srcByteSize);
  } else if (srcApi == dstApi) {
    return this->memCopyDtoD(dst, src, srcByteOffset, srcByteSize);
  }
  return Error("Device API mismatch copying from \"" + srcApi + "\" to \"" +
               dstApi + "\"");
}

MODULAR_CXX_EXPORT ErrorOrSuccess
DeviceContext::enqueueEventHandler(std::function<void()> handler) const {
  DEFINE_OR_RETURN_ERROR(DeviceEventRef, event, this->enqueueEvent());
  PendingEventHandler *eventHandler = new PendingEventHandler(
      DeviceContextRef::copy(this), std::move(event), std::move(handler));
  setPendingEventHandler(eventHandler);
  return success();
}

MODULAR_CXX_EXPORT
ErrorOr<std::pair<Driver::DeviceBufferRef,
                  std::optional<llvm::sys::fs::mapped_file_region>>>
DeviceContext::memmapFile(const std::filesystem::path &filename) const {
  auto result = this->memmapFileImpl(filename);
  if (result.isError())
    return result.takeError();
  auto [ptr, size, mmap] = result.takeValue();

  return std::pair{
      // Some oddities here:
      // This doesn't own memory, but it seems that functionality is deprecated?
      // Even if we allocate on the host, this fails further up the stack (SDK)
      // if we set isHost=true, since it gets treated as pinned memory (goes
      // down some other code path).
      RCRef<Driver::DeviceBuffer>::take(
          new Driver::DeviceBuffer(*this, ptr, size, /*isHost*/ false,
                                   /*owning*/ false)),
      std::optional{std::move(mmap)},
  };
}

MODULAR_CXX_EXPORT ErrorOr<DeviceContextRef>
DeviceContext::create(StringRef api, int id) {
  ErrorOr<DeviceRef> device = Device::create(api, id);
  if (device.isError()) {
    return device.takeError();
  }
  return (*device)->createContext();
}

MODULAR_CXX_EXPORT ErrorOr<DeviceContextRef>
DeviceContext::getOrCreate(StringRef api, int id, int64_t deviceMemoryLimit) {
  static std::mutex mu;
  static llvm::StringMap<const DeviceContext *> contextMap;

  ErrorOr<DeviceRef> errDev = Device::create(api, id, deviceMemoryLimit);
  if (errDev.isError())
    return errDev.takeError();

  // Create the cache key in such a way that for example an `api` of "gpu" and
  // "cuda" are equivalent on an CUDA based system.
  std::string cacheKey = ((*errDev)->api() + "[" + llvm::Twine(id) + "]").str();
  {
    std::lock_guard<std::mutex> lock(mu);
    auto iter = contextMap.find(cacheKey);
    if (iter != contextMap.end()) {
      const DeviceContext *cachedCtx = iter->second;
      return DeviceContextRef::copy(cachedCtx);
    }
    ErrorOr<DeviceContextRef> errCtx = (*errDev)->createContext();
    if (errCtx.isError())
      return errCtx.takeError();

    const DeviceContext *ctx = errCtx.takeValue().release();
    contextMap.insert({cacheKey, ctx});
    return DeviceContextRef::copy(ctx);
  }
}

MODULAR_CXX_EXPORT bool
DeviceContext::operator==(const DeviceContext &other) const {
  ErrorOr<DeviceStreamRef> errStream = stream();
  if (errStream.isError())
    return false;
  void *streamThis = (*errStream)->ptr();

  errStream = other.stream();
  if (errStream.isError())
    return false;
  void *streamOther = (*errStream)->ptr();

  return streamThis == streamOther;
}

MODULAR_CXX_EXPORT bool
DeviceContext::operator!=(const DeviceContext &other) const {
  return !(*this == other);
}

MemoryManagerFreeList &DeviceContext::freeList() const { return *freeList_; }
MemoryManagerFreeList &DeviceContext::hostFreeList() const {
  return *hostFreeList_;
}

//
// ===== Device ===== //
//
MODULAR_CXX_EXPORT int Device::numberOfDevices(StringRef api) {
  // In virtual device mode, return the virtual device count for GPU APIs.
  // This allows validation checks to pass even when requesting more GPUs
  // than are physically available.
  int virtualCount = getVirtualDeviceCount();
  if (virtualCount > 0 && api == gpuAPI) {
    return virtualCount;
  }

  // TODO(iposva): Implement dynamic registration mechanism.
  if ((api == gpuAPI) || (api == cudaAPI)) {
    ErrorOr<int> count = CUDADevice::numberOfGPUs();
    if (!count.isError())
      return *count;
  }
  if ((api == gpuAPI) || (api == hipAPI)) {
    ErrorOr<int> count = HIPDevice::numberOfGPUs();
    if (!count.isError())
      return *count;
  }
  if ((api == gpuAPI) || (api == metalAPI)) {
#if MODULAR_MACOS
    ErrorOr<int> count = MetalDevice::numberOfGPUs();
    if (!count.isError())
      return *count;
#endif
  }
  if (api == cpuAPI) {
    // For CPU we go back to the device to get the number of supported devices.
    ErrorOr<const CpuDevice &> device = CpuDevice::get(0);
    if (!device.isError())
      return device->deviceCount();
  }

  return 0;
}

MemoryManager &Device::memoryManager() const {
  assert(memoryManager_ != nullptr);
  return *memoryManager_;
}

MemoryManager &Device::hostMemoryManager() const {
  assert(hostMemoryManager_ != nullptr);
  return *hostMemoryManager_;
}

bool DeviceContext::syncMode() const {
  return syncMode_.load(std::memory_order_relaxed) > 0;
}
void DeviceContext::incSyncMode() const {
  syncMode_.fetch_add(1, std::memory_order_relaxed);
}
void DeviceContext::decSyncMode() const {
  syncMode_.fetch_add(-1, std::memory_order_relaxed);
}

bool DeviceContext::isGraphCapturing() const {
  if (device().api() == Device::cudaAPI) {
    const auto &cudaCtx = CUDADeviceContext::cast(*this);
    CUstreamCaptureStatus status =
        CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    if (cudaCtx.cudaDriver().cuStreamIsCapturing(
            cudaCtx.cudaStream(), &status) == CUDADriver::SUCCESS &&
        status == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE) {
      return true;
    }
  }

  return false;
}

bool DeviceContext::configAsBool(StringRef key, bool defaultValue,
                                 bool *isDefault) {
  std::lock_guard<std::mutex> lock(getGlobalMutex());
  ErrorOr<Config> &config = getConfig();
  if (config.isError()) {
    if (isDefault != nullptr) {
      *isDefault = true;
    }
    return defaultValue;
  }

  if (isDefault != nullptr) {
    StringRef s =
        config->getValue((llvm::Twine("device-context.") + key).str());
    *isDefault = s.empty();
  }
  return config->getValueAsBool((llvm::Twine("device-context.") + key).str(),
                                defaultValue);
}

int64_t DeviceContext::configAsInt(StringRef key, int64_t defaultValue,
                                   bool *isDefault) {
  std::lock_guard<std::mutex> lock(getGlobalMutex());
  ErrorOr<Config> &config = getConfig();
  if (config.isError()) {
    if (isDefault != nullptr) {
      *isDefault = true;
    }
    return defaultValue;
  }

  StringRef s = config->getValue((llvm::Twine("device-context.") + key).str());
  if (isDefault != nullptr) {
    *isDefault = s.empty();
  }
  long long result = defaultValue;
  if (s.getAsInteger(10, result)) {
    if (isDefault != nullptr) {
      *isDefault = true;
    }
    return defaultValue;
  }
  return result;
}

StringRef DeviceContext::configAsString(StringRef key, StringRef defaultValue,
                                        bool *isDefault) {
  std::lock_guard<std::mutex> lock(getGlobalMutex());
  ErrorOr<Config> &config = getConfig();
  if (config.isError()) {
    if (isDefault != nullptr) {
      *isDefault = true;
    }
    return defaultValue;
  }

  if (isDefault != nullptr) {
    StringRef s =
        config->getValue((llvm::Twine("device-context.") + key).str());
    *isDefault = s.empty();
  }
  return config->getValueOr((llvm::Twine("device-context.") + key).str(),
                            defaultValue);
}

bool DeviceContext::isDebugAllocatorEnabled(StringRef name) const {
  std::lock_guard<std::mutex> lock(getGlobalMutex());
  ErrorOr<Config> &config = getConfig();
  return config->isValueInList("debug.device_allocator", name);
}

ErrorOr<Config> &DeviceContext::getConfig() {
  static ErrorOr<Config> configuration = Config::open();
  return configuration;
}

// Maps a device API name to its vendor prefix for accelerator arch strings.
static StringRef apiToVendorPrefix(StringRef api) {
  if (api == "cuda")
    return "nvidia";
  if (api == "hip")
    return "amdgpu";
  if (api == "metal")
    return "metal";
  return api;
}

MODULAR_CXX_EXPORT const std::string &Device::getAcceleratorArchOrEmpty() {
  static std::string arch = [] {
    // First check if we're in virtual device mode (cross-compilation).
    // Virtual device config is set at startup and takes precedence over
    // physical hardware detection.
    std::string virtualTargetArch = getVirtualDeviceTargetArch();
    if (!virtualTargetArch.empty()) {
      std::string virtualAPI = getVirtualDeviceAPI();
      return (llvm::Twine(apiToVendorPrefix(virtualAPI)) + ":" +
              virtualTargetArch)
          .str();
    }

    // Otherwise, detect from physical hardware
    std::string vendor("nvidia");
    ErrorOr<std::string> archName = CUDADevice::getFirstArchName();

    if (archName.isError()) {
      vendor = "amdgpu";
      archName = HIPDevice::getArchName(0);
    }

#if MODULAR_MACOS
    if (archName.isError()) {
      vendor = "metal";
      archName = MetalDevice::getArchName(nullptr);
    }
#endif

    if (archName.isError())
      return std::string();

    return (llvm::Twine(vendor) + ":" + archName.get()).str();
  }();
  return arch;
}

//
// ===== DriverVersion ===== //
//

DriverVersion::DriverVersion(std::string version) : version_(version) {
  std::sscanf(version.c_str(), "%d.%d.%d", &major_, &minor_, &patch_);
}

int DriverVersion::major() const { return major_; }

int DriverVersion::minor() const { return minor_; }

int DriverVersion::patch() const { return patch_; }

MODULAR_CXX_EXPORT std::string DriverVersion::toString() const {
  return version_;
}

ErrorOrSuccess Device::enablePeerAccess(const Device &peer) const {
  RETURN_ERROR("enablePeerAccess not implemented for this device API" +
               Twine(peer.api()));
}
} // namespace M::Driver
