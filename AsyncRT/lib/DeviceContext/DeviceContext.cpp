//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CPU device context.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/DeviceContext/DeviceContext.h"

#include "BufferCache.h"
#include "CPU/CpuDeviceContext.h"
#include "CUDA/CUDACompilationDevice.h"
#include "CUDA/CUDADeviceContext.h"
#include "HIP/HIPCompilationDevice.h"
#include "HIP/HIPDeviceContext.h"
#include "Metal/MetalDeviceContext.h"
#include "Support/AlignedAlloc.h"
#include "Support/ErrorOr.h"
#include "Support/SymbolExport.h"
#include "Virtual/VirtualDeviceContext.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Process.h"

#include <atomic>
#include <cstdio>
#include <mutex>
#include <string.h>

#define ERR_CHECK_ERR(err_call)                                                \
  do {                                                                         \
    auto errResult = (err_call);                                               \
    if (errResult.isError()) {                                                 \
      return DeviceContext::formatError(__FILE__, __LINE__, __FUNCTION__,      \
                                        errResult.getError());                 \
    }                                                                          \
  } while (0)

#define ERR_CHECK(ctx, err_call)                                               \
  do {                                                                         \
    auto errResult = (err_call);                                               \
    if (errResult.isError()) {                                                 \
      M::Error err = DeviceContext::formatError(                               \
          __FILE__, __LINE__, __FUNCTION__, errResult.getError());             \
      /* We are not in a function where we can return an Error here. So we     \
       * remember the error in the context and will report at a later time.    \
       */                                                                      \
      (ctx).setPendingError(std::move(err));                                   \
    }                                                                          \
  } while (0)

#define RAISE_ERROR_MESSAGE(msg)                                               \
  do {                                                                         \
    return DeviceContext::formatError(__FILE__, __LINE__, __FUNCTION__, msg);  \
  } while (0)

#define DEFINE_OR_RAISE(type, var, call)                                       \
  auto _err_or_##var = call;                                                   \
  if (_err_or_##var.isError()) {                                               \
    return DeviceContext::formatError(__FILE__, __LINE__, __FUNCTION__,        \
                                      _err_or_##var.getError());               \
  }                                                                            \
  type var = _err_or_##var.takeValue()

namespace M::AsyncRT {

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
      bufferCache_(BufferCache::onDevice(*this, memInfo, allocAlignment,
                                         deviceMemoryLimit)),
      hostBufferCache_(BufferCache::onHost(*this, allocAlignment)) {}

Device::~Device() {
  // Destroy the function map before destroying the context.
  funcMap_.clear();
}

// DeviceFunction compiling.
ErrorOr<M::BufferRef>
Device::compileFunction(StringRef ir, StringRef debugLevel,
                        int32_t optimizationLevel,
                        StringRef targetAccelerator) const {
  const CompilationDevice *compDev = compilationDevice();
  if (!compDev) {
    return Error("No compilation device available for this device");
  }

  return compDev->compileFunction(ir, debugLevel, optimizationLevel,
                                  targetAccelerator);
}
MODULAR_CXX_EXPORT const std::string &Device::name() const { return name_; }

ErrorOr<std::string> Device::getArchName() const {
  return Error(Twine("Not implemented for device: ") + name());
}

MODULAR_CXX_EXPORT int Device::id() const { return id_; }

MODULAR_CXX_EXPORT Device::API Device::api() const { return api_; }

MODULAR_CXX_EXPORT StringRef Device::vendor() const {
  if (api() == cudaAPI)
    return "nvidia";
  if (api() == hipAPI)
    return "amdgpu";
  if (api() == cpuAPI)
    return "cpu";
  if (api() == metalAPI)
    return "metal";
  return "";
}

ErrorOr<bool> Device::supportsMulticast() const { return false; }

MODULAR_CXX_EXPORT size_t Device::allocationAlignment() const {
  return bufferCache_->allocationAlignment();
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
    RAISE_ERROR_MESSAGE("Device id must be positive. Got " + llvm::Twine(id));
  }
  // Make sure that we are not requesting a device that is out of range. If
  // numberOfDevices(api) returned a zero, then this is an unsupported API and
  // we let the code below issue the correct error.
  int numDevices = numberOfDevices(api);
  if ((numDevices != 0) && (id >= numDevices)) {
    RAISE_ERROR_MESSAGE(
        "Device(api: " + api + ", id: " + llvm::Twine(id) +
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
#ifdef MODULAR_MACOS

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

DeviceFunctionCacheEntryRef Device::findInFunctionCache(StringRef key) const {
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto iter = funcMap_.find(key);
    if (iter != funcMap_.end()) {
      return iter->second;
    }
  }
  return DeviceFunctionCacheEntryRef();
}

DeviceFunctionCacheEntryRef
Device::insertInFunctionCache(StringRef key,
                              const DeviceFunctionCacheEntry &entry) const {
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto iter = funcMap_.find(key);
    if (iter != funcMap_.end()) {
      return iter->second;
    }
    funcMap_.insert({key, DeviceFunctionCacheEntryRef::copy(&entry)});
    return DeviceFunctionCacheEntryRef::copy(&entry);
  }
}

void Device::clearFunctionCache() const {
  std::lock_guard<std::mutex> lock(mu_);
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
  // ERR_CHECK(ctx, ctx.enqueueSetMemory(*this, 0xf3, 1));
  // \DEBUGGING AID
  ERR_CHECK(ctx, ctx.checkPendingWork());
  if (!isHost()) {
    ERR_CHECK(ctx, ctx.freeList().free(ptr_));
  } else {
    ERR_CHECK(
        ctx, ctx.enqueueEventHandler(
                 [contextRef = DeviceContextRef::copy(&ctx), ptr = ptr_]() {
                   ERR_CHECK(*contextRef, contextRef->hostFreeList().free(ptr));
                 }));
  }
}

MODULAR_CXX_EXPORT const DeviceContext &DeviceBuffer::context() const {
  return *context_;
}

const DeviceBuffer *DeviceBuffer::parent() const {
  return parent_.getPointer();
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

MODULAR_CXX_EXPORT ErrorOrSuccess
DeviceBuffer::copyTo(const DeviceBuffer &dst) const {
  const DeviceBuffer &src = *this;
  return dst.context().enqueueCopyDeviceToDevice(dst, src);
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
    RAISE_ERROR_MESSAGE("cannot reassign a buffer between different devices");
  }
  DEFINE_OR_RAISE(DeviceEventRef, srcEvent, context().enqueueEvent());
  ERR_CHECK_ERR(ctx.enqueueWaitForEvent(*srcEvent));

  context_ = DeviceContextRef::copy(&ctx);
  return success();
}

MODULAR_CXX_EXPORT ErrorOr<DeviceBufferRef>
DeviceBuffer::createSubBuffer(size_t byteOffset, size_t byteSize) const {
  char *origPtr = static_cast<char *>(this->ptr());
  size_t origByteSize = this->byteSize();

  if ((byteSize > origByteSize) || (origByteSize - byteSize < byteOffset)) {
    RAISE_ERROR_MESSAGE("offset, len and elem_size are out of range");
  }
  return DeviceBufferRef::take(new DeviceBuffer(
      context(), *this, origPtr + byteOffset, byteSize, isHost()));
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
        RAISE_ERROR_MESSAGE("All contexts need to use the same API.");
      }
    }
    // Check for duplicate entries.
    for (const auto &dev : devices) {
      if (dev->id() == ctx->device().id()) {
        RAISE_ERROR_MESSAGE("Duplicate context on a device.");
      }
    }
    // Check that multicast is supported on this particular device.
    DEFINE_OR_RAISE(bool, multicastSupported,
                    ctx->device().supportsMulticast());
    if (!multicastSupported)
      RAISE_ERROR_MESSAGE("Multicast buffer not supported.");

    // Add to the set of devices participating in this multicast allocation.
    devices.emplace_back(&ctx->device());
  }
  if (devices.size() < 2) {
    RAISE_ERROR_MESSAGE("At least two devices necessary.");
  }
  if (devices[0]->api() == CUDADevice::api) {
    return CUDADeviceMulticastBuffer::allocate(devices, byteSize);
  }
  RAISE_ERROR_MESSAGE("Multicast only supported on CUDA devices.");
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
// ===== DeviceStream ===== //
//
DeviceStream::DeviceStream(const Device &device)
    : device_(DeviceRef::copy(&device)) {}

const Device &DeviceStream::device() const { return *device_; }

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
  _NUM_ENTRIES_ // This entry must be last.
};

const char *DeviceFunction::typeName(ArgumentType argType) {
  const char *typeNames[] = {"unset",   "uint8",   "uint16",      "uint32",
                             "uint64",  "int8",    "int16",       "int32",
                             "int64",   "float4",  "float8",      "float16",
                             "float32", "float64", "DeviceBuffer"};
  static_assert(sizeof(typeNames) ==
                (static_cast<int>(_NUM_ENTRIES_) * sizeof(*typeNames)));
  int index = static_cast<int>(argType);
  return ((index >= 0) && (index < _NUM_ENTRIES_)) ? typeNames[index]
                                                   : "unknown";
}

DeviceFunction::DeviceFunction(
    const DeviceContext &context,
    const DeviceFunction::SignatureBuilder &signature)
    : context_(DeviceContextRef::copy(&context)), signature_(signature.types_) {
}

const DeviceContext &DeviceFunction::context() const { return *context_; }

size_t DeviceFunction::argumentCount() const { return signature_.size(); }

ErrorOrSuccess DeviceFunction::checkSignatureAt(int index,
                                                ArgumentType argType) const {
  int expectedSize = signature_.size();
  if ((index < 0) || index >= expectedSize) {
    return Error("Argument count mismatch at " + Twine(index + 1) +
                 "th position: Expected " + Twine(expectedSize) +
                 " arguments.");
  }
  ArgumentType expectedType = signature_.at(index);
  if (expectedType != argType) {
    return Error(llvm::Twine("Argument mismatch at ") + Twine(index + 1) +
                 "th position: Expected " + typeName(expectedType) + ", got " +
                 typeName(argType) + ".");
  }
  return success();
}

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
// ===== DeviceFunction::Arguments ===== //
//
DeviceFunction::Arguments::Arguments(const DeviceFunction &fn)
    : fn_(DeviceFunctionRef::copy(&fn)) {}

bool DeviceFunction::Arguments::hasError() const { return err_.isError(); }

const ErrorOrSuccess &DeviceFunction::Arguments::getError() const {
  return err_;
}

MODULAR_CXX_EXPORT void
DeviceFunction::Arguments::checkArg(int index, const uint64_t &val) {
  if (err_)
    return;
  err_ = fn_->checkSignatureAt(index, UINT64);
  pushArg(index, val);
}

MODULAR_CXX_EXPORT void
DeviceFunction::Arguments::checkArg(int index, const DeviceBuffer &val) {
  if (err_)
    return;
  err_ = fn_->checkSignatureAt(index, BUFFER);
  pushArg(index, val);
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
      freeList_(std::make_unique<BufferCacheFreeList>(
          this, device.bufferCache(), device.bufferCache().freeList())),
      hostFreeList_(std::make_unique<BufferCacheFreeList>(
          this, device.hostBufferCache(), device.hostBufferCache().freeList())),
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
  return device().setCurrentDriverContext(driverContext());
}

MODULAR_CXX_EXPORT ErrorOr<Device::MemInfo>
DeviceContext::getMemoryInfo() const {
  // If we are using the buffer cache, get memory information from it.
  BufferCache &cache = device().bufferCache();
  if (cache.isEnabled()) {
    return cache.getMemoryInfo();
  }

  // Otherwise, go directly to the device.
  return getDeviceMemoryInfo();
}

void DeviceContext::setPendingError(Error err) const {
  {
    std::lock_guard<std::mutex> lock(mu_);
    // Do not overwrite an already pending error.
    if (pendingError_.isError())
      return;

    pendingError_ = std::move(err);
  }
  // After releasing mu_ also set the atomic value to true.
  hasPendingWork_.store(1, std::memory_order_release);
}

class DeviceContext::PendingEventHandler {
public:
  PendingEventHandler(DeviceContextRef ctx, DeviceEventRef event,
                      std::function<void()> handler)
      : next_(nullptr), ctx_(ctx), event_(std::move(event)),
        handler_(std::move(handler)) {}

  ErrorOr<bool> isReady() { return event_->isReady(); }

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
    std::lock_guard<std::mutex> lock(mu_);
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
  // After releasing mu_ also set the atomic value to true.
  hasPendingWork_.store(1, std::memory_order_release);
}

ErrorOrSuccess DeviceContext::checkPendingWork() const {
  // Quickly check if we should check for a pending error.
  if (hasPendingWork_.load(std::memory_order_acquire) == 0) {
    return success();
  }

  // We can safely remove the flag indicating that the pending error should be
  // checked. We will unconditionally acquire the lock below and check the
  // pending event handlers and errors while protected by the lock.
  hasPendingWork_.store(0, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(mu_);

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
      mu_.unlock();
      curHandler->runHandler();
      delete curHandler;
      mu_.lock();
    }

    // Check if we have a pending error recorded.
    if (!pendingError_.isError()) {
      return success();
    }
    // Return the error and reset.
    Error result = pendingError_.takeError();
    pendingError_ = success();
    return result;
  }
}

MODULAR_CXX_EXPORT ErrorOr<DeviceBufferRef>
DeviceContext::enqueueCreateBuffer(size_t byteSize) const {
  if (byteSize == 0)
    return this->createBufferOwning(nullptr, 0, true);
  return allocBuffer(byteSize, Device::AllocMode::ASYNC);
}

MODULAR_CXX_EXPORT ErrorOr<DeviceBufferRef>
DeviceContext::createBuffer(size_t byteSize) const {
  if (byteSize == 0)
    return this->createBufferOwning(nullptr, 0, true);
  return allocBuffer(byteSize, Device::AllocMode::SYNC);
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
  // Allocate from the correct buffer cache.
  ErrorOr<void *> errPtr =
      ((mode == Device::AllocMode::HOST) ? hostFreeList() : freeList())
          .allocate(byteSize, mode);
  if (errPtr.isError()) {
    return errPtr.takeError();
  }
  ptr = errPtr.takeValue();
  return DeviceBufferRef::take(
      new DeviceBuffer(*this, ptr, byteSize, mode == Device::AllocMode::HOST));
}

MODULAR_CXX_EXPORT ErrorOrSuccess DeviceContext::enqueueCopyDeviceToDevice(
    const DeviceBuffer &dst, const DeviceBuffer &src) const {
  const Device::API srcApi = src.context().device().api();
  const Device::API dstApi = dst.context().device().api();
  if (src.byteSize() < dst.byteSize()) {
    RAISE_ERROR_MESSAGE("not enough data in src");
  }
  if (src.isHost() || (srcApi == Device::cpuAPI)) {
    return dst.context().enqueueCopyToDevice(dst, src.ptr());
  } else if (dst.isHost() || (dstApi == Device::cpuAPI)) {
    return src.context().enqueueCopyFromDevice(dst.ptr(), src);
  } else if (srcApi == dstApi) {
    return this->memCopyDtoD(dst, src);
  }
  return Error("Device API mismatch copying from \"" + srcApi + "\" to \"" +
               dstApi + "\"");
}

MODULAR_CXX_EXPORT ErrorOrSuccess
DeviceContext::enqueueEventHandler(std::function<void()> handler) const {
  DEFINE_OR_RAISE(DeviceEventRef, event, this->enqueueEvent());
  PendingEventHandler *eventHandler = new PendingEventHandler(
      DeviceContextRef::copy(this), std::move(event), std::move(handler));
  setPendingEventHandler(eventHandler);
  return success();
}

MODULAR_CXX_EXPORT
ErrorOr<std::pair<AsyncRT::DeviceBufferRef,
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
      RCRef<AsyncRT::DeviceBuffer>::take(
          new AsyncRT::DeviceBuffer(*this, ptr, size, /*isHost*/ false,
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

BufferCacheFreeList &DeviceContext::freeList() const { return *freeList_; }
BufferCacheFreeList &DeviceContext::hostFreeList() const {
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
#ifdef MODULAR_MACOS
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

BufferCache &Device::bufferCache() const {
  assert(bufferCache_ != nullptr);
  return *bufferCache_;
}

BufferCache &Device::hostBufferCache() const {
  assert(hostBufferCache_ != nullptr);
  return *hostBufferCache_;
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

ErrorOr<Config> &DeviceContext::getConfig() {
  static ErrorOr<Config> configuration = Config::open();
  return configuration;
}

MODULAR_CXX_EXPORT const std::string &Device::getAcceleratorArchOrEmpty() {
  static std::string arch = [] {
    ErrorOr<DeviceRef> device = Device::create(gpuAPI);
    if (device.isError())
      return std::string();

    // Try to get the arch name directly. For AMD GPUs, we need this to succeed,
    // since major/minor compute capabilities don't describe the LLVM target
    // sufficiently.
    ErrorOr<std::string> archName = (*device)->getArchName();

    if (!archName.isError())
      return (llvm::Twine((*device)->vendor()) + ":" + archName.get()).str();

    // Fallback to compute capability if arch name is not available.
    ErrorOr<int> major =
        (*device)->getAttribute(Device::Attribute::COMPUTE_CAPABILITY_MAJOR);
    ErrorOr<int> minor =
        (*device)->getAttribute(Device::Attribute::COMPUTE_CAPABILITY_MINOR);
    if (major.isError() || minor.isError()) {
      return std::string();
    }

    // For metal only return the major compute capability
    if ((*device)->api() == Device::metalAPI) {
      return (llvm::Twine((*device)->vendor()) + ":" + llvm::Twine(*major))
          .str();
    }
    // For CUDA and HIP return the major and minor compute capability
    return (llvm::Twine((*device)->vendor()) + ":" + llvm::Twine(*major) +
            llvm::Twine(*minor))
        .str();
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
  return DeviceContext::formatError(
      __FILE__, __LINE__, __FUNCTION__,
      "enablePeerAccess not implemented for this device API" +
          Twine(peer.api()));
}
} // namespace M::AsyncRT
