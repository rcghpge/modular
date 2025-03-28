//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the HIP device context.
//
//===----------------------------------------------------------------------===//

#include "HIPDeviceContext.h"

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "AsyncRT/lib/DeviceContext/BufferCache.h"
#include "HIPDriver.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>

#define HIP_CHECK_ERR(hip_call)                                                \
  do {                                                                         \
    HIPDriver::Result hipResult = hip_call;                                    \
    if (hipResult != HIPDriver::SUCCESS) {                                     \
      return formatHipError(__FILE__, __LINE__, __FUNCTION__, hipResult);      \
    }                                                                          \
  } while (0)

#define HIP_CHECK(hip_call)                                                    \
  do {                                                                         \
    HIPDriver::Result hipResult = hip_call;                                    \
    if (hipResult != HIPDriver::SUCCESS) {                                     \
      M::Error err =                                                           \
          formatHipError(__FILE__, __LINE__, __FUNCTION__, hipResult);         \
      /* We are not in a context where we can return an Error here. So we   */ \
      /* print the message and abort().                                     */ \
      std::cerr << err << "\n";                                                \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define RAISE_ERROR(msg)                                                       \
  do {                                                                         \
    return DeviceContext::formatError(__FILE__, __LINE__, __FUNCTION__, msg);  \
  } while (0)

#define CHECK_IS_HIP_ERR(dev)                                                  \
  do {                                                                         \
    if ((dev).api() != Device::hipAPI) {                                       \
      RAISE_ERROR("Not a HIP device.");                                        \
    }                                                                          \
  } while (0)

#define DEFINE_OR_RAISE(type, var, call)                                       \
  auto _err_or_##var = call;                                                   \
  if (_err_or_##var.isError()) {                                               \
    return DeviceContext::formatError(__FILE__, __LINE__, __FUNCTION__,        \
                                      _err_or_##var.getError());               \
  }                                                                            \
  type var = _err_or_##var.takeValue()

#define ERR_CHECK(err_call)                                                    \
  do {                                                                         \
    auto err_result = (err_call);                                              \
    if (err_result.isError()) {                                                \
      M::Error err = DeviceContext::formatError(                               \
          __FILE__, __LINE__, __FUNCTION__, err_result.getError());            \
      /* We are not in a context where we can return an Error here. So we   */ \
      /* print the message and abort().                                     */ \
      llvm::errs() << err << "\n";                                             \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define HIP_WITH_CTX_ERR(ctx)                                                  \
  HIPDeviceScope __hipWithContext((ctx).hipDevice(), (ctx).hipDriver());       \
  HIP_CHECK_ERR(__hipWithContext.status());

#define HIP_WITH_CTX(ctx)                                                      \
  HIPDeviceScope __hipWithContext((ctx).hipDevice(), (ctx).hipDriver());       \
  HIP_CHECK(__hipWithContext.status());

#define HIP_WITH_DEV_ERR(dev)                                                  \
  HIPDeviceScope __hipWithContext((dev).hipDevice(), (dev).hipDriver());       \
  HIP_CHECK_ERR(__hipWithContext.status());

#define HIP_WITH_DEV(dev)                                                      \
  HIPDeviceScope __hipWithContext((dev).hipDevice(), (dev).hipDriver());       \
  HIP_CHECK(__hipWithContext.status());

namespace M::AsyncRT {

// The helper class HIPDeviceScope is used to set and restore the hipDevice_t
// around interactions with the HIP driver.
class HIPDeviceScope {
public:
  HIPDeviceScope(hipDevice_t device, const HIPDriver &driver)
      : driver_(driver), device_(device), prevDevice_(-1),
        status_(HIPDriver::SUCCESS) {
    driver_.GetDevice(&prevDevice_);
    if (prevDevice_ == device_)
      return;
    status_ = driver_.SetDevice(device_);
  }

  ~HIPDeviceScope() {
    if ((prevDevice_ == device_) || (status_ != HIPDriver::SUCCESS))
      return;

    // Only pop the context if push was necessary and successful.
    hipDevice_t device = -1;
    driver_.GetDevice(&device);
    driver_.SetDevice(prevDevice_);
    assert(device == device_);
  }

  HIPDriver::Result status() { return status_; }

private:
  const HIPDriver &driver_;
  const hipDevice_t device_;
  hipDevice_t prevDevice_;
  HIPDriver::Result status_;
};

static Error formatHipError(const char *file, int line, const char *func,
                            HIPDriver::Result result) {
  ErrorOr<const HIPDriver &> errDriver = HIPDriver::get();
  const char *errName = nullptr;
  const char *errStr = nullptr;

  if (!errDriver) {
    // Ignore errors and use nullptr of char* instead.
    errDriver->GetErrorName(result, &errName);
    errDriver->GetErrorString(result, &errStr);
  }

  if ((errName != nullptr) && (errStr != nullptr)) {
    return DeviceContext::formatError(file, line, func,
                                      llvm::Twine("HIP call failed: ") +
                                          errName + " (" + errStr + ")");
  } else if ((errName != nullptr) || (errStr != nullptr)) {
    return DeviceContext::formatError(
        file, line, func,
        llvm::Twine("HIP call failed: ") +
            (errName != nullptr ? errName : errStr));
  } else {
    return DeviceContext::formatError(file, line, func,
                                      llvm::Twine("HIP call failed: ") +
                                          llvm::Twine(result));
  }
}

// ===== HIPDevice ===== //
// Plan:
// DeviceContext::Attribute will reuse the CUDA enum mapping
// CUDADeviceContext::getAttribute can thus use the enum values directly
// HIPDeviceContext will map the DeviceContext attributes to their HIP
// equivalents
static ErrorOr<hipDeviceAttribute_t>
gethipDeviceAttribute(Device::Attribute attr) {
#define ENTRY(X, Y)                                                            \
  case Device::Attribute::X:                                                   \
    return hipDeviceAttribute_t::hipDeviceAttribute##Y

  switch (attr) {
    ENTRY(COMPUTE_CAPABILITY_MAJOR, ComputeCapabilityMajor);
    ENTRY(COMPUTE_CAPABILITY_MINOR, ComputeCapabilityMinor);
    ENTRY(CLOCK_RATE, ClockRate);
    ENTRY(WARP_SIZE, WarpSize);
    ENTRY(MAX_THREADS_PER_BLOCK, MaxThreadsPerBlock);
    ENTRY(MAX_SHARED_MEMORY_PER_BLOCK, MaxSharedMemoryPerBlock);
    ENTRY(MAX_BLOCK_DIM_X, MaxBlockDimX);
    ENTRY(MAX_BLOCK_DIM_Y, MaxBlockDimY);
    ENTRY(MAX_BLOCK_DIM_Z, MaxBlockDimZ);
    ENTRY(MAX_GRID_DIM_X, MaxGridDimX);
    ENTRY(MAX_GRID_DIM_Y, MaxGridDimY);
    ENTRY(MAX_GRID_DIM_Z, MaxGridDimZ);
    ENTRY(MULTIPROCESSOR_COUNT, MultiprocessorCount);
    ENTRY(MAX_REGISTERS_PER_BLOCK, MaxRegistersPerBlock);
    ENTRY(MAX_THREADS_PER_MULTIPROCESSOR, MaxThreadsPerMultiProcessor);
  default:
    RAISE_ERROR("Attribute not defined for AMD GPU");
  }
#undef ENTRY
}

HIPDevice::HIPDevice(const HIPDriver &driver, int id, hipDevice_t device,
                     std::string name, size_t allocAlignment)
    : Device(api, name, id, allocAlignment), driver_(driver), device_(device) {}

HIPDevice::~HIPDevice() {
  // Destroy the function map before destroying other device structures.
  clearFunctionCache();
}

const HIPDriver &HIPDevice::hipDriver() const { return driver_; }

hipDevice_t HIPDevice::hipDevice() const { return device_; }

ErrorOr<int> HIPDevice::getDriverVersion() const {
  int driverVersion = 0;
  HIP_CHECK_ERR(driver_.DriverGetVersion(&driverVersion));
  return driverVersion;
}

ErrorOrSuccess HIPDevice::isCompatible() const {
#ifndef MODULAR_LINUX
  return Error("MAX with GPUs is only supported on Linux based systems.");
#endif

  ErrorOr<int> compute = computeCapability();
  if (compute.isError())
    return compute.takeError();
  if (*compute < MIN_COMPUTE_CAPABILITY) {
    RAISE_ERROR("MAX only supports HIP Ampere architectures or higher");
  }

  int driverVersion = 0;
  HIP_CHECK_ERR(driver_.DriverGetVersion(&driverVersion));
  if (driverVersion < MIN_DRIVER_VERSION) {
    RAISE_ERROR("MAX not supported using current GPU driver");
  }

  return success();
}

ErrorOr<int> HIPDevice::computeCapability() const {
  int computeMajor = 0, computeMinor = 0;
  HIP_CHECK_ERR(driver_.DeviceGetAttribute(
      &computeMajor,
      hipDeviceAttribute_t::hipDeviceAttributeComputeCapabilityMajor,
      hipDevice()));
  HIP_CHECK_ERR(driver_.DeviceGetAttribute(
      &computeMinor,
      hipDeviceAttribute_t::hipDeviceAttributeComputeCapabilityMinor,
      hipDevice()));
  return 10 * computeMajor + computeMinor;
}

ErrorOr<bool> HIPDevice::canAccess(const Device &peer) const {
  if (id() == peer.id()) {
    // Always disable self access (same behaviour as hipDeviceCanAccessPeer).
    return false;
  }

  const HIPDevice *peerDevice = HIPDevice::verified_cast(&peer);
  if (peerDevice == nullptr) {
    RAISE_ERROR("Peer is not a HIP device");
  }

  int canAccess = 0;
  HIP_CHECK_ERR(driver_.DeviceCanAccessPeer(&canAccess, hipDevice(),
                                            peerDevice->hipDevice()));
  return canAccess == 1;
}

ErrorOr<int> HIPDevice::getAttribute(Attribute attr) const {
  HIP_WITH_DEV_ERR(*this);
  ErrorOr<hipDeviceAttribute_t> errorOrAttr = gethipDeviceAttribute(attr);
  if (errorOrAttr.isError())
    return errorOrAttr.takeError();
  int value = 0;
  HIP_CHECK_ERR(driver_.DeviceGetAttribute(&value, *errorOrAttr, hipDevice()));
  return value;
}

ErrorOr<DeviceContextRef> HIPDevice::createContext() const {
  // Make sure that we associate the stream with the correct device.
  HIP_WITH_DEV_ERR(*this);

  hipStream_t stream;
  HIP_CHECK_ERR(this->hipDriver().StreamCreate(&stream, 0));

  size_t cacheChunkSize = 0;
  size_t maxCacheSize = 0;
  constexpr size_t KB = 1024; // NOLINT
  DeviceContext::MemInfo memInfo;
  HIP_CHECK_ERR(this->hipDriver().GetMemInfo(&memInfo.free, &memInfo.total));
  // Make the buffer cache disabled by setting percentages to zero.
  // Limit to 90% (default) of available memory.
  int64_t cachePercent =
      DeviceContext::configAsInt("buffer_cache_size_percent", 0);
  maxCacheSize = static_cast<size_t>(
      llvm::alignToPowerOf2((memInfo.free * cachePercent) / 100, 256 * KB));
  // Allocate chunks of 2% (default) of available memory.
  int64_t chunkPercent =
      DeviceContext::configAsInt("buffer_cache_chunk_percent", 0);
  cacheChunkSize = std::min(
      maxCacheSize, static_cast<size_t>(llvm::alignToPowerOf2(
                        (maxCacheSize * chunkPercent) / 100, 256 * KB)));

  return HIPDeviceContextRef::create(*this, stream, cacheChunkSize,
                                     maxCacheSize);
}

ErrorOr<DeviceStreamRef>
HIPDevice::createExternalStream(void *externalStream) const {
  HIP_WITH_DEV_ERR(*this);
  return HIPDeviceStreamRef::create(
      *this, reinterpret_cast<hipStream_t>(externalStream));
}

int HIPDevice::deviceCount() const {
  static int count = []() -> int {
    ErrorOr<const HIPDriver &> errDriver = HIPDriver::get();
    if (errDriver.isError())
      return 0;

    int result;
    if ((*errDriver).DeviceGetCount(&result) != HIPDriver::SUCCESS)
      return 0;

    return result;
  }();
  return count;
}

ErrorOrSuccess HIPDevice::chunkAlloc(void **ptr, size_t byteSize) const {
  assert(byteSize != 0);
  HIP_WITH_DEV_ERR(*this);
  HIP_CHECK_ERR(driver_.MemAlloc(ptr, byteSize));
  return success();
}

ErrorOrSuccess HIPDevice::chunkFree(void *ptr) const {
  HIP_WITH_DEV_ERR(*this);
  HIP_CHECK_ERR(driver_.MemFree(ptr));
  return success();
}

const HIPDevice &HIPDevice::cast(const Device &device) {
  return static_cast<const HIPDevice &>(device);
}

const HIPDevice *HIPDevice::verified_cast(const Device *device) {
  if (device->api() != HIPDevice::api) {
    return nullptr;
  }
  return static_cast<const HIPDevice *>(device);
}

ErrorOr<const HIPDevice &> HIPDevice::get(int id) {
  ErrorOr<const DeviceList &> errOrDevices = deviceList();
  if (errOrDevices) {
    return errOrDevices.takeError();
  }
  return *errOrDevices.get()[id];
}

ErrorOr<const HIPDevice::DeviceList &> HIPDevice::deviceList() {
  static DeviceList devices; // Filled in by the initialization lambda below.
  static ErrorOrSuccess err = [&]() -> ErrorOrSuccess {
    ErrorOr<const HIPDriver &> errDriver = HIPDriver::get();
    if (errDriver) {
      RAISE_ERROR(llvm::Twine("HIP Driver initialization failed: ") +
                  errDriver.getError());
    }

    const HIPDriver &driver = *errDriver;
    int count = 0;
    HIP_CHECK_ERR(driver.DeviceGetCount(&count));

    for (int i = 0; i < count; i++) {
      char name[256];
      int alignment;

      hipDevice_t device = static_cast<hipDevice_t>(i);
      HIP_CHECK_ERR(driver.DeviceGetName(name, sizeof(name), device));
      HIP_CHECK_ERR(driver.DeviceGetAttribute(
          &alignment, hipDeviceAttribute_t::hipDeviceAttributeTextureAlignment,
          device));

      devices.emplace_back(
          HIPDeviceRef::create(driver, i, device, name, alignment));
    }
    return success();
  }();

  if (err) {
    RAISE_ERROR(err.getError());
  }
  return devices;
}

// ===== HIPDeviceEvent ===== //
HIPDeviceEvent::HIPDeviceEvent(const HIPDevice &device, hipEvent_t event)
    : DeviceEvent(device), event_(event) {}

HIPDeviceEvent::~HIPDeviceEvent() {
  const auto &hipDevice = HIPDevice::cast(device());

  HIP_WITH_DEV(hipDevice);
  HIP_CHECK(hipDevice.hipDriver().EventDestroy(event_));
}

ErrorOrSuccess HIPDeviceEvent::synchronize() const {
  const auto &hipDevice = HIPDevice::cast(device());

  HIP_WITH_DEV_ERR(hipDevice);
  HIP_CHECK_ERR(hipDevice.hipDriver().EventSynchronize(event_));
  return success();
}

hipEvent_t HIPDeviceEvent::hipEvent() const { return event_; }

const HIPDeviceEvent &HIPDeviceEvent::cast(const DeviceEvent &event) {
  return static_cast<const HIPDeviceEvent &>(event);
}

// ===== HIPDeviceStream ===== //
HIPDeviceStream::HIPDeviceStream(const HIPDevice &device, hipStream_t stream)
    : DeviceStream(device), stream_(stream) {}

ErrorOrSuccess HIPDeviceStream::synchronize() const {
  const auto &hipDevice = HIPDevice::cast(device());

  HIP_WITH_DEV_ERR(hipDevice);
  HIP_CHECK_ERR(hipDevice.hipDriver().StreamSynchronize(stream_));
  return success();
}

ErrorOr<DeviceEventRef> HIPDeviceStream::enqueueEvent() const {
  const auto &hipDevice = HIPDevice::cast(device());

  HIP_WITH_DEV_ERR(hipDevice);
  hipEvent_t event;
  HIP_CHECK_ERR(hipDevice.hipDriver().EventCreate(&event, 0));
  HIP_CHECK_ERR(hipDevice.hipDriver().EventRecord(event, stream_));
  return HIPDeviceEventRef::create(HIPDevice::cast(this->device()), event);
}

ErrorOrSuccess
HIPDeviceStream::enqueueWaitForEvent(const DeviceEvent &event) const {
  const auto &hipEvent = HIPDeviceEvent::cast(event);
  const auto &hipDevice = HIPDevice::cast(device());

  HIP_WITH_DEV_ERR(hipDevice);
  HIP_CHECK_ERR(
      hipDevice.hipDriver().StreamWaitEvent(stream_, hipEvent.hipEvent(), 0));
  return success();
}

void *HIPDeviceStream::ptr() const { return hipStream(); }

hipStream_t HIPDeviceStream::hipStream() const { return stream_; }

const HIPDeviceStream *
HIPDeviceStream::verified_cast(const DeviceStream *stream) {
  if (stream->device().api() != HIPDevice::api) {
    return nullptr;
  }
  return static_cast<const HIPDeviceStream *>(stream);
}

// ===== HIPDeviceFunctionArguments ===== //
HIPDeviceFunctionArguments::HIPDeviceFunctionArguments(
    const HIPDeviceFunction &fn)
    : DeviceFunction::Arguments(fn) {}

void HIPDeviceFunctionArguments::pushArg(int index, const uint64_t &val) {
  argValues_.push_back(reinterpret_cast<void *>(val));
}

void HIPDeviceFunctionArguments::pushArg(int index, const DeviceBuffer &val) {
  // Ensure buffer is preserved by increasing the reference count through
  // copy().
  assert(val.context().device().api() == Device::hipAPI);
  refs_.emplace_back(DeviceBufferRef::copy(&val));
  argValues_.push_back(val.ptr());
}

std::vector<void *> HIPDeviceFunctionArguments::hipArgs() const {
  std::vector<void *> launchArgs;
  launchArgs.reserve(argValues_.size());
  for (size_t i = 0; i < argValues_.size(); i++) {
    launchArgs.push_back(const_cast<void **>(&argValues_[i]));
  }
  return launchArgs;
}

const HIPDeviceFunctionArguments &
HIPDeviceFunctionArguments::cast(const DeviceFunction::Arguments &args) {
  return static_cast<const HIPDeviceFunctionArguments &>(args);
}

// ===== HIPModuleFunction ===== //
HIPModuleFunction::HIPModuleFunction(const std::string name,
                                     const HIPDriver &driver,
                                     hipModule_t module, hipFunction_t function)
    : DeviceFunctionCacheEntry(std::move(name)), driver_(driver),
      module_(module), function_(function) {}

HIPModuleFunction::~HIPModuleFunction() {
  HIP_CHECK(hipDriver().ModuleUnload(module_));
}

const HIPDriver &HIPModuleFunction::hipDriver() const { return driver_; }
hipModule_t HIPModuleFunction::hipModule() const { return module_; }
hipFunction_t HIPModuleFunction::hipFunction() const { return function_; }

const HIPModuleFunction *
HIPModuleFunction::cast(const DeviceFunctionCacheEntry *entry) {
  return static_cast<const HIPModuleFunction *>(entry);
}

// ===== HIPDeviceFunction ===== //
HIPDeviceFunction::HIPDeviceFunction(
    const HIPDeviceContext &context,
    const DeviceFunction::SignatureBuilder &signature,
    HIPModuleFunctionRef modFunc)
    : DeviceFunction(context, signature), moduleFunction_(std::move(modFunc)) {}

HIPDeviceFunction::~HIPDeviceFunction() {}

DeviceFunction::Arguments *
HIPDeviceFunction::allocateArguments(const DeviceFunction &fn) const {
  return new HIPDeviceFunctionArguments(HIPDeviceFunction::cast(fn));
}

ErrorOrSuccess HIPDeviceFunction::hipCallKernel(
    uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
    uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    const HIPDeviceFunctionArguments &args, hipStream_t stream) const {
  const auto &hipContext = HIPDeviceContext::cast(context());

  HIP_WITH_CTX_ERR(hipContext);
  HIP_CHECK_ERR(hipContext.hipDriver().ModuleLaunchKernel(
      moduleFunction_->hipFunction(), gridX, gridY, gridZ, blockX, blockY,
      blockZ, sharedMemBytes, stream, args.hipArgs().data(), nullptr));
  return success();
}

ErrorOrSuccess HIPDeviceFunction::hipCallKernelDirect(
    uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
    uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes, void **args,
    hipStream_t stream) const {
  const auto &hipContext = HIPDeviceContext::cast(context());

  HIP_WITH_CTX_ERR(hipContext);
  HIP_CHECK_ERR(hipContext.hipDriver().ModuleLaunchKernel(
      moduleFunction_->hipFunction(), gridX, gridY, gridZ, blockX, blockY,
      blockZ, sharedMemBytes, stream, args, nullptr));
  return success();
}

const std::string &HIPDeviceFunction::name() const {
  return moduleFunction_->name();
}

ErrorOrSuccess HIPDeviceFunction::copyToNamedConstant(const char *name,
                                                      const void *data,
                                                      size_t byteSize) const {
  hipDeviceptr_t devicePtr;
  size_t deviceSize;
  const auto &hipContext = HIPDeviceContext::cast(context());

  HIP_WITH_CTX_ERR(hipContext);
  HIP_CHECK_ERR(hipContext.hipDriver().ModuleGetGlobal(
      &devicePtr, &deviceSize, moduleFunction_->hipModule(), name));

  if (deviceSize < byteSize) {
    RAISE_ERROR("byteSize too big");
  }

  HIP_CHECK_ERR(hipContext.hipDriver().MemcpyHtoDAsync(
      devicePtr, data, byteSize, hipContext.hipStream()));
  return success();
}

ErrorOr<int32_t> HIPDeviceFunction::getAttribute(int32_t code) const {
  int32_t result;
  const auto &hipContext = HIPDeviceContext::cast(context());

  HIP_WITH_CTX_ERR(hipContext);
  HIP_CHECK_ERR(hipContext.hipDriver().FuncGetAttribute(
      &result, static_cast<hipFunction_attribute>(code),
      moduleFunction_->hipFunction()));
  return result;
}

const HIPDeviceFunction &HIPDeviceFunction::cast(const DeviceFunction &func) {
  return static_cast<const HIPDeviceFunction &>(func);
}

// ===== HIPDeviceTimer ===== //
HIPDeviceTimer::HIPDeviceTimer(const HIPDeviceContext &context,
                               hipEvent_t start, hipEvent_t stop)
    : DeviceTimer(context), start_(start), stop_(stop) {}

HIPDeviceTimer::~HIPDeviceTimer() {
  const auto &hipContext = HIPDeviceContext::cast(context());

  HIP_WITH_CTX(hipContext);
  HIP_CHECK(hipContext.hipDriver().EventDestroy(start_));
  HIP_CHECK(hipContext.hipDriver().EventDestroy(stop_));
}

hipEvent_t HIPDeviceTimer::hipStartEvent() const { return start_; }
hipEvent_t HIPDeviceTimer::hipStopEvent() const { return stop_; }

const HIPDeviceTimer &HIPDeviceTimer::cast(const DeviceTimer &timer) {
  return static_cast<const HIPDeviceTimer &>(timer);
}

// ===== HIPDeviceContext ===== //
HIPDeviceContext::HIPDeviceContext(const HIPDevice &device, hipStream_t stream,
                                   size_t cacheChunkSize, size_t maxCacheSize)
    : DeviceContext(device, std::make_unique<BufferCache>(
                                *this, cacheChunkSize, maxCacheSize,
                                device.allocationAlignment())),
      driver_(device.hipDriver()), device_(device.hipDevice()),
      stream_(stream) {}

HIPDeviceContext::~HIPDeviceContext() {
  HIP_WITH_CTX(*this);
  HIP_CHECK(driver_.StreamSynchronize(stream_));
  bufferCache().clear();
  HIP_CHECK(driver_.StreamDestroy(stream_));
}

ErrorOrSuccess HIPDeviceContext::memAlloc(void **ptr, size_t byteSize,
                                          AllocMode mode) const {
  assert(byteSize != 0);
  HIP_WITH_CTX_ERR(*this);
  switch (mode) {
  case AllocMode::ASYNC: {
    HIP_CHECK_ERR(driver_.MemAllocAsync(ptr, byteSize, stream_));
    break;
  }
  case AllocMode::HOST: {
    HIP_CHECK_ERR(driver_.MemAllocHost(ptr, byteSize));
    break;
  }
  case AllocMode::SYNC: {
    HIP_CHECK_ERR(driver_.MemAlloc(ptr, byteSize));
    break;
  }
  }
  return success();
}

ErrorOrSuccess HIPDeviceContext::memFree(void *ptr, bool isHost) const {
  HIP_WITH_CTX_ERR(*this);
  if (isHost) {
    HIP_CHECK_ERR(driver_.MemFreeHost(ptr));
  } else {
    HIP_CHECK_ERR(driver_.MemFreeAsync(ptr, stream_));
  }
  return success();
}

ErrorOrSuccess HIPDeviceContext::enqueueCopyToDevice(const DeviceBuffer &dst,
                                                     const void *src) const {
  CHECK_IS_HIP_ERR(dst.context().device());
  HIP_WITH_CTX_ERR(*this);

  hipDeviceptr_t dstPtr = dst.ptr();
  size_t byteSize = dst.byteSize();
  HIP_CHECK_ERR(driver_.MemcpyHtoDAsync(dstPtr, src, byteSize, stream_));
  return success();
}

ErrorOrSuccess
HIPDeviceContext::enqueueCopyFromDevice(void *dst,
                                        const DeviceBuffer &src) const {
  CHECK_IS_HIP_ERR(src.context().device());
  HIP_WITH_CTX_ERR(*this);

  hipDeviceptr_t srcPtr = src.ptr();
  size_t byteSize = src.byteSize();
  HIP_CHECK_ERR(driver_.MemcpyDtoHAsync(dst, srcPtr, byteSize, stream_));
  return success();
}

ErrorOrSuccess HIPDeviceContext::memCopyDtoD(const DeviceBuffer &dst,
                                             const DeviceBuffer &src) const {
  CHECK_IS_HIP_ERR(src.context().device());
  CHECK_IS_HIP_ERR(dst.context().device());
  const auto &srcCtx = HIPDeviceContext::cast(src.context());
  const auto &dstCtx = HIPDeviceContext::cast(dst.context());

  auto srcPtr = src.ptr();
  auto dstPtr = dst.ptr();
  size_t dstByteSize = dst.byteSize();
  [[maybe_unused]] size_t srcByteSize = src.byteSize();
  assert(srcByteSize >= dstByteSize);
  if (srcCtx.hipDevice() != dstCtx.hipDevice()) {
    HIP_CHECK_ERR(
        driver_.MemcpyDtoDAsync(dstPtr, srcPtr, dstByteSize, stream_));
  } else {
    HIP_CHECK_ERR(
        driver_.MemcpyDtoDAsync(dstPtr, srcPtr, dstByteSize, stream_));
  }
  return success();
}

ErrorOrSuccess HIPDeviceContext::enqueueSetMemory(const DeviceBuffer &dst,
                                                  uint64_t val,
                                                  size_t valSize) const {
  CHECK_IS_HIP_ERR(dst.context().device());
  void *ptr = dst.ptr();
  size_t byteSize = dst.byteSize();
  if (valSize == 1) {
    HIP_CHECK_ERR(driver_.MemsetD8Async(ptr, static_cast<uint8_t>(val),
                                        byteSize, stream_));
  } else if (valSize == 2) {
    if ((byteSize & 1) != 0) {
      return Error("byteSize of buffer is misaliged");
    }
    size_t len = byteSize >> 1;
    HIP_CHECK_ERR(
        driver_.MemsetD16Async(ptr, static_cast<uint16_t>(val), len, stream_));
  } else if (valSize == 4) {
    if ((byteSize & 3) != 0) {
      return Error("byteSize of buffer is misaliged");
    }
    size_t len = byteSize >> 2;
    HIP_CHECK_ERR(
        driver_.MemsetD32Async(ptr, static_cast<uint32_t>(val), len, stream_));
  } else {
    if (valSize != 8) {
      return Error("unsupported valSize");
    }
    if ((byteSize & 7) != 0) {
      return Error("byteSize of buffer is misaliged");
    }
    uint32_t hiBits = static_cast<uint32_t>(val >> 32);
    uint32_t loBits = static_cast<uint32_t>(val & 0xfffffffful);
    if (hiBits == loBits) {
      size_t len = byteSize >> 2; // Divide by 4 as we are using 32-bit memset.
      HIP_CHECK_ERR(driver_.MemsetD32Async(ptr, loBits, len, stream_));
    } else {
      size_t len = byteSize >> 3;
      char *curPtr = reinterpret_cast<char *>(ptr);
      uint64_t curVal = val;
      for (int i = 0; i < 8; i++) {
        HIP_CHECK_ERR(driver_.Memset2DAsync(
            curPtr, 8, static_cast<int>(curVal & 0xff), 1, len, stream_));
        curPtr += 1;
        curVal >>= 8;
      }
    }
  }
  return success();
}

ErrorOr<DeviceFunctionRef> HIPDeviceContext::loadFunction(
    StringRef moduleName, StringRef functionName,
    const DeviceFunction::SignatureBuilder &signature, StringRef code,
    int32_t maxDynamicSharedBytes, StringRef debugLevel,
    int32_t optimizationLevel) const {
  DeviceFunctionCacheEntryRef entry =
      device().findInFunctionCache(moduleName.empty() ? code : moduleName);
  if (entry.getPointer() != nullptr) {
    HIPModuleFunctionRef moduleFunc =
        HIPModuleFunctionRef::take(HIPModuleFunction::cast(entry.release()));
    return HIPDeviceFunctionRef::create(*this, signature,
                                        std::move(moduleFunc));
  }

  // TODO(iposva): Remove once not needed for debugging.
  if (false) {
    std::cerr << "HIPDeviceContext::loadFunction(" << functionName.str()
              << "):\n"
              << code.data() << "\n";
  }
  hipModule_t module;

  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.ModuleLoadData(&module, code.data()));

  hipFunction_t function;
  HIP_CHECK_ERR(
      driver_.ModuleGetFunction(&function, module, functionName.data()));

  // Allocate func outside of the lock region to reduce the time spent under
  // lock. If this functionName has been registered concurrently, then we will
  // release this ref outside of the lock region before returning.
  auto moduleFunc = HIPModuleFunctionRef::create(
      std::string(functionName), this->hipDriver(), module, function);
  HIPDeviceFunctionRef result;
  entry = device().insertInFunctionCache(moduleName.empty() ? code : moduleName,
                                         *moduleFunc);
  result = HIPDeviceFunctionRef::create(
      *this, signature,
      HIPModuleFunctionRef::take(HIPModuleFunction::cast(entry.release())));
  return result;
}

ErrorOrSuccess HIPDeviceContext::enqueueFunctionExec(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    const DeviceFunction::Arguments &args) const {
  if (args.hasError()) {
    RAISE_ERROR(args.getError().getError());
  }

  const auto &hipFunc = HIPDeviceFunction::cast(func);

  return hipFunc.hipCallKernel(gridX, gridY, gridZ, blockX, blockY, blockZ,
                               sharedMemBytes,
                               HIPDeviceFunctionArguments::cast(args), stream_);
}

ErrorOrSuccess HIPDeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args) const {
  const auto &hipFunc = HIPDeviceFunction::cast(func);
  if (numAttrs > 0)
    RAISE_ERROR("Function launch attributes not supported;");
  return hipFunc.hipCallKernelDirect(gridX, gridY, gridZ, blockX, blockY,
                                     blockZ, sharedMemBytes, args, stream_);
}

ErrorOr<DeviceStreamRef> HIPDeviceContext::stream() const {
  return HIPDeviceStreamRef::create(HIPDevice::cast(this->device()), stream_);
}

ErrorOrSuccess HIPDeviceContext::synchronize() const {
  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.StreamSynchronize(stream_));
  return success();
}

ErrorOr<DeviceEventRef> HIPDeviceContext::enqueueEvent() const {
  hipEvent_t event;
  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.EventCreate(&event, 0));
  HIP_CHECK_ERR(driver_.EventRecord(event, stream_));
  return HIPDeviceEventRef::create(HIPDevice::cast(this->device()), event);
}

ErrorOrSuccess
HIPDeviceContext::enqueueWaitFor(const DeviceEvent &event) const {
  const auto &hipEvent = HIPDeviceEvent::cast(event);
  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.StreamWaitEvent(stream_, hipEvent.hipEvent(), 0));
  return success();
}

ErrorOrSuccess
HIPDeviceContext::enqueueWaitFor(const DeviceContext &ctx) const {
  DEFINE_OR_RAISE(DeviceEventRef, event, ctx.enqueueEvent());
  ERR_CHECK(enqueueWaitFor(*event));
  return success();
}

ErrorOr<DeviceTimerRef> HIPDeviceContext::startTimer() const {
  hipEvent_t start;
  hipEvent_t stop;

  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.EventCreate(&start, 0));
  HIP_CHECK_ERR(driver_.EventCreate(&stop, 0));

  auto timer = HIPDeviceTimerRef::create(*this, start, stop);

  HIP_CHECK_ERR(driver_.EventRecord(timer->hipStartEvent(), stream_));

  return timer;
}

ErrorOr<std::chrono::nanoseconds>
HIPDeviceContext::stopTimer(const DeviceTimer &timer) const {
  const auto &hipTimer = HIPDeviceTimer::cast(timer);

  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.EventRecord(hipTimer.hipStopEvent(), stream_));
  HIP_CHECK_ERR(driver_.EventSynchronize(hipTimer.hipStopEvent()));

  float millis;
  HIP_CHECK_ERR(driver_.EventElapsedTime(&millis, hipTimer.hipStartEvent(),
                                         hipTimer.hipStopEvent()));

  constexpr double nanosPerMillisecond = 1000.0 * 1000.0;
  double nanos = static_cast<double>(millis) * nanosPerMillisecond;
  return std::chrono::nanoseconds(static_cast<int64_t>(nanos));
}

ErrorOrSuccess
HIPDeviceContext::enablePeerAccess(const DeviceContext &peer) const {
  if (device().id() == peer.device().id()) {
    // Enabling peer access to self is not allowed.
    RAISE_ERROR("Cannot enable peer access to self");
  }

  const HIPDevice *peerDevice = HIPDevice::verified_cast(&peer.device());
  if (peerDevice == nullptr) {
    RAISE_ERROR("Peer is not a HIP device");
  }

  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.DeviceEnablePeerAccess(peerDevice->hipDevice(), 0));
  return success();
}

ErrorOr<DeviceContext::MemInfo> HIPDeviceContext::getMemoryInfo() const {
  MemInfo memInfo;
  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.GetMemInfo(&memInfo.free, &memInfo.total));
  return memInfo;
}

const HIPDriver &HIPDeviceContext::hipDriver() const { return driver_; }

hipDevice_t HIPDeviceContext::hipDevice() const { return device_; }

hipStream_t HIPDeviceContext::hipStream() const { return stream_; }

const HIPDeviceContext &HIPDeviceContext::cast(const DeviceContext &ctx) {
  return static_cast<const HIPDeviceContext &>(ctx);
}

const HIPDeviceContext *
HIPDeviceContext::verified_cast(const DeviceContext *ctx) {
  if (ctx->device().api() != HIPDevice::api) {
    return nullptr;
  }
  return static_cast<const HIPDeviceContext *>(ctx);
}

} // namespace M::AsyncRT
