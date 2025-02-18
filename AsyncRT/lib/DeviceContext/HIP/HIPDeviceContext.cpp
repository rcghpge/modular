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

namespace M::AsyncRT {

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

ErrorOr<bool> HIPDevice::canAccess(const Device &peer) const { return false; }

ErrorOr<int> HIPDevice::getAttribute(Attribute attr) const {
  ErrorOr<hipDeviceAttribute_t> errorOrAttr = gethipDeviceAttribute(attr);
  if (errorOrAttr.isError())
    return errorOrAttr.takeError();
  int value = 0;
  HIP_CHECK_ERR(driver_.DeviceGetAttribute(&value, *errorOrAttr, hipDevice()));
  return value;
}

ErrorOr<DeviceContextRef>
HIPDevice::createContext(size_t bufferCacheSize) const {
  hipCtx_t context;
  HIP_CHECK_ERR(this->hipDriver().CtxCreate(&context, 0, this->id()));

  hipStream_t stream;
  HIP_CHECK_ERR(this->hipDriver().StreamCreate(&stream, 0));

  size_t cacheChunkSize = 0;
  size_t maxCacheSize = 0;
  if (bufferCacheSize != 0) {
    constexpr size_t KB = 1024; // NOLINT
    DeviceContext::MemInfo memInfo;
    HIP_CHECK_ERR(this->hipDriver().GetMemInfo(&memInfo.free, &memInfo.total));
    // Make the buffer cache disabled by setting percentages to zero.
    // Limit to 90% (default) of available memory.
    int64_t cachePercent =
        DeviceContext::configAsInt("buffer_cache_size_percent", 0);
    maxCacheSize = std::min(
        bufferCacheSize, static_cast<size_t>(llvm::alignToPowerOf2(
                             (memInfo.free * cachePercent) / 100, 256 * KB)));
    // Allocate chunks of 2% (default) of available memory.
    int64_t chunkPercent =
        DeviceContext::configAsInt("buffer_cache_chunk_percent", 0);
    cacheChunkSize = std::min(
        maxCacheSize, static_cast<size_t>(llvm::alignToPowerOf2(
                          (maxCacheSize * chunkPercent) / 100, 256 * KB)));
  }

  return HIPDeviceContextRef::create(*this, context, stream, cacheChunkSize,
                                     maxCacheSize);
}

ErrorOr<DeviceStreamRef>
HIPDevice::createExternalStream(void *externalStream) const {
  return HIPDeviceStreamRef::create(
      *this, reinterpret_cast<hipStream_t>(externalStream));
}

int HIPDevice::deviceCount() const {
  ErrorOr<const HIPDriver &> errDriver = HIPDriver::get();
  if (errDriver) {
    return 0;
  }

  int count = 0;
  HIPDriver::Result hipResult = errDriver.get().DeviceGetCount(&count);
  if (hipResult != HIPDriver::SUCCESS) {
    return 0;
  }
  return count;
}

HIPDriver::Result HIPDevice::setCurrent() const {
  return driver_.SetDevice(device_);
}

const HIPDevice &HIPDevice::cast(const Device &device) {
  return static_cast<const HIPDevice &>(device);
};

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
    int count = 0;
    HIP_CHECK_ERR(errDriver.get().DeviceGetCount(&count));

    char tmpName[256];
    int tmpAlignment;

    for (int i = 0; i < count; i++) {
      // TODO(iposva): Create correct hipDevice_t from ordinal.
      HIP_CHECK_ERR(errDriver.get().DeviceGetName(tmpName, sizeof(tmpName), i));
      HIP_CHECK_ERR(errDriver.get().DeviceGetAttribute(
          &tmpAlignment,
          hipDeviceAttribute_t::hipDeviceAttributeTextureAlignment, i));
      devices.emplace_back(
          HIPDeviceRef::create(errDriver.get(), i, i, tmpName, tmpAlignment));
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
  HIP_CHECK(hipDevice.hipDriver().EventDestroy(event_));
}

ErrorOrSuccess HIPDeviceEvent::synchronize() const {
  const auto &hipDevice = HIPDevice::cast(device());
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

  HIP_CHECK_ERR(hipDevice.setCurrent());
  HIP_CHECK_ERR(hipDevice.hipDriver().StreamSynchronize(stream_));
  return success();
}

ErrorOrSuccess
HIPDeviceStream::enqueueWaitForEvent(const DeviceEvent &event) const {
  const auto &hipEvent = HIPDeviceEvent::cast(event);
  const auto &hipDevice = HIPDevice::cast(device());
  HIP_CHECK_ERR(
      hipDevice.hipDriver().StreamWaitEvent(stream_, hipEvent.hipEvent(), 0));
  return success();
}

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
    : driver_(driver), module_(module), function_(function),
      name_(std::move(name)) {}

HIPModuleFunction::~HIPModuleFunction() {
  HIP_CHECK(hipDriver().ModuleUnload(module_));
}

const std::string &HIPModuleFunction::name() const { return name_; }
const HIPDriver &HIPModuleFunction::hipDriver() const { return driver_; }
hipModule_t HIPModuleFunction::hipModule() const { return module_; }
hipFunction_t HIPModuleFunction::hipFunction() const { return function_; }

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

  HIP_CHECK(hipContext.hipDriver().EventDestroy(start_));
  HIP_CHECK(hipContext.hipDriver().EventDestroy(stop_));
}

hipEvent_t HIPDeviceTimer::hipStartEvent() const { return start_; }
hipEvent_t HIPDeviceTimer::hipStopEvent() const { return stop_; }

const HIPDeviceTimer &HIPDeviceTimer::cast(const DeviceTimer &timer) {
  return static_cast<const HIPDeviceTimer &>(timer);
}

// ===== HIPDeviceContext ===== //
HIPDeviceContext::HIPDeviceContext(const HIPDevice &device, hipCtx_t context,
                                   hipStream_t stream, size_t cacheChunkSize,
                                   size_t maxCacheSize)
    : DeviceContext(device, std::make_unique<BufferCache>(
                                *this, cacheChunkSize, maxCacheSize,
                                device.allocationAlignment())),
      driver_(device.hipDriver()), context_(context), stream_(stream) {}

HIPDeviceContext::~HIPDeviceContext() {
  // Destroy the function map before destroying the context.
  funcMap_.clear();
  HIP_CHECK(driver_.StreamSynchronize(stream_));
  bufferCache().clear();
  HIP_CHECK(driver_.StreamDestroy(stream_));
  // We do not need to re-synchronize on possible work queued on this stream.
  // Resources will be released automatically once work has been completed.
  HIP_CHECK(driver_.CtxDestroy(context_));
}

ErrorOrSuccess HIPDeviceContext::memAlloc(void **ptr, size_t byteSize,
                                          AllocMode mode) const {
  assert(byteSize != 0);
  HIP_CHECK_ERR(setCurrent());
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
  HIP_CHECK_ERR(setCurrent());
  if (isHost) {
    HIP_CHECK_ERR(driver_.MemFreeHost(ptr));
  } else {
    HIP_CHECK_ERR(driver_.MemFreeAsync(ptr, stream_));
  }
  return success();
}

ErrorOrSuccess HIPDeviceContext::copyToDevice(const DeviceBuffer &dst,
                                              const void *src,
                                              SyncMode mode) const {
  CHECK_IS_HIP_ERR(dst.context().device());
  HIP_CHECK_ERR(setCurrent());

  hipDeviceptr_t dstPtr = dst.ptr();
  size_t byteSize = dst.byteSize();
  if (mode == ASYNC) {
    HIP_CHECK_ERR(driver_.MemcpyHtoDAsync(dstPtr, src, byteSize, stream_));
  } else {
    HIP_CHECK_ERR(driver_.MemcpyHtoD(dstPtr, src, byteSize));
  }
  return success();
}

ErrorOrSuccess HIPDeviceContext::copyFromDevice(void *dst,
                                                const DeviceBuffer &src,
                                                SyncMode mode) const {
  CHECK_IS_HIP_ERR(src.context().device());
  HIP_CHECK_ERR(setCurrent());

  hipDeviceptr_t srcPtr = src.ptr();
  size_t byteSize = src.byteSize();
  if (mode == ASYNC) {
    HIP_CHECK_ERR(driver_.MemcpyDtoHAsync(dst, srcPtr, byteSize, stream_));
  } else {
    HIP_CHECK_ERR(driver_.MemcpyDtoH(dst, srcPtr, byteSize));
  }
  return success();
}

ErrorOrSuccess HIPDeviceContext::copyDeviceToDevice(const DeviceBuffer &dst,
                                                    const DeviceBuffer &src,
                                                    SyncMode mode) const {
  CHECK_IS_HIP_ERR(src.context().device());
  CHECK_IS_HIP_ERR(dst.context().device());
  const auto &srcCtx = HIPDeviceContext::cast(src.context());
  const auto &dstCtx = HIPDeviceContext::cast(dst.context());

  auto srcPtr = src.ptr();
  auto dstPtr = dst.ptr();
  size_t dstByteSize = dst.byteSize();
  size_t srcByteSize = src.byteSize();
  if (srcByteSize < dstByteSize) {
    RAISE_ERROR("not enough data in src");
  }
  if (srcCtx.context_ != dstCtx.context_) {
    // See
    // https://github.com/ROCm/HIPIFY/blob/amd-staging/docs/tables/CUDA_Driver_API_functions_supported_by_HIP.md
    // Missing cuMemcpyPeer
    RAISE_ERROR("Peer copy not implemented yet");
  } else {
    if (mode == ASYNC) {
      HIP_CHECK_ERR(
          driver_.MemcpyDtoDAsync(dstPtr, srcPtr, dstByteSize, stream_));
    } else {
      HIP_CHECK_ERR(driver_.MemcpyDtoD(dstPtr, srcPtr, dstByteSize));
    }
  }
  return success();
}

ErrorOrSuccess HIPDeviceContext::setMemory(const DeviceBuffer &dst,
                                           uint32_t val, size_t valSize,
                                           SyncMode mode) const {
  CHECK_IS_HIP_ERR(dst.context().device());
  void *ptr = dst.ptr();
  size_t byteSize = dst.byteSize();
  if (valSize == 1) {
    if (mode == ASYNC) {
      HIP_CHECK_ERR(driver_.MemsetD8Async(ptr, static_cast<uint8_t>(val),
                                          byteSize, stream_));
    } else {
      HIP_CHECK_ERR(driver_.MemsetD8(ptr, static_cast<uint8_t>(val), byteSize));
    }
  } else if (valSize == 2) {
    if ((byteSize & 1) != 0) {
      return Error("byteSize of buffer is misaliged");
    }
    size_t len = byteSize >> 1;
    if (mode == ASYNC) {
      HIP_CHECK_ERR(driver_.MemsetD16Async(ptr, static_cast<uint16_t>(val), len,
                                           stream_));
    } else {
      HIP_CHECK_ERR(driver_.MemsetD16(ptr, static_cast<uint16_t>(val), len));
    }
  } else {
    if (valSize != 4) {
      return Error("unsupported valSize");
    }
    if ((byteSize & 3) != 0) {
      return Error("byteSize of buffer is misaliged");
    }
    size_t len = byteSize >> 2;
    if (mode == ASYNC) {
      HIP_CHECK_ERR(driver_.MemsetD32Async(ptr, val, len, stream_));
    } else {
      HIP_CHECK_ERR(driver_.MemsetD32(ptr, val, len));
    }
  }
  return success();
}

ErrorOr<DeviceFunctionRef> HIPDeviceContext::loadFunction(
    StringRef moduleName, StringRef functionName,
    const DeviceFunction::SignatureBuilder &signature, const void *data,
    int32_t maxDynamicSharedBytes, StringRef debugLevel,
    int32_t optimizationLevel) const {
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto iter = funcMap_.find(moduleName);
    if (iter != funcMap_.end()) {
      HIPModuleFunctionRef moduleFunc = iter->second;
      return HIPDeviceFunctionRef::create(*this, signature,
                                          std::move(moduleFunc));
    }
  }

  // TODO(iposva): Remove once not needed for debugging.
  if (false) {
    std::cerr << "HIPDeviceContext::loadFunction(" << functionName.str()
              << "):\n"
              << reinterpret_cast<const char *>(data) << "\n";
  }
  hipModule_t module;

  HIP_CHECK_ERR(driver_.ModuleLoadData(&module, data));

  hipFunction_t function;
  HIP_CHECK_ERR(
      driver_.ModuleGetFunction(&function, module, functionName.data()));

  // Allocate func outside of the lock region to reduce the time spent under
  // lock. If this functionName has been registered concurrently, then we will
  // release this ref outside of the lock region before returning.
  auto moduleFunc = HIPModuleFunctionRef::create(
      std::string(functionName), this->hipDriver(), module, function);
  HIPDeviceFunctionRef result;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto iter = funcMap_.find(moduleName);
    if (iter != funcMap_.end()) {
      result = HIPDeviceFunctionRef::create(
          *this, signature,
          HIPModuleFunctionRef::copy(iter->second.getPointer()));
    } else {
      funcMap_.insert({moduleName, moduleFunc});
      result =
          HIPDeviceFunctionRef::create(*this, signature, std::move(moduleFunc));
    }
  }
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
  HIP_CHECK_ERR(driver_.StreamSynchronize(stream_));
  return success();
}

ErrorOr<DeviceEventRef> HIPDeviceContext::enqueueEvent() const {
  hipEvent_t event;
  HIP_CHECK_ERR(driver_.EventCreate(&event, 0));
  HIP_CHECK_ERR(driver_.EventRecord(event, stream_));
  return HIPDeviceEventRef::create(HIPDevice::cast(this->device()), event);
}

ErrorOrSuccess
HIPDeviceContext::enqueueWaitForEvent(const DeviceEvent &event) const {
  const auto &hipEvent = HIPDeviceEvent::cast(event);
  HIP_CHECK_ERR(driver_.StreamWaitEvent(stream_, hipEvent.hipEvent(), 0));
  return success();
}

ErrorOr<DeviceTimerRef> HIPDeviceContext::startTimer() const {
  hipEvent_t start;
  hipEvent_t stop;

  HIP_CHECK_ERR(driver_.EventCreate(&start, 0));
  HIP_CHECK_ERR(driver_.EventCreate(&stop, 0));

  auto timer = HIPDeviceTimerRef::create(*this, start, stop);

  HIP_CHECK_ERR(driver_.EventRecord(timer->hipStartEvent(), stream_));

  return timer;
}

ErrorOr<std::chrono::nanoseconds>
HIPDeviceContext::stopTimer(const DeviceTimer &timer) const {
  const auto &hipTimer = HIPDeviceTimer::cast(timer);

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
  RAISE_ERROR("Unimplemented - HIPDeviceContext::enablePeerAccess");
}

ErrorOr<DeviceContext::MemInfo> HIPDeviceContext::getMemoryInfo() const {
  MemInfo memInfo;
  HIP_CHECK_ERR(driver_.GetMemInfo(&memInfo.free, &memInfo.total));
  return memInfo;
}

const HIPDriver &HIPDeviceContext::hipDriver() const { return driver_; }

hipCtx_t HIPDeviceContext::hipContext() const { return context_; }

hipStream_t HIPDeviceContext::hipStream() const { return stream_; }

HIPDriver::Result HIPDeviceContext::setCurrent() const {
  return driver_.CtxSetCurrent(context_);
}

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
