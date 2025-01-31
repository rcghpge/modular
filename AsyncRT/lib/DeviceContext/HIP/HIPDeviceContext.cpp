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
    HIPDriver::Result hip_result = hip_call;                                   \
    if (hip_result != HIPDriver::SUCCESS) {                                    \
      return formatError(__FILE__, __LINE__, __FUNCTION__, hip_result);        \
    }                                                                          \
  } while (0)

#define HIP_CHECK(hip_call)                                                    \
  do {                                                                         \
    HIPDriver::Result hip_result = hip_call;                                   \
    if (hip_result != HIPDriver::SUCCESS) {                                    \
      M::Error err =                                                           \
          formatError(__FILE__, __LINE__, __FUNCTION__, hip_result);           \
      /* We are not in a context where we can return an Error here. So we   */ \
      /* print the message and abort().                                     */ \
      std::cerr << err << "\n";                                                \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define RAISE_ERROR(msg)                                                       \
  do {                                                                         \
    return formatError(__FILE__, __LINE__, __FUNCTION__, msg);                 \
  } while (0)

#define CHECK_IS_HIP_ERR(dev)                                                  \
  do {                                                                         \
    if ((dev).api() != Device::hipAPI) {                                       \
      RAISE_ERROR("Not a HIP device.");                                        \
    }                                                                          \
  } while (0)

namespace M::AsyncRT {

// Allow printing of detailed error messages with C++ source information when
// debugging low-level implementation.
static constexpr bool detailed_err_msg = false;

// TODO(iposva) : Switch error handling to using `std::source_location` once
// we are using C++ 20.
static Error formatError(const char *file, int line, const char *func,
                         const llvm::Twine &msg) {
  if (detailed_err_msg) {
    return Error(llvm::Twine(file) + ":" + llvm::Twine(line) + " - " + func +
                 " - " + msg);
  }
  return Error(msg);
}

static Error formatError(const char *file, int line, const char *func,
                         HIPDriver::Result result) {
  ErrorOr<const HIPDriver &> err_driver = HIPDriver::get();
  const char *err_name = nullptr;
  const char *err_str = nullptr;

  if (!err_driver) {
    // Ignore errors and use nullptr of char* instead.
    err_driver->GetErrorName(result, &err_name);
    err_driver->GetErrorString(result, &err_str);
  }

  if ((err_name != nullptr) && (err_str != nullptr)) {
    return formatError(file, line, func,
                       llvm::Twine("HIP call failed: ") + err_name + " (" +
                           err_str + ")");
  } else if ((err_name != nullptr) || (err_str != nullptr)) {
    return formatError(file, line, func,
                       llvm::Twine("HIP call failed: ") +
                           (err_name != nullptr ? err_name : err_str));
  } else {
    return formatError(file, line, func,
                       llvm::Twine("HIP call failed: ") + llvm::Twine(result));
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
  int driver_version = 0;
  HIP_CHECK_ERR(driver_.DriverGetVersion(&driver_version));
  return driver_version;
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

  int driver_version = 0;
  HIP_CHECK_ERR(driver_.DriverGetVersion(&driver_version));
  if (driver_version < MIN_DRIVER_VERSION) {
    RAISE_ERROR("MAX not supported using current GPU driver");
  }

  return success();
}

ErrorOr<int> HIPDevice::computeCapability() const {
  int compute_major = 0, compute_minor = 0;
  HIP_CHECK_ERR(driver_.DeviceGetAttribute(
      &compute_major,
      hipDeviceAttribute_t::hipDeviceAttributeComputeCapabilityMajor,
      hipDevice()));
  HIP_CHECK_ERR(driver_.DeviceGetAttribute(
      &compute_minor,
      hipDeviceAttribute_t::hipDeviceAttributeComputeCapabilityMinor,
      hipDevice()));
  return 10 * compute_major + compute_minor;
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
    constexpr size_t KB = 1024;
    DeviceContext::MemInfo mem_info;
    HIP_CHECK_ERR(
        this->hipDriver().GetMemInfo(&mem_info.free, &mem_info.total));
    // Make the buffer cache disabled by setting percentages to zero.
    // Limit to 90% (default) of available memory.
    int64_t cache_percent =
        DeviceContext::configAsInt("buffer_cache_size_percent", 0);
    maxCacheSize = std::min(
        bufferCacheSize, static_cast<size_t>(llvm::alignToPowerOf2(
                             (mem_info.free * cache_percent) / 100, 256 * KB)));
    // Allocate chunks of 2% (default) of available memory.
    int64_t chunk_percent =
        DeviceContext::configAsInt("buffer_cache_chunk_percent", 0);
    cacheChunkSize = std::min(
        maxCacheSize, static_cast<size_t>(llvm::alignToPowerOf2(
                          (maxCacheSize * chunk_percent) / 100, 256 * KB)));
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
  ErrorOr<const HIPDriver &> err_driver = HIPDriver::get();
  if (err_driver) {
    return 0;
  }

  int count = 0;
  HIPDriver::Result hip_result = err_driver.get().DeviceGetCount(&count);
  if (hip_result != HIPDriver::SUCCESS) {
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
  ErrorOr<const DeviceList &> err_or_devices = deviceList();
  if (err_or_devices) {
    return err_or_devices.takeError();
  }
  return *err_or_devices.get()[id];
}

ErrorOr<const HIPDevice::DeviceList &> HIPDevice::deviceList() {
  static DeviceList devices; // Filled in by the initialization lambda below.
  static ErrorOrSuccess err = [&]() -> ErrorOrSuccess {
    ErrorOr<const HIPDriver &> err_driver = HIPDriver::get();
    if (err_driver) {
      RAISE_ERROR(llvm::Twine("HIP Driver initialization failed: ") +
                  err_driver.getError());
    }
    int count = 0;
    HIP_CHECK_ERR(err_driver.get().DeviceGetCount(&count));

    char tmp_name[256];
    int tmp_alignment;

    for (int i = 0; i < count; i++) {
      // TODO(iposva): Create correct hipDevice_t from ordinal.
      HIP_CHECK_ERR(
          err_driver.get().DeviceGetName(tmp_name, sizeof(tmp_name), i));
      HIP_CHECK_ERR(err_driver.get().DeviceGetAttribute(
          &tmp_alignment,
          hipDeviceAttribute_t::hipDeviceAttributeTextureAlignment, i));
      devices.emplace_back(HIPDeviceRef::create(err_driver.get(), i, i,
                                                tmp_name, tmp_alignment));
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
  const auto &hip_device = HIPDevice::cast(device());
  HIP_CHECK_ERR(
      hip_device.hipDriver().StreamWaitEvent(stream_, hipEvent.hipEvent(), 0));
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
  std::vector<void *> launch_args;
  launch_args.reserve(argValues_.size());
  for (size_t i = 0; i < argValues_.size(); i++) {
    launch_args.push_back(const_cast<void **>(&argValues_[i]));
  }
  return launch_args;
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
  const auto &hip_context = HIPDeviceContext::cast(context());

  HIP_CHECK_ERR(hip_context.hipDriver().ModuleLaunchKernel(
      moduleFunction_->hipFunction(), gridX, gridY, gridZ, blockX, blockY,
      blockZ, sharedMemBytes, stream, args.hipArgs().data(), nullptr));
  return success();
}

ErrorOrSuccess HIPDeviceFunction::hipCallKernelDirect(
    uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
    uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes, void **args,
    hipStream_t stream) const {
  const auto &hip_context = HIPDeviceContext::cast(context());

  HIP_CHECK_ERR(hip_context.hipDriver().ModuleLaunchKernel(
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
  hipDeviceptr_t device_ptr;
  size_t device_size;
  const auto &hip_context = HIPDeviceContext::cast(context());
  HIP_CHECK_ERR(hip_context.hipDriver().ModuleGetGlobal(
      &device_ptr, &device_size, moduleFunction_->hipModule(), name));

  if (device_size < byteSize) {
    RAISE_ERROR("byteSize too big");
  }

  HIP_CHECK_ERR(hip_context.hipDriver().MemcpyHtoDAsync(
      device_ptr, data, byteSize, hip_context.hipStream()));
  return success();
}

ErrorOr<int32_t> HIPDeviceFunction::getAttribute(int32_t code) const {
  int32_t result;
  const auto &hip_context = HIPDeviceContext::cast(context());

  HIP_CHECK_ERR(hip_context.hipDriver().FuncGetAttribute(
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
  const auto &hip_context = HIPDeviceContext::cast(context());

  HIP_CHECK(hip_context.hipDriver().EventDestroy(start_));
  HIP_CHECK(hip_context.hipDriver().EventDestroy(stop_));
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
                                          SyncMode mode) const {
  assert(byteSize != 0);
  HIP_CHECK_ERR(setCurrent());
  if (mode == ASYNC) {
    HIP_CHECK_ERR(driver_.MemAllocAsync(ptr, byteSize, stream_));
  } else {
    HIP_CHECK_ERR(driver_.MemAlloc(ptr, byteSize));
  }
  return success();
}

ErrorOrSuccess HIPDeviceContext::memFree(void *ptr) const {
  HIP_CHECK_ERR(setCurrent());

  HIP_CHECK_ERR(driver_.MemFreeAsync(ptr, stream_));
  return success();
}

ErrorOr<void *> HIPDeviceContext::mallocHost(size_t size) const {
  void *ptr;
  HIP_CHECK_ERR(driver_.MemAllocHost(&ptr, size));
  return ptr;
}

ErrorOrSuccess HIPDeviceContext::freeHost(void *ptr) const {
  HIP_CHECK_ERR(driver_.MemFreeHost(ptr));
  return success();
}

ErrorOrSuccess HIPDeviceContext::copyToDevice(const DeviceBuffer &dst,
                                              const void *src,
                                              SyncMode mode) const {
  CHECK_IS_HIP_ERR(dst.context().device());
  HIP_CHECK_ERR(setCurrent());

  hipDeviceptr_t dst_ptr = dst.ptr();
  size_t byteSize = dst.byteSize();
  if (mode == ASYNC) {
    HIP_CHECK_ERR(driver_.MemcpyHtoDAsync(dst_ptr, src, byteSize, stream_));
  } else {
    HIP_CHECK_ERR(driver_.MemcpyHtoD(dst_ptr, src, byteSize));
  }
  return success();
}

ErrorOrSuccess HIPDeviceContext::copyFromDevice(void *dst,
                                                const DeviceBuffer &src,
                                                SyncMode mode) const {
  CHECK_IS_HIP_ERR(src.context().device());
  HIP_CHECK_ERR(setCurrent());

  hipDeviceptr_t src_ptr = src.ptr();
  size_t byteSize = src.byteSize();
  if (mode == ASYNC) {
    HIP_CHECK_ERR(driver_.MemcpyDtoHAsync(dst, src_ptr, byteSize, stream_));
  } else {
    HIP_CHECK_ERR(driver_.MemcpyDtoH(dst, src_ptr, byteSize));
  }
  return success();
}

ErrorOrSuccess HIPDeviceContext::copyDeviceToDevice(const DeviceBuffer &dst,
                                                    const DeviceBuffer &src,
                                                    SyncMode mode) const {
  CHECK_IS_HIP_ERR(src.context().device());
  CHECK_IS_HIP_ERR(dst.context().device());
  const auto &src_ctx = HIPDeviceContext::cast(src.context());
  const auto &dst_ctx = HIPDeviceContext::cast(dst.context());

  auto src_ptr = src.ptr();
  auto dst_ptr = dst.ptr();
  size_t dstByteSize = dst.byteSize();
  size_t srcByteSize = src.byteSize();
  if (srcByteSize < dstByteSize) {
    RAISE_ERROR("not enough data in src");
  }
  if (src_ctx.context_ != dst_ctx.context_) {
    // See
    // https://github.com/ROCm/HIPIFY/blob/amd-staging/docs/tables/CUDA_Driver_API_functions_supported_by_HIP.md
    // Missing cuMemcpyPeer
    RAISE_ERROR("Peer copy not implemented yet");
  } else {
    if (mode == ASYNC) {
      HIP_CHECK_ERR(
          driver_.MemcpyDtoDAsync(dst_ptr, src_ptr, dstByteSize, stream_));
    } else {
      HIP_CHECK_ERR(driver_.MemcpyDtoD(dst_ptr, src_ptr, dstByteSize));
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
      HIPModuleFunctionRef module_func = iter->second;
      return HIPDeviceFunctionRef::create(*this, signature,
                                          std::move(module_func));
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
  auto module_func = HIPModuleFunctionRef::create(
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
      funcMap_.insert({moduleName, module_func});
      result = HIPDeviceFunctionRef::create(*this, signature,
                                            std::move(module_func));
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

  const auto &hip_func = HIPDeviceFunction::cast(func);

  return hip_func.hipCallKernel(
      gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
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
  const auto &hip_timer = HIPDeviceTimer::cast(timer);

  HIP_CHECK_ERR(driver_.EventRecord(hip_timer.hipStopEvent(), stream_));
  HIP_CHECK_ERR(driver_.EventSynchronize(hip_timer.hipStopEvent()));

  float millis;
  HIP_CHECK_ERR(driver_.EventElapsedTime(&millis, hip_timer.hipStartEvent(),
                                         hip_timer.hipStopEvent()));

  constexpr double nanos_per_millisecond = 1000.0 * 1000.0;
  double nanos = static_cast<double>(millis) * nanos_per_millisecond;
  return std::chrono::nanoseconds(static_cast<int64_t>(nanos));
}

ErrorOrSuccess
HIPDeviceContext::enablePeerAccess(const DeviceContext &peer) const {
  RAISE_ERROR("Unimplemented - HIPDeviceContext::enablePeerAccess");
}

ErrorOr<DeviceContext::MemInfo> HIPDeviceContext::getMemoryInfo() const {
  MemInfo mem_info;
  HIP_CHECK_ERR(driver_.GetMemInfo(&mem_info.free, &mem_info.total));
  return mem_info;
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
