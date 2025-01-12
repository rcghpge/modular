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
#include "HIPDriver.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/Twine.h"

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

namespace M::AsyncRT {

// Allow printing of detailed error messages with C++ source information when
// debugging low-level implementation.
static constexpr bool detailed_err_msg = false;

// TODO(iposva) : Switch error handling to using `std::source_location` once we
// are using C++ 20.
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

ErrorOrSuccess HIPDevice::isCompatibleWithMAX() const {
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

  hipDeviceptr_t cache_base = nullptr;
  if (bufferCacheSize != 0) {
    HIP_CHECK_ERR(this->hipDriver().MemAlloc(&cache_base, bufferCacheSize));
  }

  return HIPDeviceContextRef::create(*this, context, stream, cache_base,
                                     bufferCacheSize);
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

// ===== HIPDeviceBuffer ===== //
HIPDeviceBuffer::HIPDeviceBuffer(const HIPDeviceContext &context,
                                 size_t byteSize, hipDeviceptr_t dptr)
    : DeviceBuffer(context, byteSize), dptr_(dptr) {}

HIPDeviceBuffer::HIPDeviceBuffer(const HIPDeviceContext &context,
                                 const HIPDeviceBuffer &parent, size_t byteSize,
                                 hipDeviceptr_t dptr)
    : DeviceBuffer(context, parent, byteSize), dptr_(dptr) {}
HIPDeviceBuffer::HIPDeviceBuffer(const HIPDeviceContext &context,
                                 size_t byteSize, hipDeviceptr_t dptr,
                                 bool owning)
    : DeviceBuffer(context, byteSize), dptr_(dptr), owning_(owning) {}

HIPDeviceBuffer::~HIPDeviceBuffer() {
  if (!owning_) {
    // TODO: Remove the V1 compatibility.
    // This is a device buffer created by pointer in Mojo while ownership is
    // maintained elsewhere.
    return;
  }
  if ((parent() != nullptr) || (dptr_ == nullptr)) {
    // Parent owns the device memory or detached buffer.
    return;
  }

  const auto &hip_context = HIPDeviceContext::cast(context());
  if (!hip_context.bufferCache().free(dptr_)) {
    HIP_CHECK(
        hip_context.hipDriver().MemFreeAsync(dptr_, hip_context.hipStream()));
  }
}

hipDeviceptr_t HIPDeviceBuffer::hipPtr() const { return dptr_; }
void *HIPDeviceBuffer::ptr() const { return hipPtr(); }

// TODO: Remove the V1 compatibility.
void *HIPDeviceBuffer::releasePtr() const {
  auto tmp = dptr_;
  dptr_ = nullptr;
  owning_ = false;
  return tmp;
}

ErrorOr<DeviceBufferRef>
HIPDeviceBuffer::createSubBuffer(size_t byteOffset, size_t byteSize) const {
  char *orig_ptr = static_cast<char *>(this->hipPtr());
  size_t origByteSize = this->byteSize();

  if ((byteSize > origByteSize) || (origByteSize - byteSize < byteOffset)) {
    RAISE_ERROR("offset, len and elem_size are out of range");
  }
  return HIPDeviceBufferRef::create(
      HIPDeviceContext::cast(context()), *this, byteSize,
      static_cast<hipDeviceptr_t>(orig_ptr + byteOffset));
}

const HIPDeviceBuffer &HIPDeviceBuffer::cast(const DeviceBuffer &buf) {
  return static_cast<const HIPDeviceBuffer &>(buf);
}

// ===== HIPDeviceStream ===== //
HIPDeviceStream::HIPDeviceStream(const HIPDeviceContext &context,
                                 hipStream_t stream)
    : DeviceStream(context), stream_(stream) {}

ErrorOrSuccess HIPDeviceStream::synchronize() const {
  const auto &hip_context = HIPDeviceContext::cast(context());

  HIP_CHECK_ERR(hip_context.setCurrent());
  HIP_CHECK_ERR(hip_context.hipDriver().StreamSynchronize(stream_));
  return success();
}

hipStream_t HIPDeviceStream::hipStream() const { return stream_; }

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
  refs_.emplace_back(DeviceBufferRef::copy(&val));
  argValues_.push_back(HIPDeviceBuffer::cast(val).hipPtr());
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

// ===== HIPDeviceFunction ===== //
HIPDeviceFunction::HIPDeviceFunction(
    const HIPDeviceContext &context, std::string name,
    const DeviceFunction::SignatureBuilder &signature, hipModule_t module,
    hipFunction_t function)
    : DeviceFunction(context, signature), module_(module), function_(function),
      name_(std::move(name)) {}

HIPDeviceFunction::~HIPDeviceFunction() {
  const auto &hip_context = HIPDeviceContext::cast(context());

  HIP_CHECK(hip_context.hipDriver().ModuleUnload(module_));
}

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
      function_, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
      stream, args.hipArgs().data(), nullptr));
  return success();
}

ErrorOrSuccess HIPDeviceFunction::hipCallKernelDirect(
    uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
    uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes, void **args,
    hipStream_t stream) const {
  const auto &hip_context = HIPDeviceContext::cast(context());

  HIP_CHECK_ERR(hip_context.hipDriver().ModuleLaunchKernel(
      function_, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
      stream, args, nullptr));
  return success();
}

const std::string &HIPDeviceFunction::name() const { return name_; }

ErrorOrSuccess HIPDeviceFunction::copyToNamedConstant(const char *name,
                                                      const void *data,
                                                      size_t byteSize) const {
  hipDeviceptr_t device_ptr;
  size_t device_size;
  const auto &hip_context = HIPDeviceContext::cast(context());
  HIP_CHECK_ERR(hip_context.hipDriver().ModuleGetGlobal(
      &device_ptr, &device_size, module_, name));

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
      &result, static_cast<hipFunction_attribute>(code), function_));
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
                                   hipStream_t stream, void *bufferCacheBase,
                                   size_t bufferCacheSize)
    : DeviceContext(device), driver_(device.hipDriver()), context_(context),
      stream_(stream), bufCache_(bufferCacheBase, bufferCacheSize,
                                 device.allocationAlignment()) {}

HIPDeviceContext::~HIPDeviceContext() {
  // We do not need to synchronize on possible work queued on this stream.
  // Resources will be released automatically once work has been completed.
  bufferCache().clear();
  HIP_CHECK(driver_.StreamDestroy(stream_));
  HIP_CHECK(driver_.CtxDestroy(context_));
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

ErrorOr<DeviceBufferRef> HIPDeviceContext::allocBuffer(size_t byteSize,
                                                       SyncMode mode) const {
  assert(byteSize != 0);
  HIP_CHECK_ERR(setCurrent());

  hipDeviceptr_t dptr = bufferCache().allocate(byteSize);
  if (dptr == nullptr) {
    if (mode == ASYNC) {
      HIP_CHECK_ERR(driver_.MemAllocAsync(&dptr, byteSize, stream_));
    } else {
      HIP_CHECK_ERR(driver_.MemAlloc(&dptr, byteSize));
    }
  }
  return HIPDeviceBufferRef::create(*this, byteSize, dptr);
}

DeviceBufferRef HIPDeviceContext::createBufferOwning(void *ptr, size_t byteSize,
                                                     bool owning) const {
  hipDeviceptr_t dptr = static_cast<hipDeviceptr_t>(ptr);
  return HIPDeviceBufferRef::create(*this, byteSize, dptr, owning);
}

ErrorOrSuccess HIPDeviceContext::copyToDevice(const DeviceBuffer &dst,
                                              const void *src,
                                              SyncMode mode) const {
  HIP_CHECK_ERR(setCurrent());

  hipDeviceptr_t dst_ptr = HIPDeviceBuffer::cast(dst).hipPtr();
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
  HIP_CHECK_ERR(setCurrent());

  hipDeviceptr_t src_ptr = HIPDeviceBuffer::cast(src).hipPtr();
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
  const auto &src_buffer = HIPDeviceBuffer::cast(src);
  const auto &dst_buffer = HIPDeviceBuffer::cast(dst);
  const auto &src_ctx = HIPDeviceContext::cast(src_buffer.context());
  const auto &dst_ctx = HIPDeviceContext::cast(dst_buffer.context());

  auto src_ptr = src_buffer.hipPtr();
  auto dst_ptr = dst_buffer.hipPtr();
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
  void *ptr = HIPDeviceBuffer::cast(dst).hipPtr();
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
    const char *moduleName, const char *functionName,
    const DeviceFunction::SignatureBuilder &signature, const void *data,
    int32_t maxDynamicSharedBytes, const char *debugLevel,
    int32_t optimizationLevel) const {
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto iter = funcMap_.find(moduleName);
    if (iter != funcMap_.end()) {
      HIPDeviceFunctionRef result = iter->second;
      return result;
    }
  }

  // TODO(iposva): Remove once not needed for debugging.
  if (false) {
    std::cerr << "HIPDeviceContext::loadFunction(" << functionName << "):\n"
              << reinterpret_cast<const char *>(data) << "\n";
  }
  hipModule_t module;

  HIP_CHECK_ERR(driver_.ModuleLoadData(&module, data));

  hipFunction_t function;
  HIP_CHECK_ERR(driver_.ModuleGetFunction(&function, module, functionName));

  // Allocate func outside of the lock region to reduce the time spent under
  // lock. If this functionName has been registered concurrently, then we will
  // release this ref outside of the lock region before returning.
  auto func = HIPDeviceFunctionRef::create(*this, std::string(functionName),
                                           signature, module, function);
  HIPDeviceFunctionRef result;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto iter = funcMap_.find(moduleName);
    if (iter != funcMap_.end()) {
      result = HIPDeviceFunctionRef::copy(iter->second.getPointer());
    } else {
      funcMap_.insert({moduleName, func});
      result = std::move(func);
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
  return HIPDeviceStreamRef::create(*this, stream_);
}

ErrorOrSuccess HIPDeviceContext::synchronize() const {
  HIP_CHECK_ERR(driver_.StreamSynchronize(stream_));
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

hipStream_t HIPDeviceContext::hipStream() const { return stream_; }

BufferCache &HIPDeviceContext::bufferCache() const { return bufCache_; }

HIPDriver::Result HIPDeviceContext::setCurrent() const {
  return driver_.CtxSetCurrent(context_);
}

const HIPDeviceContext &HIPDeviceContext::cast(const DeviceContext &ctx) {
  return static_cast<const HIPDeviceContext &>(ctx);
}

} // namespace M::AsyncRT
