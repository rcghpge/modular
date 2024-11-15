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

// TODO(iposva) : Switch error handling to using `std::source_location` once we
// are using C++ 20.
static Error formatError(const char *file, int line, const char *func,
                         const llvm::Twine &msg) {
  return Error(llvm::Twine(file) + ":" + llvm::Twine(line) + " - " + func +
               " - " + msg);
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
HIPDevice::HIPDevice(const HIPDriver &driver, int id, hipDevice_t device,
                     std::string name)
    : Device(kind, name, id), driver_(driver), device_(device) {}

const HIPDriver &HIPDevice::driver() const { return driver_; }

hipDevice_t HIPDevice::hip_device() const { return device_; }

ErrorOr<int> HIPDevice::numberOfDevices() {
  ErrorOr<const HIPDriver &> err_driver = HIPDriver::get();
  if (err_driver) {
    return Error(llvm::Twine("HIP Driver initialization failed: ") +
                 err_driver.getError());
  }

  int count = 0;
  HIP_CHECK_ERR(err_driver.get().DeviceGetCount(&count));
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
      return Error(llvm::Twine("HIP Driver initialization failed: ") +
                   err_driver.getError());
    }
    int count = 0;
    HIP_CHECK_ERR(err_driver.get().DeviceGetCount(&count));

    char tmp_name[256];

    for (int i = 0; i < count; i++) {
      // TODO(iposva): Create correct hipDevice_t from ordinal.
      HIP_CHECK_ERR(
          err_driver.get().DeviceGetName(tmp_name, sizeof(tmp_name), i));
      devices.emplace_back(
          HIPDeviceRef::create(err_driver.get(), i, i, tmp_name));
    }
    return success();
  }();

  if (err) {
    return Error(err.getError());
  }
  return devices;
}

// ===== HIPDeviceBuffer ===== //
HIPDeviceBuffer::HIPDeviceBuffer(const HIPDeviceContext &context, size_t len,
                                 size_t elem_size, hipDeviceptr_t dptr)
    : DeviceBuffer(context, len, elem_size), dptr_(dptr) {}

HIPDeviceBuffer::HIPDeviceBuffer(const HIPDeviceContext &context,
                                 const HIPDeviceBuffer &parent, size_t len,
                                 size_t elem_size, hipDeviceptr_t dptr)
    : DeviceBuffer(context, parent, len, elem_size), dptr_(dptr) {}
HIPDeviceBuffer::HIPDeviceBuffer(const HIPDeviceContext &context, size_t len,
                                 size_t elem_size, hipDeviceptr_t dptr,
                                 bool owning)
    : DeviceBuffer(context, len, elem_size), dptr_(dptr), owning_(owning) {}

HIPDeviceBuffer::~HIPDeviceBuffer() {
  if (!owning_) {
    // TODO: Remove the V1 compatibility.
    // This is a device buffer created by pointer in Mojo while ownership is
    // maintained elsewhere.
    return;
  }
  if (parent() != nullptr) {
    // Parent owns the device memory.
    return;
  }

  const auto &hip_context = HIPDeviceContext::cast(context());
  HIP_CHECK(
      hip_context.hip_driver().MemFreeAsync(dptr_, hip_context.hip_stream()));
}

hipDeviceptr_t HIPDeviceBuffer::hip_ptr() const { return dptr_; }
void *HIPDeviceBuffer::ptr() const { return hip_ptr(); }

// TODO: Remove the V1 compatibility.
void *HIPDeviceBuffer::releasePtr() const {
  auto tmp = dptr_;
  dptr_ = nullptr;
  owning_ = false;
  return tmp;
}

ErrorOr<DeviceBufferRef>
HIPDeviceBuffer::createSubBuffer(size_t offset, size_t len,
                                 size_t elem_size) const {
  char *orig_ptr = static_cast<char *>(this->hip_ptr());
  size_t orig_byte_size = this->len() * this->elem_size();

  size_t byte_offset = offset * elem_size;
  size_t byte_size = len * elem_size;
  if ((byte_size > orig_byte_size) ||
      (orig_byte_size - byte_size < byte_offset)) {
    RAISE_ERROR("offset, len and elem_size are out of range");
  }
  return HIPDeviceBufferRef::create(
      HIPDeviceContext::cast(context()), *this, len, elem_size,
      static_cast<hipDeviceptr_t>(orig_ptr + byte_offset));
}

const HIPDeviceBuffer &HIPDeviceBuffer::cast(const DeviceBuffer &buf) {
  return static_cast<const HIPDeviceBuffer &>(buf);
}

// ===== HIPDeviceFunctionArguments ===== //
HIPDeviceFunctionArguments::HIPDeviceFunctionArguments(
    const HIPDeviceFunction &fn)
    : DeviceFunction::Arguments(fn) {}

void HIPDeviceFunctionArguments::pushArg(int index, const uint64_t &val) {
  arg_values_.push_back(reinterpret_cast<void *>(val));
}

void HIPDeviceFunctionArguments::pushArg(int index, const DeviceBuffer &val) {
  // Ensure buffer is preserved by increasing the reference count through
  // copy().
  refs_.emplace_back(DeviceBufferRef::copy(&val));
  arg_values_.push_back(HIPDeviceBuffer::cast(val).hip_ptr());
}

std::vector<void *> HIPDeviceFunctionArguments::hip_args() const {
  std::vector<void *> launch_args;
  launch_args.reserve(arg_values_.size());
  for (size_t i = 0; i < arg_values_.size(); i++) {
    launch_args.push_back(const_cast<void **>(&arg_values_[i]));
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

  HIP_CHECK(hip_context.hip_driver().ModuleUnload(module_));
}

DeviceFunction::Arguments *
HIPDeviceFunction::allocateArguments(const DeviceFunction &fn) const {
  return new HIPDeviceFunctionArguments(HIPDeviceFunction::cast(fn));
}

ErrorOrSuccess HIPDeviceFunction::hip_callKernel(
    uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
    uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    const HIPDeviceFunctionArguments &args, hipStream_t stream) const {
  const auto &hip_context = HIPDeviceContext::cast(context());

  HIP_CHECK_ERR(hip_context.hip_driver().ModuleLaunchKernel(
      function_, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
      stream, args.hip_args().data(), nullptr));
  return success();
}

ErrorOrSuccess HIPDeviceFunction::hip_callKernelDirect(
    uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
    uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes, void **args,
    hipStream_t stream) const {
  const auto &hip_context = HIPDeviceContext::cast(context());

  HIP_CHECK_ERR(hip_context.hip_driver().ModuleLaunchKernel(
      function_, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
      stream, args, nullptr));
  return success();
}

const std::string &HIPDeviceFunction::name() const { return name_; }

ErrorOrSuccess HIPDeviceFunction::copyToNamedConstant(const char *name,
                                                      const void *data,
                                                      size_t byte_size) const {
  hipDeviceptr_t device_ptr;
  size_t device_size;
  const auto &hip_context = HIPDeviceContext::cast(context());
  HIP_CHECK_ERR(hip_context.hip_driver().ModuleGetGlobal(
      &device_ptr, &device_size, module_, name));

  if (device_size < byte_size) {
    RAISE_ERROR("byte_size too big");
  }

  HIP_CHECK_ERR(hip_context.hip_driver().MemcpyHtoDAsync(
      device_ptr, data, byte_size, hip_context.hip_stream()));
  return success();
}

ErrorOr<int32_t> HIPDeviceFunction::TEST_ONLY_getAttribute(int32_t code) const {
  int32_t result;
  const auto &hip_context = HIPDeviceContext::cast(context());

  HIP_CHECK_ERR(hip_context.hip_driver().FuncGetAttribute(
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

  HIP_CHECK(hip_context.hip_driver().EventDestroy(start_));
  HIP_CHECK(hip_context.hip_driver().EventDestroy(stop_));
}

hipEvent_t HIPDeviceTimer::hip_start_event() const { return start_; }
hipEvent_t HIPDeviceTimer::hip_stop_event() const { return stop_; }

const HIPDeviceTimer &HIPDeviceTimer::cast(const DeviceTimer &timer) {
  return static_cast<const HIPDeviceTimer &>(timer);
}

// ===== HIPDeviceContext ===== //
HIPDeviceContext::HIPDeviceContext(const HIPDevice &device, hipCtx_t context,
                                   hipStream_t stream)
    : DeviceContext(device), driver_(device.driver()), context_(context),
      stream_(stream) {}

HIPDeviceContext::~HIPDeviceContext() {
  // We do not need to synchronize on possible work queued on this stream.
  // Resources will be released automatically once work has been completed.
  HIP_CHECK(driver_.StreamDestroy(stream_));
  HIP_CHECK(driver_.CtxDestroy(context_));
}

ErrorOr<void *> HIPDeviceContext::malloc_host(size_t size) const {
  void *ptr;
  HIP_CHECK_ERR(driver_.MemAllocHost(&ptr, size));
  return ptr;
}

ErrorOrSuccess HIPDeviceContext::free_host(void *ptr) const {
  HIP_CHECK_ERR(driver_.MemFreeHost(ptr));
  return success();
}

ErrorOr<DeviceBufferRef> HIPDeviceContext::createBuffer(size_t len,
                                                        size_t elem_size,
                                                        SyncMode mode) const {
  HIP_CHECK_ERR(setCurrent());

  size_t bytesize = len * elem_size;
  hipDeviceptr_t dptr;
  if (mode == ASYNC) {
    HIP_CHECK_ERR(driver_.MemAllocAsync(&dptr, bytesize, stream_));
  } else {
    HIP_CHECK_ERR(driver_.MemAlloc(&dptr, bytesize));
  }
  return HIPDeviceBufferRef::create(*this, len, elem_size, dptr);
}

DeviceBufferRef HIPDeviceContext::createBufferOwning(void *ptr, size_t len,
                                                     size_t elem_size,
                                                     bool owning) const {
  hipDeviceptr_t dptr = static_cast<hipDeviceptr_t>(ptr);
  return HIPDeviceBufferRef::create(*this, len, elem_size, dptr, owning);
}

ErrorOrSuccess HIPDeviceContext::copyToDevice(const DeviceBuffer &dst,
                                              const void *src,
                                              SyncMode mode) const {
  HIP_CHECK_ERR(setCurrent());

  hipDeviceptr_t dst_ptr = HIPDeviceBuffer::cast(dst).hip_ptr();
  size_t byte_size = dst.len() * dst.elem_size();
  if (mode == ASYNC) {
    HIP_CHECK_ERR(driver_.MemcpyHtoDAsync(dst_ptr, src, byte_size, stream_));
  } else {
    HIP_CHECK_ERR(driver_.MemcpyHtoD(dst_ptr, src, byte_size));
  }
  return success();
}

ErrorOrSuccess HIPDeviceContext::copyFromDevice(void *dst,
                                                const DeviceBuffer &src,
                                                SyncMode mode) const {
  HIP_CHECK_ERR(setCurrent());

  hipDeviceptr_t src_ptr = HIPDeviceBuffer::cast(src).hip_ptr();
  size_t byte_size = src.len() * src.elem_size();
  if (mode == ASYNC) {
    HIP_CHECK_ERR(driver_.MemcpyDtoHAsync(dst, src_ptr, byte_size, stream_));
  } else {
    HIP_CHECK_ERR(driver_.MemcpyDtoH(dst, src_ptr, byte_size));
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

  auto src_ptr = src_buffer.hip_ptr();
  auto dst_ptr = dst_buffer.hip_ptr();
  size_t dst_byte_size = dst.len() * dst.elem_size();
  size_t src_byte_size = src.len() * src.elem_size();
  if (src_byte_size < dst_byte_size) {
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
          driver_.MemcpyDtoDAsync(dst_ptr, src_ptr, dst_byte_size, stream_));
    } else {
      HIP_CHECK_ERR(driver_.MemcpyDtoD(dst_ptr, src_ptr, dst_byte_size));
    }
  }
  return success();
}

ErrorOrSuccess HIPDeviceContext::setMemory(const DeviceBuffer &dst,
                                           uint32_t val, size_t elem_size,
                                           SyncMode mode) const {
  void *ptr = HIPDeviceBuffer::cast(dst).hip_ptr();
  size_t len = dst.len();
  if (elem_size == 1) {
    if (mode == ASYNC) {
      HIP_CHECK_ERR(
          driver_.MemsetD8Async(ptr, static_cast<uint8_t>(val), len, stream_));
    } else {
      HIP_CHECK_ERR(driver_.MemsetD8(ptr, static_cast<uint8_t>(val), len));
    }
  } else if (elem_size == 2) {
    if (mode == ASYNC) {
      HIP_CHECK_ERR(driver_.MemsetD16Async(ptr, static_cast<uint16_t>(val), len,
                                           stream_));
    } else {
      HIP_CHECK_ERR(driver_.MemsetD16(ptr, static_cast<uint16_t>(val), len));
    }
  } else {
    if (mode == ASYNC) {
      HIP_CHECK_ERR(driver_.MemsetD32Async(ptr, val, len, stream_));
    } else {
      HIP_CHECK_ERR(driver_.MemsetD32(ptr, val, len));
    }
  }
  return success();
}

ErrorOr<DeviceFunctionRef> HIPDeviceContext::loadFunction(
    const char *module_name, const char *function_name,
    const DeviceFunction::SignatureBuilder &signature, const void *data,
    int32_t max_registers, int32_t threads_per_block, int32_t cache_mode,
    int32_t cache_config, int32_t max_dynamic_shared_bytes,
    const char *debug_level, int32_t optimization_level) const {
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto iter = func_map_.find(module_name);
    if (iter != func_map_.end()) {
      HIPDeviceFunctionRef result = iter->second;
      return result;
    }
  }

  // TODO(iposva): Remove once not needed for debugging.
  if (false) {
    std::cerr << "HIPDeviceContext::loadFunction(" << function_name << "):\n"
              << reinterpret_cast<const char *>(data) << "\n";
  }
  hipModule_t module;

  if (max_registers >= 0) {
    return Error("not implemented yet: max_registers");
  }

  if (threads_per_block >= 0) {
    return Error("not implemented yet: threads_per_block");
  }

  if (cache_mode >= 0) {
    return Error("not implemented yet: cache_mode");
  }

  HIP_CHECK_ERR(driver_.ModuleLoadData(&module, data));

  hipFunction_t function;
  HIP_CHECK_ERR(driver_.ModuleGetFunction(&function, module, function_name));

  // Allocate func outside of the lock region to reduce the time spent under
  // lock. If this function_name has been registered concurrently, then we will
  // release this ref outside of the lock region before returning.
  auto func = HIPDeviceFunctionRef::create(*this, std::string(function_name),
                                           signature, module, function);
  HIPDeviceFunctionRef result;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto iter = func_map_.find(module_name);
    if (iter != func_map_.end()) {
      result = HIPDeviceFunctionRef::copy(iter->second.getPointer());
    } else {
      func_map_.insert({module_name, func});
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
    return Error(args.getError().getError());
  }

  const auto &hip_func = HIPDeviceFunction::cast(func);

  return hip_func.hip_callKernel(
      gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
      HIPDeviceFunctionArguments::cast(args), stream_);
}

ErrorOrSuccess HIPDeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t num_attrs, void **args) const {

  const auto &hipFunc = HIPDeviceFunction::cast(func);
  if (num_attrs > 0)
    RAISE_ERROR("Function launch attributes not supported;");
  return hipFunc.hip_callKernelDirect(gridX, gridY, gridZ, blockX, blockY,
                                      blockZ, sharedMemBytes, args, stream_);
}

ErrorOr<DeviceStreamRef> HIPDeviceContext::stream() const {
  RAISE_ERROR("Unimplemented");
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

  HIP_CHECK_ERR(driver_.EventRecord(timer->hip_start_event(), stream_));

  return timer;
}

ErrorOr<std::chrono::nanoseconds>
HIPDeviceContext::stopTimer(const DeviceTimer &timer) const {
  const auto &hip_timer = HIPDeviceTimer::cast(timer);

  HIP_CHECK_ERR(driver_.EventRecord(hip_timer.hip_stop_event(), stream_));
  HIP_CHECK_ERR(driver_.EventSynchronize(hip_timer.hip_stop_event()));

  float millis;
  HIP_CHECK_ERR(driver_.EventElapsedTime(&millis, hip_timer.hip_start_event(),
                                         hip_timer.hip_stop_event()));

  constexpr double nanos_per_millisecond = 1000.0 * 1000.0;
  double nanos = static_cast<double>(millis) * nanos_per_millisecond;
  return std::chrono::nanoseconds(static_cast<int64_t>(nanos));
}

ErrorOrSuccess
HIPDeviceContext::enablePeerAccess(const DeviceContext &peer) const {
  RAISE_ERROR("Unimplemented - HIPDeviceContext::enablePeerAccess");
}

ErrorOr<int> HIPDeviceContext::getDriverVersion() const {
  int driver_version = 0;
  HIP_CHECK_ERR(driver_.DriverGetVersion(&driver_version));
  return driver_version;
}

ErrorOr<DeviceMemInfo> HIPDeviceContext::getMemoryInfo() const {
  DeviceMemInfo mem_info;
  HIP_CHECK_ERR(driver_.GetMemInfo(&mem_info.free, &mem_info.total));
  return mem_info;
}

const HIPDriver &HIPDeviceContext::hip_driver() const { return driver_; }

hipStream_t HIPDeviceContext::hip_stream() const { return stream_; }

HIPDriver::Result HIPDeviceContext::setCurrent() const {
  return driver_.CtxSetCurrent(context_);
}

const HIPDeviceContext &HIPDeviceContext::cast(const DeviceContext &ctx) {
  return static_cast<const HIPDeviceContext &>(ctx);
}

ErrorOrSuccess HIPDeviceContext::isCompatibleWithMAX() const {
  // We only support HIP architectures sm_80 and above.
  auto compute = computeCapability();
  if (compute.isError())
    return compute.takeError();
  constexpr int MIN_COMPUTE_CAPABILITY = 80;
  if (*compute < MIN_COMPUTE_CAPABILITY) {
    return Error("MAX only supports HIP Ampere architectures or higher");
  }

  // We only support HIP versions 12.0 and above.
  int driver_version = 0;
  constexpr int MIN_DRIVER_VERSION = 12.0;
  HIP_CHECK_ERR(driver_.DriverGetVersion(&driver_version));
  if (driver_version < MIN_DRIVER_VERSION) {
    return Error("MAX not supported using current GPU driver");
  }

  return success();
}

static ErrorOr<hipDeviceAttribute_t>
gethipDeviceAttribute(DeviceContext::Attribute attr) {
#define ENTRY(X, Y)                                                            \
  case DeviceContext::Attribute::X:                                            \
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
    return Error("Attribute not defined for AMD GPU");
  }
#undef ENTRY
}

// Plan:
// DeviceContext::Attribute will reuse the CUDA enum mapping
// CUDADeviceContext::getAttribute can thus use the enum values directly
// HIPDeviceContext will map the DeviceContext attributes to their HIP
// equivalents

ErrorOr<int> HIPDeviceContext::getAttribute(Attribute attr) const {

  auto errorOrAttr = gethipDeviceAttribute(attr);
  if (errorOrAttr.isError())
    return errorOrAttr.takeError();
  int value = 0;
  HIP_CHECK_ERR(driver_.DeviceGetAttribute(
      &value, *errorOrAttr, HIPDevice::cast(device()).hip_device()));
  return value;
}

ErrorOr<int> HIPDeviceContext::computeCapability() const {
  int compute_major = 0, compute_minor = 0;
  HIP_CHECK_ERR(driver_.DeviceGetAttribute(
      &compute_major,
      hipDeviceAttribute_t::hipDeviceAttributeComputeCapabilityMajor,
      HIPDevice::cast(device()).hip_device()));
  HIP_CHECK_ERR(driver_.DeviceGetAttribute(
      &compute_minor,
      hipDeviceAttribute_t::hipDeviceAttributeComputeCapabilityMinor,
      HIPDevice::cast(device()).hip_device()));
  return 10 * compute_major + compute_minor;
}

ErrorOr<HIPDeviceContextRef> HIPDeviceContext::create(int id) {
  ErrorOr<const HIPDevice &> err_or_device = HIPDevice::get(id);
  if (err_or_device) {
    return err_or_device.takeError();
  }
  const HIPDevice &device = *err_or_device;
  const HIPDriver &driver = device.driver();

  hipCtx_t context;
  HIP_CHECK_ERR(driver.CtxCreate(&context, 0, id));

  hipStream_t stream;
  HIP_CHECK_ERR(driver.StreamCreate(&stream, 0));

  return HIPDeviceContextRef::create(device, context, stream);
}

} // namespace M::AsyncRT
