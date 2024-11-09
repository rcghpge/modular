//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CUDA device context.
//
//===----------------------------------------------------------------------===//

#include "CUDADeviceContext.h"

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "CUDADriver.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <mutex>
#include <string>

#define CUDA_CHECK_ERR(cuda_call)                                              \
  do {                                                                         \
    CUDADriver::Result cuda_result = cuda_call;                                \
    if (cuda_result != CUDADriver::SUCCESS) {                                  \
      return formatError(__FILE__, __LINE__, __FUNCTION__, cuda_result);       \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(cuda_call)                                                  \
  do {                                                                         \
    CUDADriver::Result cuda_result = cuda_call;                                \
    if (cuda_result != CUDADriver::SUCCESS) {                                  \
      M::Error err =                                                           \
          formatError(__FILE__, __LINE__, __FUNCTION__, cuda_result);          \
      /* We are not in a context where we can return an Error here. So we   */ \
      /* print the message and abort().                                     */ \
      llvm::errs() << err << "\n";                                             \
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
                         CUDADriver::Result result) {
  ErrorOr<const CUDADriver &> err_driver = CUDADriver::get();
  const char *err_name = nullptr;
  const char *err_str = nullptr;

  if (!err_driver) {
    // Ignore errors and use nullptr of char* instead.
    err_driver->GetErrorName(result, &err_name);
    err_driver->GetErrorString(result, &err_str);
  }

  if ((err_name != nullptr) && (err_str != nullptr)) {
    return formatError(file, line, func,
                       llvm::Twine("CUDA call failed: ") + err_name + " (" +
                           err_str + ")");
  } else if ((err_name != nullptr) || (err_str != nullptr)) {
    return formatError(file, line, func,
                       llvm::Twine("CUDA call failed: ") +
                           (err_name != nullptr ? err_name : err_str));
  } else {
    return formatError(file, line, func,
                       llvm::Twine("CUDA call failed: ") + llvm::Twine(result));
  }
}

// ===== CUDADevice ===== //
CUDADevice::CUDADevice(const CUDADriver &driver, int id, CUdevice device,
                       std::string name)
    : Device(std::move(name)), driver_(driver), id_(id), device_(device) {}

const CUDADriver &CUDADevice::driver() const { return driver_; }

int CUDADevice::id() const { return static_cast<int>(id_); }

CUdevice CUDADevice::cuda_device() const { return device_; }

ErrorOr<int> CUDADevice::numberOfDevices() {
  ErrorOr<const DeviceList &> err_or_devices = deviceList();
  if (err_or_devices) {
    return err_or_devices.takeError();
  }
  return err_or_devices.get().size();
}

ErrorOr<bool> CUDADevice::canAccess(const Device &peer) const {
  int canAccessPeer = 0;
  CUDA_CHECK_ERR(driver_.DeviceCanAccessPeer(
      &canAccessPeer, cuda_device(), CUDADevice::cast(peer).cuda_device()));
  return canAccessPeer;
}

const CUDADevice &CUDADevice::cast(const Device &device) {
  return static_cast<const CUDADevice &>(device);
};

ErrorOr<const CUDADevice &> CUDADevice::get(int id) {
  ErrorOr<const DeviceList &> err_or_devices = deviceList();
  if (err_or_devices) {
    return err_or_devices.takeError();
  }
  return *err_or_devices.get()[id];
}

ErrorOr<const CUDADevice::DeviceList &> CUDADevice::deviceList() {
  // Initialization of these two statics only happens once on the first call. We
  // ensure this by initializing the err using a lambda.
  static DeviceList devices;
  static const ErrorOrSuccess err = [&]() -> ErrorOrSuccess {
    ErrorOr<const CUDADriver &> err_driver = CUDADriver::get();
    if (err_driver) {
      return Error(llvm::Twine("CUDA Driver initialization failed: ") +
                   err_driver.getError());
    }
    int count = 0;
    CUDA_CHECK_ERR(err_driver.get().DeviceGetCount(&count));

    char tmp_name[256];
    CUdevice tmp_device;

    for (int i = 0; i < count; i++) {
      CUDA_CHECK_ERR(err_driver.get().DeviceGet(&tmp_device, i));
      CUDA_CHECK_ERR(err_driver.get().DeviceGetName(tmp_name, sizeof(tmp_name),
                                                    tmp_device));
      devices.emplace_back(
          CUDADeviceRef::create(err_driver.get(), i, tmp_device, tmp_name));
    }
    return success();
  }();

  if (err) {
    return Error(err.getError());
  }
  return devices;
}

// ===== CUDADeviceBuffer ===== //
CUDADeviceBuffer::CUDADeviceBuffer(const CUDADeviceContext &context, size_t len,
                                   size_t elem_size, CUdeviceptr dptr)
    : DeviceBuffer(context, len, elem_size), dptr_(dptr) {}

CUDADeviceBuffer::CUDADeviceBuffer(const CUDADeviceContext &context,
                                   const CUDADeviceBuffer &parent, size_t len,
                                   size_t elem_size, CUdeviceptr dptr)
    : DeviceBuffer(context, parent, len, elem_size), dptr_(dptr) {}
CUDADeviceBuffer::CUDADeviceBuffer(const CUDADeviceContext &context, size_t len,
                                   size_t elem_size, CUdeviceptr dptr,
                                   bool owning)
    : DeviceBuffer(context, len, elem_size), dptr_(dptr), owning_(owning) {}

CUDADeviceBuffer::~CUDADeviceBuffer() {
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

  const auto &cuda_context = CUDADeviceContext::cast(context());
  CUDA_CHECK(cuda_context.setCurrent());
  CUDA_CHECK(cuda_context.cuda_driver().MemFreeAsync(
      dptr_, cuda_context.cuda_stream()));
}

CUdeviceptr CUDADeviceBuffer::cuda_ptr() const { return dptr_; }
void *CUDADeviceBuffer::ptr() const { return cuda_ptr(); }

// TODO: Remove the V1 compatibility.
void *CUDADeviceBuffer::releasePtr() const {
  auto tmp = dptr_;
  dptr_ = nullptr;
  owning_ = false;
  return tmp;
}

ErrorOr<DeviceBufferRef>
CUDADeviceBuffer::createSubBuffer(size_t offset, size_t len,
                                  size_t elem_size) const {
  char *orig_ptr = static_cast<char *>(this->cuda_ptr());
  size_t orig_byte_size = this->len() * this->elem_size();

  size_t byte_offset = offset * elem_size;
  size_t byte_size = len * elem_size;
  if ((byte_size > orig_byte_size) ||
      (orig_byte_size - byte_size < byte_offset)) {
    RAISE_ERROR("offset, len and elem_size are out of range");
  }
  return CUDADeviceBufferRef::create(
      CUDADeviceContext::cast(context()), *this, len, elem_size,
      static_cast<CUdeviceptr>(orig_ptr + byte_offset));
}

const CUDADeviceBuffer &CUDADeviceBuffer::cast(const DeviceBuffer &buf) {
  return static_cast<const CUDADeviceBuffer &>(buf);
}

// ===== CUDADeviceStream ===== //
CUDADeviceStream::CUDADeviceStream(const CUDADeviceContext &context,
                                   CUstream stream)
    : DeviceStream(context), stream_(stream) {}

ErrorOrSuccess CUDADeviceStream::synchronize() const {
  const auto &cuda_context = CUDADeviceContext::cast(context());

  CUDA_CHECK_ERR(cuda_context.setCurrent());
  CUDA_CHECK_ERR(cuda_context.cuda_driver().StreamSynchronize(stream_));
  return success();
}

CUstream CUDADeviceStream::cuda_stream() const { return stream_; }

// ===== CUDADeviceFunctionArguments ===== //
CUDADeviceFunctionArguments::CUDADeviceFunctionArguments(
    const CUDADeviceFunction &fn)
    : DeviceFunction::Arguments(fn) {}

void CUDADeviceFunctionArguments::pushArg(int index, const uint64_t &val) {
  arg_values_.push_back(reinterpret_cast<void *>(val));
}

void CUDADeviceFunctionArguments::pushArg(int index, const DeviceBuffer &val) {
  // Ensure buffer is preserved by increasing the reference count through
  // copy().
  refs_.emplace_back(DeviceBufferRef::copy(&val));
  arg_values_.push_back(CUDADeviceBuffer::cast(val).cuda_ptr());
}

std::vector<void *> CUDADeviceFunctionArguments::cuda_args() const {
  std::vector<void *> launch_args;
  launch_args.reserve(arg_values_.size());
  for (size_t i = 0; i < arg_values_.size(); i++) {
    launch_args.push_back(const_cast<void **>(&arg_values_[i]));
  }
  return launch_args;
}

const CUDADeviceFunctionArguments &
CUDADeviceFunctionArguments::cast(const DeviceFunction::Arguments &args) {
  return static_cast<const CUDADeviceFunctionArguments &>(args);
}

// ===== CUDAModuleFunction ===== //
CUDAModuleFunction::CUDAModuleFunction(const std::string name,
                                       const CUDADriver &driver,
                                       CUmodule module, CUfunction function)
    : driver_(driver), module_(module), function_(function),
      name_(std::move(name)) {}

CUDAModuleFunction::~CUDAModuleFunction() {
  CUDA_CHECK(cuda_driver().ModuleUnload(module_));
}

const std::string &CUDAModuleFunction::name() const { return name_; }
const CUDADriver &CUDAModuleFunction::cuda_driver() const { return driver_; }
CUmodule CUDAModuleFunction::cuda_module() const { return module_; }
CUfunction CUDAModuleFunction::cuda_function() const { return function_; }

// ===== CUDADeviceFunction ===== //
CUDADeviceFunction::CUDADeviceFunction(
    const CUDADeviceContext &context,
    const DeviceFunction::SignatureBuilder &signature,
    CUDAModuleFunctionRef mod_func)
    : DeviceFunction(context, signature),
      module_function_(std::move(mod_func)) {}
CUDADeviceFunction::~CUDADeviceFunction() {}

DeviceFunction::Arguments *
CUDADeviceFunction::allocateArguments(const DeviceFunction &fn) const {
  return new CUDADeviceFunctionArguments(CUDADeviceFunction::cast(fn));
}

ErrorOrSuccess CUDADeviceFunction::cuda_callKernel(
    uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
    uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    const CUDADeviceFunctionArguments &args, CUstream stream) const {
  CUDA_CHECK_ERR(module_function_->cuda_driver().LaunchKernel(
      module_function_->cuda_function(), gridX, gridY, gridZ, blockX, blockY,
      blockZ, sharedMemBytes, stream, args.cuda_args().data(), nullptr));
  return success();
}

ErrorOrSuccess CUDADeviceFunction::cuda_callKernelDirect(
    uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
    uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    CUlaunchAttribute *attrs, uint32_t num_attrs, void **args,
    CUstream stream) const {
  CUlaunchConfig config;
  config.gridDimX = gridX;
  config.gridDimY = gridY;
  config.gridDimZ = gridZ;
  config.blockDimX = blockX;
  config.blockDimY = blockY;
  config.blockDimZ = blockZ;
  config.sharedMemBytes = sharedMemBytes;
  config.hStream = stream;
  config.attrs = attrs;
  config.numAttrs = num_attrs;

  CUDA_CHECK_ERR(module_function_->cuda_driver().LaunchKernelEx(
      &config, module_function_->cuda_function(), args, nullptr));
  return success();
}

const std::string &CUDADeviceFunction::name() const {
  return module_function_->name();
}

ErrorOrSuccess CUDADeviceFunction::copyToNamedConstant(const char *name,
                                                       const void *data,
                                                       size_t byte_size) const {
  CUdeviceptr device_ptr;
  size_t device_size;
  const auto &cuda_context = CUDADeviceContext::cast(context());
  CUDA_CHECK_ERR(cuda_context.cuda_driver().ModuleGetGlobal(
      &device_ptr, &device_size, module_function_->cuda_module(), name));

  if (device_size < byte_size) {
    RAISE_ERROR("byte_size too big");
  }

  CUDA_CHECK_ERR(cuda_context.cuda_driver().MemcpyHtoDAsync(
      device_ptr, data, byte_size, cuda_context.cuda_stream()));
  return success();
}

ErrorOr<int32_t>
CUDADeviceFunction::TEST_ONLY_getAttribute(int32_t code) const {
  int32_t result;
  CUDA_CHECK_ERR(module_function_->cuda_driver().FuncGetAttribute(
      &result, static_cast<CUfunction_attribute>(code),
      module_function_->cuda_function()));
  return result;
}

const CUDADeviceFunction &CUDADeviceFunction::cast(const DeviceFunction &func) {
  return static_cast<const CUDADeviceFunction &>(func);
}

// ===== CUDADeviceTimer ===== //
CUDADeviceTimer::CUDADeviceTimer(const CUDADeviceContext &context,
                                 CUevent start, CUevent stop)
    : DeviceTimer(context), start_(start), stop_(stop) {}

CUDADeviceTimer::~CUDADeviceTimer() {
  const auto &cuda_context = CUDADeviceContext::cast(context());

  CUDA_CHECK(cuda_context.cuda_driver().EventDestroy(start_));
  CUDA_CHECK(cuda_context.cuda_driver().EventDestroy(stop_));
}

CUevent CUDADeviceTimer::cuda_start_event() const { return start_; }
CUevent CUDADeviceTimer::cuda_stop_event() const { return stop_; }

const CUDADeviceTimer &CUDADeviceTimer::cast(const DeviceTimer &timer) {
  return static_cast<const CUDADeviceTimer &>(timer);
}

// ===== CUDADeviceContext ===== //
CUDADeviceContext::CUDADeviceContext(const CUDADevice &device,
                                     CUcontext context, CUstream stream)
    : DeviceContext(device), driver_(device.driver()), context_(context),
      stream_(stream) {}

CUDADeviceContext::~CUDADeviceContext() {
  CUDA_CHECK(setCurrent());
  // We do not need to synchronize on possible work queued on this stream.
  // Resources will be released automatically once work has been completed.
  CUDA_CHECK(driver_.StreamDestroy(stream_));
  // Destroy the function map before destroying the context.
  func_map_.clear();
  CUDA_CHECK(driver_.CtxDestroy(context_));
}

ErrorOr<void *> CUDADeviceContext::malloc_host(size_t size) const {
  void *ptr;
  CUDA_CHECK_ERR(driver_.MemAllocHost(&ptr, size));
  return ptr;
}

ErrorOrSuccess CUDADeviceContext::free_host(void *ptr) const {
  CUDA_CHECK_ERR(driver_.MemFreeHost(ptr));
  return success();
}

ErrorOr<DeviceBufferRef> CUDADeviceContext::createBuffer(size_t len,
                                                         size_t elem_size,
                                                         SyncMode mode) const {
  CUDA_CHECK_ERR(setCurrent());

  size_t bytesize = len * elem_size;
  CUdeviceptr dptr;
  if (mode == ASYNC) {
    CUDA_CHECK_ERR(driver_.MemAllocAsync(&dptr, bytesize, stream_));
  } else {
    CUDA_CHECK_ERR(driver_.MemAlloc(&dptr, bytesize));
  }
  return CUDADeviceBufferRef::create(*this, len, elem_size, dptr);
}

DeviceBufferRef CUDADeviceContext::createBufferOwning(void *ptr, size_t len,
                                                      size_t elem_size,
                                                      bool owning) const {
  CUdeviceptr dptr = static_cast<CUdeviceptr>(ptr);
  return CUDADeviceBufferRef::create(*this, len, elem_size, dptr, owning);
}

ErrorOrSuccess CUDADeviceContext::copyToDevice(const DeviceBuffer &dst,
                                               const void *src,
                                               SyncMode mode) const {
  CUDA_CHECK_ERR(setCurrent());

  CUdeviceptr dst_ptr = CUDADeviceBuffer::cast(dst).cuda_ptr();
  size_t byte_size = dst.len() * dst.elem_size();
  if (mode == ASYNC) {
    CUDA_CHECK_ERR(driver_.MemcpyHtoDAsync(dst_ptr, src, byte_size, stream_));
  } else {
    CUDA_CHECK_ERR(driver_.MemcpyHtoD(dst_ptr, src, byte_size));
  }
  return success();
}

ErrorOrSuccess CUDADeviceContext::copyFromDevice(void *dst,
                                                 const DeviceBuffer &src,
                                                 SyncMode mode) const {
  CUDA_CHECK_ERR(setCurrent());

  CUdeviceptr src_ptr = CUDADeviceBuffer::cast(src).cuda_ptr();
  size_t byte_size = src.len() * src.elem_size();
  if (mode == ASYNC) {
    CUDA_CHECK_ERR(driver_.MemcpyDtoHAsync(dst, src_ptr, byte_size, stream_));
  } else {
    CUDA_CHECK_ERR(driver_.MemcpyDtoH(dst, src_ptr, byte_size));
  }
  return success();
}

ErrorOr<std::unique_ptr<void, std::function<void(void *)>>>
allocHostPtr(size_t byteSize, const CUDADriver &driver) {
  auto mallocHostDtor = [&](void *ptr) -> void {
    CUDA_CHECK(driver.MemFreeHost(ptr));
  };
  void *tmp;
  CUDA_CHECK_ERR(driver.MemAllocHost(&tmp, byteSize));
  return std::unique_ptr<void, decltype(mallocHostDtor)>(tmp, mallocHostDtor);
}

ErrorOrSuccess CUDADeviceContext::copyDeviceToDevice(const DeviceBuffer &dst,
                                                     const DeviceBuffer &src,
                                                     SyncMode mode) const {
  CUDA_CHECK_ERR(setCurrent());

  const auto &src_buffer = CUDADeviceBuffer::cast(src);
  const auto &dst_buffer = CUDADeviceBuffer::cast(dst);
  const auto &src_ctx = CUDADeviceContext::cast(src_buffer.context());
  const auto &dst_ctx = CUDADeviceContext::cast(dst_buffer.context());

  if (this->context_ != dst_ctx.context_) {
    RAISE_ERROR("device context does not match context of dst");
  }

  CUdeviceptr src_ptr = src_buffer.cuda_ptr();
  CUdeviceptr dst_ptr = dst_buffer.cuda_ptr();
  size_t dst_byte_size = dst.len() * dst.elem_size();
  size_t src_byte_size = src.len() * src.elem_size();
  if (src_byte_size < dst_byte_size) {
    RAISE_ERROR("not enough data in src");
  }
  if (src_ctx.context_ != dst_ctx.context_) {
    if (mode == ASYNC) {
      CUDA_CHECK_ERR(driver_.MemcpyPeerAsync(dst_ptr, dst_ctx.context_, src_ptr,
                                             src_ctx.context_, dst_byte_size,
                                             stream_));
    } else {
      CUDA_CHECK_ERR(driver_.MemcpyPeer(dst_ptr, dst_ctx.context_, src_ptr,
                                        src_ctx.context_, dst_byte_size));
    }
  } else {
    if (mode == ASYNC) {
      CUDA_CHECK_ERR(
          driver_.MemcpyDtoDAsync(dst_ptr, src_ptr, dst_byte_size, stream_));
    } else {
      CUDA_CHECK_ERR(driver_.MemcpyDtoD(dst_ptr, src_ptr, dst_byte_size));
    }
  }
  return success();
}

ErrorOrSuccess CUDADeviceContext::setMemory(const DeviceBuffer &dst,
                                            uint32_t val, size_t elem_size,
                                            SyncMode mode) const {
  CUDA_CHECK_ERR(setCurrent());

  void *ptr = CUDADeviceBuffer::cast(dst).cuda_ptr();
  size_t len = dst.len();
  if (elem_size == 1) {
    if (mode == ASYNC) {
      CUDA_CHECK_ERR(
          driver_.MemsetD8Async(ptr, static_cast<uint8_t>(val), len, stream_));
    } else {
      CUDA_CHECK_ERR(driver_.MemsetD8(ptr, static_cast<uint8_t>(val), len));
    }
  } else if (elem_size == 2) {
    if (mode == ASYNC) {
      CUDA_CHECK_ERR(driver_.MemsetD16Async(ptr, static_cast<uint16_t>(val),
                                            len, stream_));
    } else {
      CUDA_CHECK_ERR(driver_.MemsetD16(ptr, static_cast<uint16_t>(val), len));
    }
  } else {
    if (mode == ASYNC) {
      CUDA_CHECK_ERR(driver_.MemsetD32Async(ptr, val, len, stream_));
    } else {
      CUDA_CHECK_ERR(driver_.MemsetD32(ptr, val, len));
    }
  }
  return success();
}

ErrorOr<DeviceFunctionRef> CUDADeviceContext::loadFunction(
    const char *function_name,
    const DeviceFunction::SignatureBuilder &signature, const void *data,
    int32_t max_registers, int32_t threads_per_block, int32_t cache_mode,
    int32_t cache_config, int32_t max_dynamic_shared_bytes,
    const char *debug_level, int32_t optimization_level) const {
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto iter = func_map_.find(function_name);
    if (iter != func_map_.end()) {
      CUDAModuleFunctionRef module_func = iter->second;
      return CUDADeviceFunctionRef::create(*this, signature,
                                           std::move(module_func));
    }
  }

  // TODO(iposva): Remove once not needed for debugging.
  if (false) {
    llvm::errs() << "CUDADeviceContext::loadFunction(" << function_name
                 << "):\n"
                 << reinterpret_cast<const char *>(data) << "\n";
  }
  CUmodule module;
  unsigned int num_options = 0;
  CUjit_option option_kind[10];
  uintptr_t option_values[10];

  CUDA_CHECK_ERR(setCurrent());

  if (max_registers >= 0) {
    option_kind[num_options] = CUjit_option::JIT_MAX_REGISTERS;
    option_values[num_options] = max_registers;
    num_options++;
  }

  if (threads_per_block >= 0) {
    option_kind[num_options] = CUjit_option::JIT_THREADS_PER_BLOCK;
    option_values[num_options] = threads_per_block;
    num_options++;
  }

  if (cache_mode >= 0) {
    option_kind[num_options] = CUjit_option::JIT_CACHE_MODE;
    option_values[num_options] = cache_mode;
    num_options++;
  }

  if (strcmp(debug_level, "full") == 0) {
    option_kind[num_options] = CUjit_option::JIT_GENERATE_DEBUG_INFO;
    option_values[num_options] = 1;
    num_options++;
  } else if (strcmp(debug_level, "line-tables") == 0) {
    option_kind[num_options] = CUjit_option::JIT_GENERATE_LINE_INFO;
    option_values[num_options] = 1;
    num_options++;
  }

  if (optimization_level >= 0) {
    if (optimization_level < 3) {
      // Note that NVidia's optimization level goes up to (and defaults to)
      // 4, while ours only goes up to (and defaults to) 3.
      if ((strcmp(debug_level, "full") == 0) && (optimization_level > 0)) {
        // Note that the PTX compiler complains if debug_level=full and
        // optimization level is something other than 0 or default.
        RAISE_ERROR("debug_level of \"full\" requires optimization_level of "
                    "zero (or default)");
      }
      option_kind[num_options] = CUjit_option::JIT_OPTIMIZATION_LEVEL;
      option_values[num_options] = optimization_level;
      num_options++;
    }
  }

  if (num_options > 0) {
    CUDA_CHECK_ERR(
        driver_.ModuleLoadDataEx(&module, data, num_options, option_kind,
                                 reinterpret_cast<void **>(option_values)));
  } else {
    CUDA_CHECK_ERR(driver_.ModuleLoadData(&module, data));
  }

  CUfunction function;
  CUDA_CHECK_ERR(driver_.ModuleGetFunction(&function, module, function_name));

  if (cache_config >= 0) {
    CUDA_CHECK_ERR(driver_.FuncSetCacheConfig(
        function, static_cast<CUfunc_cache>(cache_config)));
  }

  if (max_dynamic_shared_bytes >= 0) {
    CUDA_CHECK_ERR(driver_.FuncSetAttribute(
        function,
        CUfunction_attribute::FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        max_dynamic_shared_bytes));
  }

  // Allocate func outside of the lock region to reduce the time spent under
  // lock. If this function_name has been registered concurrently, then we will
  // release this ref outside of the lock region before returning.
  auto module_func = CUDAModuleFunctionRef::create(
      std::string(function_name), this->cuda_driver(), module, function);
  CUDADeviceFunctionRef result;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto iter = func_map_.find(function_name);
    if (iter != func_map_.end()) {
      result = CUDADeviceFunctionRef::create(
          *this, signature,
          CUDAModuleFunctionRef::copy(iter->second.getPointer()));
    } else {
      func_map_.insert({module_func->name(), module_func});
      result = CUDADeviceFunctionRef::create(*this, signature,
                                             std::move(module_func));
    }
  }
  return result;
}

ErrorOrSuccess CUDADeviceContext::enqueueFunctionExec(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    const DeviceFunction::Arguments &args) const {
  if (args.hasError()) {
    return Error(args.getError().getError());
  }

  const auto &cuda_func = CUDADeviceFunction::cast(func);

  CUDA_CHECK_ERR(setCurrent());
  return cuda_func.cuda_callKernel(
      gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
      CUDADeviceFunctionArguments::cast(args), stream_);
}

ErrorOrSuccess CUDADeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t num_attrs, void **args) const {
  const auto &cuda_func = CUDADeviceFunction::cast(func);

  CUDA_CHECK_ERR(setCurrent());
  return cuda_func.cuda_callKernelDirect(
      gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
      reinterpret_cast<CUlaunchAttribute *>(attrs), num_attrs, args, stream_);
}

ErrorOr<DeviceStreamRef> CUDADeviceContext::stream() const {
  return CUDADeviceStreamRef::create(*this, stream_);
}

ErrorOrSuccess CUDADeviceContext::synchronize() const {
  CUDA_CHECK_ERR(setCurrent());
  CUDA_CHECK_ERR(driver_.StreamSynchronize(stream_));
  return success();
}

ErrorOr<DeviceTimerRef> CUDADeviceContext::startTimer() const {
  CUevent start;
  CUevent stop;

  CUDA_CHECK_ERR(setCurrent());
  CUDA_CHECK_ERR(driver_.EventCreate(&start, 0));
  CUDA_CHECK_ERR(driver_.EventCreate(&stop, 0));

  auto timer = CUDADeviceTimerRef::create(*this, start, stop);

  CUDA_CHECK_ERR(driver_.EventRecord(timer->cuda_start_event(), stream_));

  return timer;
}

ErrorOr<std::chrono::nanoseconds>
CUDADeviceContext::stopTimer(const DeviceTimer &timer) const {
  const auto &cuda_timer = CUDADeviceTimer::cast(timer);

  CUDA_CHECK_ERR(setCurrent());
  CUDA_CHECK_ERR(driver_.EventRecord(cuda_timer.cuda_stop_event(), stream_));
  CUDA_CHECK_ERR(driver_.EventSynchronize(cuda_timer.cuda_stop_event()));

  float millis;
  CUDA_CHECK_ERR(driver_.EventElapsedTime(
      &millis, cuda_timer.cuda_start_event(), cuda_timer.cuda_stop_event()));

  constexpr double nanos_per_millisecond = 1000.0 * 1000.0;
  double nanos = static_cast<double>(millis) * nanos_per_millisecond;
  return std::chrono::nanoseconds(static_cast<int64_t>(nanos));
}

ErrorOrSuccess
CUDADeviceContext::enablePeerAccess(const DeviceContext &peer) const {
  CUDA_CHECK_ERR(setCurrent());
  CUDA_CHECK_ERR(driver_.CtxEnablePeerAccess(
      CUDADeviceContext::cast(peer).cuda_context(), 0));
  return success();
}

ErrorOr<int> CUDADeviceContext::getDriverVersion() const {
  int driver_version = 0;
  CUDA_CHECK_ERR(driver_.DriverGetVersion(&driver_version));
  return driver_version;
}

ErrorOr<DeviceMemInfo> CUDADeviceContext::getMemoryInfo() const {
  DeviceMemInfo mem_info;
  CUDA_CHECK_ERR(setCurrent());
  CUDA_CHECK_ERR(driver_.GetMemInfo(&mem_info.free, &mem_info.total));
  return mem_info;
}

const CUDADriver &CUDADeviceContext::cuda_driver() const { return driver_; }

CUstream CUDADeviceContext::cuda_stream() const { return stream_; }

CUcontext CUDADeviceContext::cuda_context() const { return context_; }

CUDADriver::Result CUDADeviceContext::setCurrent() const {
  return driver_.CtxSetCurrent(context_);
}

const CUDADeviceContext &CUDADeviceContext::cast(const DeviceContext &ctx) {
  return static_cast<const CUDADeviceContext &>(ctx);
}

const CUDADeviceContext *
CUDADeviceContext::verified_cast(const DeviceContext *ctx) {
  // TODO(iposva): Implement verification of the underlying device type.
  return static_cast<const CUDADeviceContext *>(ctx);
}

ErrorOrSuccess CUDADeviceContext::isCompatibleWithMAX() const {
  // We only support CUDA architectures sm_80 and above.
  auto compute = computeCapability();
  if (compute.isError())
    return compute.takeError();
  constexpr int MIN_COMPUTE_CAPABILITY = 80;
  if (*compute < MIN_COMPUTE_CAPABILITY) {
    return Error("MAX only supports CUDA Ampere architectures or higher");
  }

  // We only support CUDA versions 12.0 and above.
  int driver_version = 0;
  constexpr int MIN_DRIVER_VERSION = 12.0;
  CUDA_CHECK_ERR(driver_.DriverGetVersion(&driver_version));
  if (driver_version < MIN_DRIVER_VERSION) {
    return Error("MAX not supported using current GPU driver");
  }

  return success();
}

ErrorOr<int> CUDADeviceContext::getAttribute(Attribute attr) const {
  int value = 0;
  CUDA_CHECK_ERR(
      driver_.DeviceGetAttribute(&value, static_cast<CUdevice_attribute>(attr),
                                 CUDADevice::cast(device()).cuda_device()));
  return value;
}

ErrorOr<int> CUDADeviceContext::computeCapability() const {
  int compute_major = 0, compute_minor = 0;
  CUDA_CHECK_ERR(driver_.DeviceGetAttribute(
      &compute_major, CUdevice_attribute::COMPUTE_CAPABILITY_MAJOR,
      CUDADevice::cast(device()).cuda_device()));
  CUDA_CHECK_ERR(driver_.DeviceGetAttribute(
      &compute_minor, CUdevice_attribute::COMPUTE_CAPABILITY_MINOR,
      CUDADevice::cast(device()).id()));
  return 10 * compute_major + compute_minor;
}

ErrorOr<CUDADeviceContextRef> CUDADeviceContext::create(int id) {
  ErrorOr<const CUDADevice &> err_or_device = CUDADevice::get(id);
  if (err_or_device) {
    return err_or_device.takeError();
  }
  const CUDADevice &device = *err_or_device;
  const CUDADriver &driver = device.driver();

  CUcontext context;
  CUDA_CHECK_ERR(driver.CtxCreate(&context, 0, device.cuda_device()));

  CUstream stream;
  CUDA_CHECK_ERR(driver.StreamCreate(&stream, 0));

  return CUDADeviceContextRef::create(device, context, stream);
}

} // namespace M::AsyncRT
