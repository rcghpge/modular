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

#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>

// TODO(iposva) : Switch error handling to using `std::source_location` once we
// are using C++ 20.
namespace M::AsyncRT {
namespace {
Error formatError(const char *file, int line, const char *func,
                  CUDADriver::Result result) {
  return Error(llvm::Twine(file) + ":" + llvm::Twine(line) +
               " - CUDA call in " + func + " failed: " + llvm::Twine(result));
}
} // namespace
} // namespace M::AsyncRT

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
      std::cerr << __FILE__ << ":" << __LINE__ << " - CUDA call in "           \
                << __FUNCTION__ << " failed: " << cuda_result << "\n";         \
      abort();                                                                 \
    }                                                                          \
  } while (0)

namespace M::AsyncRT {

// ===== CUDADevice ===== //
CUDADevice::CUDADevice(const CUDADriver &driver, CUdevice id)
    : Device(), driver_(driver), id_(id) {}

const CUDADriver &CUDADevice::driver() const { return driver_; }

int CUDADevice::id() const { return static_cast<int>(id_); }

std::string CUDADevice::name() const {
  char char_data[256];

  CUDADriver::Result cuda_result =
      driver_.DeviceGetName(char_data, sizeof(char_data), id_);
  if (cuda_result == CUDADriver::SUCCESS) {
    // TODO(iposva): Possibly add the device id here.
    return std::string(char_data);
  }
  return "cuda";
}

ErrorOr<int> CUDADevice::numberOfDevices() {
  ErrorOr<const CUDADriver &> err_driver = CUDADriver::get();
  if (err_driver) {
    return Error(llvm::Twine("CUDA Driver initialization failed: ") +
                 err_driver.getError());
  }

  int count = 0;
  CUDA_CHECK_ERR(err_driver.get().DeviceGetCount(&count));
  return count;
}

// ===== CUDADeviceBuffer ===== //
CUDADeviceBuffer::CUDADeviceBuffer(const CUDADeviceContext &context, size_t len,
                                   size_t elem_size, CUdeviceptr dptr)
    : DeviceBuffer(context, len, elem_size), dptr_(dptr) {}

CUDADeviceBuffer::~CUDADeviceBuffer() {
  const auto &cuda_context = CUDADeviceContext::cast(context());

  CUDA_CHECK(cuda_context.cuda_driver().MemFreeAsync(
      dptr_, cuda_context.cuda_stream()));
}

CUdeviceptr CUDADeviceBuffer::cuda_ptr() const { return dptr_; }
void *CUDADeviceBuffer::ptr() const { return cuda_ptr(); }

const CUDADeviceBuffer &CUDADeviceBuffer::cast(const DeviceBuffer &buf) {
  return static_cast<const CUDADeviceBuffer &>(buf);
}

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

// ===== CUDADeviceFunction ===== //
CUDADeviceFunction::CUDADeviceFunction(
    const CUDADeviceContext &context, std::string name,
    const DeviceFunction::SignatureBuilder &signature, CUmodule module,
    CUfunction function)
    : DeviceFunction(context, signature), module_(module), function_(function),
      name_(std::move(name)) {}

CUDADeviceFunction::~CUDADeviceFunction() {
  const auto &cuda_context = CUDADeviceContext::cast(context());

  CUDA_CHECK(cuda_context.cuda_driver().ModuleUnload(module_));
}

DeviceFunction::Arguments *
CUDADeviceFunction::allocateArguments(const DeviceFunction &fn) const {
  return new CUDADeviceFunctionArguments(CUDADeviceFunction::cast(fn));
}

ErrorOrSuccess CUDADeviceFunction::cuda_callKernel(
    uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
    uint32_t blockY, uint32_t blockZ, const CUDADeviceFunctionArguments &args,
    CUstream stream) const {
  const auto &cuda_context = CUDADeviceContext::cast(context());

  CUDA_CHECK_ERR(cuda_context.cuda_driver().LaunchKernel(
      function_, gridX, gridY, gridZ, blockX, blockY, blockZ, 0, stream,
      args.cuda_args().data(), nullptr));
  return success();
}

ErrorOrSuccess CUDADeviceFunction::cuda_callKernelDirect(
    uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
    uint32_t blockY, uint32_t blockZ, void **args, CUstream stream) const {
  const auto &cuda_context = CUDADeviceContext::cast(context());

  CUDA_CHECK_ERR(cuda_context.cuda_driver().LaunchKernel(
      function_, gridX, gridY, gridZ, blockX, blockY, blockZ, 0, stream, args,
      nullptr));
  return success();
}

const std::string &CUDADeviceFunction::name() const { return name_; }

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
  // We do not need to synchronize on possible work queued on this stream.
  // Resources will be released automatically once work has been completed.
  CUDA_CHECK(driver_.StreamDestroy(stream_));
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

ErrorOr<DeviceBufferRef>
CUDADeviceContext::createBuffer(size_t len, size_t elem_size) const {
  CUDA_CHECK_ERR(setCurrent());

  size_t bytesize = len * elem_size;
  CUdeviceptr dptr;
  CUDA_CHECK_ERR(driver_.MemAllocAsync(&dptr, bytesize, stream_));
  return CUDADeviceBufferRef::create(*this, len, elem_size, dptr);
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

ErrorOrSuccess CUDADeviceContext::copyDeviceToDevice(const DeviceBuffer &dst,
                                                     const DeviceBuffer &src,
                                                     SyncMode mode) const {
  CUDA_CHECK_ERR(setCurrent());

  const auto &src_buffer = CUDADeviceBuffer::cast(src);
  const auto &dst_buffer = CUDADeviceBuffer::cast(dst);
  const auto &src_ctx = CUDADeviceContext::cast(src_buffer.context());
  const auto &dst_ctx = CUDADeviceContext::cast(dst_buffer.context());

  CUdeviceptr src_ptr = src_buffer.cuda_ptr();
  CUdeviceptr dst_ptr = dst_buffer.cuda_ptr();
  size_t byte_size = dst.len() * dst.elem_size();
  if (src_ctx.context_ != dst_ctx.context_) {
    if (mode == ASYNC) {
      CUDA_CHECK_ERR(driver_.MemcpyPeerAsync(dst_ptr, dst_ctx.context_, src_ptr,
                                             src_ctx.context_, byte_size,
                                             stream_));
    } else {
      CUDA_CHECK_ERR(driver_.MemcpyPeer(dst_ptr, dst_ctx.context_, src_ptr,
                                        src_ctx.context_, byte_size));
    }
  } else {
    if (mode == ASYNC) {
      CUDA_CHECK_ERR(
          driver_.MemcpyDtoDAsync(dst_ptr, src_ptr, byte_size, stream_));
    } else {
      CUDA_CHECK_ERR(driver_.MemcpyDtoD(dst_ptr, src_ptr, byte_size));
    }
  }
  return success();
}

ErrorOr<DeviceFunctionRef> CUDADeviceContext::compileFunction(
    const char *function_name,
    const DeviceFunction::SignatureBuilder &signature, const void *data) const {
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto iter = func_map_.find(function_name);
    if (iter != func_map_.end()) {
      CUDADeviceFunctionRef result = iter->second;
      return result;
    }
  }

  // TODO(iposva): Remove once not needed for debugging.
  if (false) {
    std::cerr << "CUDADeviceContext::compileFunction(" << function_name
              << "):\n"
              << reinterpret_cast<const char *>(data) << "\n";
  }
  CUmodule module;
  CUDA_CHECK_ERR(driver_.ModuleLoadData(&module, data));

  CUfunction function;
  CUDA_CHECK_ERR(driver_.ModuleGetFunction(&function, module, function_name));

  // Allocate func outside of the lock region to reduce the time spent under
  // lock. If this function_name has been registered concurrently, then we will
  // release this ref outside of the lock region before returning.
  auto func = CUDADeviceFunctionRef::create(*this, std::string(function_name),
                                            signature, module, function);
  CUDADeviceFunctionRef result;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto iter = func_map_.find(function_name);
    if (iter != func_map_.end()) {
      result = CUDADeviceFunctionRef::copy(iter->second.getPointer());
    } else {
      func_map_.insert({func->name(), func});
      result = std::move(func);
    }
  }
  return result;
}

ErrorOrSuccess CUDADeviceContext::enqueueFunctionExec(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ,
    const DeviceFunction::Arguments &args) const {
  if (args.hasError()) {
    return Error(args.getError().getError());
  }

  const auto &cuda_func = CUDADeviceFunction::cast(func);

  return cuda_func.cuda_callKernel(gridX, gridY, gridZ, blockX, blockY, blockZ,
                                   CUDADeviceFunctionArguments::cast(args),
                                   stream_);
}

ErrorOrSuccess CUDADeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, void **args) const {
  const auto &cuda_func = CUDADeviceFunction::cast(func);

  return cuda_func.cuda_callKernelDirect(gridX, gridY, gridZ, blockX, blockY,
                                         blockZ, args, stream_);
}

ErrorOrSuccess CUDADeviceContext::synchronize() const {
  CUDA_CHECK_ERR(driver_.StreamSynchronize(stream_));
  return success();
}

ErrorOr<DeviceTimerRef> CUDADeviceContext::startTimer() const {
  CUevent start;
  CUevent stop;

  CUDA_CHECK_ERR(driver_.EventCreate(&start, 0));
  CUDA_CHECK_ERR(driver_.EventCreate(&stop, 0));

  auto timer = CUDADeviceTimerRef::create(*this, start, stop);

  CUDA_CHECK_ERR(driver_.EventRecord(timer->cuda_start_event(), stream_));

  return timer;
}

ErrorOr<std::chrono::nanoseconds>
CUDADeviceContext::stopTimer(const DeviceTimer &timer) const {
  const auto &cuda_timer = CUDADeviceTimer::cast(timer);

  CUDA_CHECK_ERR(driver_.EventRecord(cuda_timer.cuda_stop_event(), stream_));
  CUDA_CHECK_ERR(driver_.EventSynchronize(cuda_timer.cuda_stop_event()));

  float millis;
  CUDA_CHECK_ERR(driver_.EventElapsedTime(
      &millis, cuda_timer.cuda_start_event(), cuda_timer.cuda_stop_event()));

  constexpr double nanos_per_millisecond = 1000.0 * 1000.0;
  double nanos = static_cast<double>(millis) * nanos_per_millisecond;
  return std::chrono::nanoseconds(static_cast<int64_t>(nanos));
}

ErrorOrSuccess CUDADeviceContext::getMemoryInfo(size_t *free,
                                                size_t *total) const {
  CUDA_CHECK_ERR(driver_.GetMemInfo(free, total));
  return success();
}

const CUDADriver &CUDADeviceContext::cuda_driver() const { return driver_; }

CUstream CUDADeviceContext::cuda_stream() const { return stream_; }

CUDADriver::Result CUDADeviceContext::setCurrent() const {
  return driver_.CtxSetCurrent(context_);
}

const CUDADeviceContext &CUDADeviceContext::cast(const DeviceContext &ctx) {
  return static_cast<const CUDADeviceContext &>(ctx);
}

ErrorOr<CUDADeviceContextRef> CUDADeviceContext::create(int id) {
  ErrorOr<const CUDADriver &> err_driver = CUDADriver::get();
  if (err_driver) {
    return Error(llvm::Twine("CUDA Driver initialization failed: ") +
                 err_driver.getError());
  }
  const CUDADriver &driver = err_driver.get();

  CUcontext context;
  CUDA_CHECK_ERR(driver.CtxCreate(&context, 0, id));

  CUstream stream;
  CUDA_CHECK_ERR(driver.StreamCreate(&stream, 0));

  return CUDADeviceContextRef::create(*CUDADeviceRef::create(driver, id),
                                      context, stream);
}

} // namespace M::AsyncRT
