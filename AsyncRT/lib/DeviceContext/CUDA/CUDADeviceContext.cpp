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

CUDADevice::~CUDADevice() {}

const CUDADriver &CUDADevice::driver() const { return driver_; }

CUdevice CUDADevice::id() const { return id_; }

std::string CUDADevice::name() const {
  // TODO(iposva): Return the actual device name and possibly id.
  return "cuda";
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

const CUDADeviceBuffer &CUDADeviceBuffer::cast(const DeviceBuffer &buf) {
  return static_cast<const CUDADeviceBuffer &>(buf);
}

// ===== CUDADeviceFunctionArguments ===== //
CUDADeviceFunctionArguments::CUDADeviceFunctionArguments() {}
CUDADeviceFunctionArguments::~CUDADeviceFunctionArguments() {}

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
CUDADeviceFunctionArguments::cast(const DeviceFunctionArguments &args) {
  return static_cast<const CUDADeviceFunctionArguments &>(args);
}

// ===== CUDADeviceFunction ===== //
CUDADeviceFunction::CUDADeviceFunction(CUDADeviceContextRef context,
                                       std::string name, CUmodule module,
                                       CUfunction function)
    : DeviceFunction(std::move(context)), module_(module), function_(function),
      name_(std::move(name)) {}

CUDADeviceFunction::~CUDADeviceFunction() {
  const auto &cuda_context = CUDADeviceContext::cast(*context());

  CUDA_CHECK(cuda_context.cuda_driver().ModuleUnload(module_));
}

DeviceFunctionArguments *CUDADeviceFunction::allocateArguments() const {
  return new CUDADeviceFunctionArguments();
}

ErrorOrSuccess CUDADeviceFunction::cuda_callKernel(
    uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
    uint32_t blockY, uint32_t blockZ, const CUDADeviceFunctionArguments &args,
    CUstream stream) const {
  const auto &cuda_context = CUDADeviceContext::cast(*context());

  CUDA_CHECK_ERR(cuda_context.cuda_driver().LaunchKernel(
      function_, gridX, gridY, gridZ, blockX, blockY, blockZ, 0, stream,
      args.cuda_args().data(), nullptr));
  return success();
}

const std::string &CUDADeviceFunction::name() const { return name_; }

const CUDADeviceFunction &CUDADeviceFunction::cast(const DeviceFunction &func) {
  return static_cast<const CUDADeviceFunction &>(func);
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
  size_t bytesize = len * elem_size;
  CUdeviceptr dptr;
  CUDA_CHECK_ERR(driver_.MemAllocAsync(&dptr, bytesize, stream_));
  return CUDADeviceBufferRef::create(*this, len, elem_size, dptr);
}

ErrorOrSuccess CUDADeviceContext::enqueueCopyToDevice(const DeviceBuffer &dst,
                                                      const void *src) const {
  CUdeviceptr dst_ptr = CUDADeviceBuffer::cast(dst).cuda_ptr();
  size_t byte_size = dst.len() * dst.elem_size();
  CUDA_CHECK_ERR(driver_.MemcpyHtoDAsync(dst_ptr, src, byte_size, stream_));
  return success();
}

ErrorOrSuccess
CUDADeviceContext::enqueueCopyFromDevice(void *dst,
                                         const DeviceBuffer &src) const {
  CUdeviceptr src_ptr = CUDADeviceBuffer::cast(src).cuda_ptr();
  size_t byte_size = src.len() * src.elem_size();
  CUDA_CHECK_ERR(driver_.MemcpyDtoHAsync(dst, src_ptr, byte_size, stream_));
  return success();
}

ErrorOrSuccess
CUDADeviceContext::enqueueCopyDeviceToDevice(const DeviceBuffer &dst,
                                             const DeviceBuffer &src) const {
  CUdeviceptr src_ptr = CUDADeviceBuffer::cast(src).cuda_ptr();
  CUdeviceptr dst_ptr = CUDADeviceBuffer::cast(dst).cuda_ptr();
  size_t byte_size = dst.len() * dst.elem_size();
  CUDA_CHECK_ERR(driver_.MemcpyDtoDAsync(dst_ptr, src_ptr, byte_size, stream_));
  return success();
}

ErrorOr<DeviceFunctionRef>
CUDADeviceContext::compileFunction(const char *function_name,
                                   const void *data) const {
  std::cerr << "CUDADeviceContext::compileFunction(" << function_name << "):\n"
            << reinterpret_cast<const char *>(data) << "\n";
  CUmodule module;
  CUDA_CHECK_ERR(driver_.ModuleLoadData(&module, data));

  CUfunction function;
  CUDA_CHECK_ERR(driver_.ModuleGetFunction(&function, module, function_name));

  return CUDADeviceFunctionRef::create(CUDADeviceContextRef::copy(this),
                                       std::string(function_name), module,
                                       function);
}

ErrorOrSuccess CUDADeviceContext::enqueueFunctionExec(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ,
    const DeviceFunctionArguments &args) const {
  const auto &cuda_func = CUDADeviceFunction::cast(func);

  return cuda_func.cuda_callKernel(gridX, gridY, gridZ, blockX, blockY, blockZ,
                                   CUDADeviceFunctionArguments::cast(args),
                                   stream_);
}

ErrorOrSuccess CUDADeviceContext::synchronize() const {
  CUDA_CHECK_ERR(driver_.StreamSynchronize(stream_));
  return success();
}

const CUDADriver &CUDADeviceContext::cuda_driver() const { return driver_; }

CUstream CUDADeviceContext::cuda_stream() const { return stream_; }

const CUDADeviceContext &CUDADeviceContext::cast(const DeviceContext &ctx) {
  return static_cast<const CUDADeviceContext &>(ctx);
}

ErrorOr<CUDADeviceContextRef> CUDADeviceContext::create() {
  CUdevice id = 0; // By default we select the first device.
  const CUDADriver *driver = CUDADriver::get();
  if (driver == nullptr) {
    // TODO(iposva): Surface driver initialization error.
    return Error("Driver initialization failed.");
  }

  CUcontext context;
  CUDA_CHECK_ERR(driver->CtxCreate(&context, 0, id));

  CUstream stream;
  CUDA_CHECK_ERR(driver->StreamCreate(&stream, 0));

  return CUDADeviceContextRef::create(*CUDADeviceRef::create(*driver, id),
                                      context, stream);
}

} // namespace M::AsyncRT
