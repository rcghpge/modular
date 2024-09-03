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

#include <iostream>
#include <string>

// TODO(iposva) : Switch to using `std::source_location` once we are using
// C++ 20.
#define CUDA_CHECK(cuda_call)                                                  \
  do {                                                                         \
    CUDADriver::Result cuda_result = cuda_call;                                \
    if (cuda_result != CUDADriver::SUCCESS) {                                  \
      std::cerr << __FILE__ << ":" << __LINE__ << " - CUDA call in "           \
                << __FUNCTION__ << " failed: " << cuda_result << "\n";         \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_RETURN(cuda_call, result)                                   \
  do {                                                                         \
    CUDADriver::Result cuda_result = cuda_call;                                \
    if (cuda_result != CUDADriver::SUCCESS) {                                  \
      std::cerr << __FILE__ << ":" << __LINE__ << " - CUDA call in "           \
                << __FUNCTION__ << " failed: " << cuda_result << "\n";         \
      abort();                                                                 \
      return result;                                                           \
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

void *CUDADeviceContext::malloc_host(size_t size) const {
  void *ptr;
  CUDA_CHECK_RETURN(driver_.MemAllocHost(&ptr, size), nullptr);
  return ptr;
}

void CUDADeviceContext::free_host(void *ptr) const {
  CUDA_CHECK(driver_.MemFreeHost(ptr));
}

DeviceBufferRef CUDADeviceContext::createBuffer(size_t len,
                                                size_t elem_size) const {
  size_t bytesize = len * elem_size;
  CUdeviceptr dptr;
  CUDA_CHECK_RETURN(driver_.MemAllocAsync(&dptr, bytesize, stream_),
                    CUDADeviceBufferRef());
  return CUDADeviceBufferRef::create(*this, len, elem_size, dptr);
}

void CUDADeviceContext::enqueueCopyToDevice(const DeviceBuffer &dst,
                                            const void *src) const {
  CUdeviceptr dst_ptr = CUDADeviceBuffer::cast(dst).cuda_ptr();
  size_t byte_size = dst.len() * dst.elem_size();
  CUDA_CHECK(driver_.MemcpyHtoDAsync(dst_ptr, src, byte_size, stream_));
}

void CUDADeviceContext::enqueueCopyFromDevice(void *dst,
                                              const DeviceBuffer &src) const {
  CUdeviceptr src_ptr = CUDADeviceBuffer::cast(src).cuda_ptr();
  size_t byte_size = src.len() * src.elem_size();
  CUDA_CHECK(driver_.MemcpyDtoHAsync(dst, src_ptr, byte_size, stream_));
}
void CUDADeviceContext::enqueueCopyDeviceToDevice(
    const DeviceBuffer &dst, const DeviceBuffer &src) const {
  CUdeviceptr src_ptr = CUDADeviceBuffer::cast(src).cuda_ptr();
  CUdeviceptr dst_ptr = CUDADeviceBuffer::cast(dst).cuda_ptr();
  size_t byte_size = dst.len() * dst.elem_size();
  CUDA_CHECK(driver_.MemcpyDtoDAsync(dst_ptr, src_ptr, byte_size, stream_));
}

void CUDADeviceContext::synchronize() const {
  CUDA_CHECK(driver_.StreamSynchronize(stream_));
}

const CUDADriver &CUDADeviceContext::cuda_driver() const { return driver_; }

CUstream CUDADeviceContext::cuda_stream() const { return stream_; }

const CUDADeviceContext &CUDADeviceContext::cast(const DeviceContext &ctx) {
  return static_cast<const CUDADeviceContext &>(ctx);
}

CUDADeviceContextRef CUDADeviceContext::create() {
  CUdevice id = 0; // By default we select the first device.
  const CUDADriver *driver = CUDADriver::get();
  if (driver == nullptr) {
    return CUDADeviceContextRef();
  }

  CUcontext context;
  CUDA_CHECK_RETURN(driver->CtxCreate(&context, 0, id), CUDADeviceContextRef());

  CUstream stream;
  CUDA_CHECK_RETURN(driver->StreamCreate(&stream, 0), CUDADeviceContextRef());

  return CUDADeviceContextRef::create(*CUDADeviceRef::create(*driver, id),
                                      context, stream);
}

} // namespace M::AsyncRT
