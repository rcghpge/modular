//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the CUDA driver interface.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_LIB_DEVICECONTEXT_CUDA_CUDADRIVER_H
#define ASYNCRT_LIB_DEVICECONTEXT_CUDA_CUDADRIVER_H

#include <stddef.h>

namespace M::AsyncRT {

using CUdevice = int;
using CUdeviceptr = void *;
using CUcontext = struct CUctx_st *;
using CUstream = struct CUstream_st *;

class CUDADriver {
public:
  enum Result {
    SUCCESS = 0,
  };

  Result CtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) const {
    return cuCtxCreate_v2_(pctx, flags, dev);
  }
  Result CtxDestroy(CUcontext ctx) const { return cuCtxDestroy_(ctx); }

  Result StreamCreate(CUstream *phStream, unsigned int flags) const {
    return cuStreamCreate_(phStream, flags);
  }
  Result StreamDestroy(CUstream hStream) const {
    return cuStreamDestroy_(hStream);
  }
  Result StreamSynchronize(CUstream hStream) const {
    return cuStreamSynchronize_(hStream);
  }

  Result MemAllocHost(void **pp, size_t bytesize) const {
    return cuMemAllocHost_v2_(pp, bytesize);
  }
  Result MemFreeHost(void *p) const { return cuMemFreeHost_(p); }

  Result MemAllocAsync(CUdeviceptr *pdptr, size_t bytesize,
                       CUstream hStream) const {
    return cuMemAllocAsync_(pdptr, bytesize, hStream);
  }
  Result MemFreeAsync(CUdeviceptr dptr, CUstream hStream) const {
    return cuMemFreeAsync_(dptr, hStream);
  }

  Result MemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
                         size_t bytesize, CUstream hStream) const {
    return cuMemcpyHtoDAsync_v2_(dstDevice, srcHost, bytesize, hStream);
  }
  Result MemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t bytesize,
                         CUstream hStream) const {
    return cuMemcpyDtoHAsync_v2_(dstHost, srcDevice, bytesize, hStream);
  }
  Result MemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                         size_t bytesize, CUstream hStream) const {
    return cuMemcpyDtoDAsync_v2_(dstDevice, srcDevice, bytesize, hStream);
  }

  static const CUDADriver *get();

private:
  // Shared library name for the CUDA driver.
  static const char lib_name[];

  // Driver is only initialized from internal code.
  Result Init(unsigned int flags) const { return cuInit_(flags); }
  using tcuInit = Result(unsigned int);
  tcuInit *cuInit_;

  using tcuCtxCreate_v2 = Result(CUcontext *, unsigned int, CUdevice);
  tcuCtxCreate_v2 *cuCtxCreate_v2_;

  using tcuCtxDestroy = Result(CUcontext);
  tcuCtxDestroy *cuCtxDestroy_;

  using tcuStreamCreate = Result(CUstream *, unsigned int);
  tcuStreamCreate *cuStreamCreate_;

  using tcuStreamDestroy = Result(CUstream);
  tcuStreamDestroy *cuStreamDestroy_;

  using tcuStreamSynchronize = Result(CUstream);
  tcuStreamSynchronize *cuStreamSynchronize_;

  using tcuMemAllocHost_v2 = Result(void **, size_t);
  tcuMemAllocHost_v2 *cuMemAllocHost_v2_;

  using tcuMemFreeHost = Result(void *);
  tcuMemFreeHost *cuMemFreeHost_;

  using tcuMemAllocAsync = Result(CUdeviceptr *, size_t, CUstream);
  tcuMemAllocAsync *cuMemAllocAsync_;

  using tcuMemFreeAsync = Result(CUdeviceptr, CUstream);
  tcuMemFreeAsync *cuMemFreeAsync_;

  using tcuMemcpyHtoDAsync_v2 = Result(CUdeviceptr, const void *, size_t,
                                       CUstream);
  tcuMemcpyHtoDAsync_v2 *cuMemcpyHtoDAsync_v2_;

  using tcuMemcpyDtoHAsync_v2 = Result(void *, CUdeviceptr, size_t, CUstream);
  tcuMemcpyDtoHAsync_v2 *cuMemcpyDtoHAsync_v2_;

  using tcuMemcpyDtoDAsync_v2 = Result(CUdeviceptr, CUdeviceptr, size_t,
                                       CUstream);
  tcuMemcpyDtoDAsync_v2 *cuMemcpyDtoDAsync_v2_;

  CUDADriver() = default;

  static const CUDADriver *allocAndInitDriver();
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_CUDA_CUDADRIVER_H
