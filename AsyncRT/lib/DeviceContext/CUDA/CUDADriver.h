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

#include "Support/ErrorOr.h"

#include <stddef.h>

namespace M::AsyncRT {

using CUcontext = struct CUctx_st *;
using CUdevice = int;
using CUdeviceptr = void *;
using CUfunction = struct CUfunc_st *;
using CUmodule = struct CUmod_st *;
using CUstream = struct CUstream_st *;

class CUDADriver {
public:
  enum Result {
    SUCCESS = 0,
  };

  Result DeviceGetCount(int *count) const { return cuDeviceGetCount_(count); }

  Result DeviceGetName(char *name, int len, CUdevice dev) const {
    return cuDeviceGetName_(name, len, dev);
  }

  Result CtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) const {
    return cuCtxCreate_v2_(pctx, flags, dev);
  }
  Result CtxDestroy(CUcontext ctx) const { return cuCtxDestroy_(ctx); }
  Result CtxSetCurrent(CUcontext ctx) const { return cuCtxSetCurrent_(ctx); }

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
  Result MemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                         CUdeviceptr srcDevice, CUcontext srcContext,
                         size_t bytesize, CUstream hStream) const {
    return cuMemcpyPeerAsync_(dstDevice, dstContext, srcDevice, srcContext,
                              bytesize, hStream);
  }

  Result ModuleLoadData(CUmodule *module, const void *image) const {
    return cuModuleLoadData_(module, image);
  }
  Result ModuleUnload(CUmodule hmod) const { return cuModuleUnload_(hmod); }
  Result ModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                           const char *name) const {
    return cuModuleGetFunction_(hfunc, hmod, name);
  }

  Result LaunchKernel(CUfunction f, unsigned int gridDimX,
                      unsigned int gridDimY, unsigned int gridDimZ,
                      unsigned int blockDimX, unsigned int blockDimY,
                      unsigned int blockDimZ, unsigned int sharedMemBytes,
                      CUstream hStream, void **kernelParams,
                      void **extra) const {
    return cuLaunchKernel_(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                           blockDimY, blockDimZ, sharedMemBytes, hStream,
                           kernelParams, extra);
  }

  static ErrorOr<const CUDADriver &> get();

private:
  // Shared library name for the CUDA driver.
  static const char lib_name[];

  // Once created, the CUDADriver object cannot be deleted.
  CUDADriver() = default;
  ~CUDADriver() = delete;

  // Driver is only initialized from internal code.
  Result Init(unsigned int flags) const { return cuInit_(flags); }
  using tcuInit = Result(unsigned int);
  tcuInit *cuInit_;

  using tcuDeviceGetCount = Result(int *);
  tcuDeviceGetCount *cuDeviceGetCount_;

  using tcuDeviceGetName = Result(char *, int, CUdevice);
  tcuDeviceGetName *cuDeviceGetName_;

  using tcuCtxCreate_v2 = Result(CUcontext *, unsigned int, CUdevice);
  tcuCtxCreate_v2 *cuCtxCreate_v2_;

  using tcuCtxDestroy = Result(CUcontext);
  tcuCtxDestroy *cuCtxDestroy_;

  using tcuCtxSetCurrent = Result(CUcontext);
  tcuCtxSetCurrent *cuCtxSetCurrent_;

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

  using tcuMemcpyPeerAsync = Result(CUdeviceptr, CUcontext, CUdeviceptr,
                                    CUcontext, size_t, CUstream);
  tcuMemcpyPeerAsync *cuMemcpyPeerAsync_;

  using tcuModuleLoadData = Result(CUmodule *, const void *);
  tcuModuleLoadData *cuModuleLoadData_;

  using tcuModuleUnload = Result(CUmodule);
  tcuModuleUnload *cuModuleUnload_;

  using tcuModuleGetFunction = Result(CUfunction *, CUmodule, const char *);
  tcuModuleGetFunction *cuModuleGetFunction_;

  using tcuLaunchKernel = Result(CUfunction, unsigned int, unsigned int,
                                 unsigned int, unsigned int, unsigned int,
                                 unsigned int, unsigned int, CUstream, void **,
                                 void **);
  tcuLaunchKernel *cuLaunchKernel_;

  static ErrorOr<const CUDADriver &> allocAndInitDriver();
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_CUDA_CUDADRIVER_H
