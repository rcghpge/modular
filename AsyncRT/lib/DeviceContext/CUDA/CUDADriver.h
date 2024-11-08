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
using CUevent = struct CUevent_st *;

enum class CUdevice_attribute {
  COMPUTE_CAPABILITY_MAJOR = 75,
  COMPUTE_CAPABILITY_MINOR = 76,
};

enum class CUjit_option {
  JIT_MAX_REGISTERS = 0,
  JIT_THREADS_PER_BLOCK = 1,
  JIT_OPTIMIZATION_LEVEL = 7,
  JIT_GENERATE_DEBUG_INFO = 11,
  JIT_GENERATE_LINE_INFO = 13,
  JIT_CACHE_MODE = 14,
};

enum class CUfunction_attribute {
  FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
};

enum class CUfunc_cache {
  FUNC_CACHE_PREFER_NONE = 0,
  FUNC_CACHE_PREFER_SHARED = 1,
  FUNC_CACHE_PREFER_L1 = 2,
  FUNC_CACHE_PREFER_EQUAL = 3,
};

enum class CUlaunchAttributeID {
  CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
  CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
};

using CUlaunchAttribute = struct CUlaunchAttribute_st {
  CUlaunchAttributeID id;
  char pad[8 - sizeof(CUlaunchAttributeID)];
  char value[64]; // a union of size 64 bytes
};

using CUlaunchConfig = struct CUlaunchConfig_st {
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;
  CUstream hStream;
  CUlaunchAttribute *attrs;
  unsigned int numAttrs;
};

class CUDADriver {
public:
  enum Result {
    SUCCESS = 0,
  };

  Result GetErrorName(Result error, const char **pStr) const {
    return cuGetErrorName_(error, pStr);
  }

  Result GetErrorString(Result error, const char **pStr) const {
    return cuGetErrorString_(error, pStr);
  }

  Result DriverGetVersion(int *version) const {
    return cuDriverGetVersion_(version);
  }

  Result DeviceGetAttribute(int *pi, CUdevice_attribute attrib,
                            CUdevice dev) const {
    return cuDeviceGetAttribute_(pi, attrib, dev);
  }

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

  Result MemAlloc(CUdeviceptr *pdptr, size_t bytesize) const {
    return cuMemAlloc_v2_(pdptr, bytesize);
  }
  Result MemFree(CUdeviceptr dptr) const { return cuMemFree_(dptr); }

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
  Result DeviceCanAccessPeer(int *canAccessPeer, CUdevice dev,
                             CUdevice peerDev) const {
    return cuDeviceCanAccessPeer_(canAccessPeer, dev, peerDev);
  }

  Result CtxEnablePeerAccess(CUcontext ctx, unsigned int flags) const {
    return cuCtxEnablePeerAccess_(ctx, flags);
  }
  Result CtxDisablePeerAccess(CUcontext ctx) const {
    return cuCtxDisablePeerAccess_(ctx);
  }

  Result MemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost,
                    size_t bytesize) const {
    return cuMemcpyHtoD_v2_(dstDevice, srcHost, bytesize);
  }
  Result MemcpyDtoH(void *dstHost, CUdeviceptr srcDevice,
                    size_t bytesize) const {
    return cuMemcpyDtoH_v2_(dstHost, srcDevice, bytesize);
  }
  Result MemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                    size_t bytesize) const {
    return cuMemcpyDtoD_v2_(dstDevice, srcDevice, bytesize);
  }
  Result MemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                    CUdeviceptr srcDevice, CUcontext srcContext,
                    size_t bytesize) const {
    return cuMemcpyPeer_(dstDevice, dstContext, srcDevice, srcContext,
                         bytesize);
  }

  Result MemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N,
                       CUstream hStream) const {
    return cuMemsetD8Async_(dstDevice, uc, N, hStream);
  }
  Result MemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N,
                        CUstream hStream) const {
    return cuMemsetD16Async_(dstDevice, us, N, hStream);
  }
  Result MemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N,
                        CUstream hStream) const {
    return cuMemsetD32Async_(dstDevice, ui, N, hStream);
  }

  Result MemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) const {
    return cuMemsetD8_v2_(dstDevice, uc, N);
  }
  Result MemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) const {
    return cuMemsetD16_v2_(dstDevice, us, N);
  }
  Result MemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) const {
    return cuMemsetD32_v2_(dstDevice, ui, N);
  }

  Result ModuleLoadData(CUmodule *module, const void *image) const {
    return cuModuleLoadData_(module, image);
  }
  Result ModuleLoadDataEx(CUmodule *module, const void *image,
                          unsigned int numOptions, CUjit_option *options,
                          void **optionValues) const {
    return cuModuleLoadDataEx_(module, image, numOptions, options,
                               optionValues);
  }
  Result ModuleGetGlobal(CUdeviceptr *dptr, size_t *sizeptr, CUmodule hmod,
                         const char *name) const {
    return cuModuleGetGlobal_v2_(dptr, sizeptr, hmod, name);
  }
  Result ModuleUnload(CUmodule hmod) const { return cuModuleUnload_(hmod); }
  Result ModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                           const char *name) const {
    return cuModuleGetFunction_(hfunc, hmod, name);
  }
  Result FuncGetAttribute(int *pi, CUfunction_attribute attrib,
                          CUfunction hfunc) const {
    return cuFuncGetAttribute_(pi, attrib, hfunc);
  }
  Result FuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib,
                          int value) const {
    return cuFuncSetAttribute_(hfunc, attrib, value);
  }
  Result FuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) const {
    return cuFuncSetCacheConfig_(hfunc, config);
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
  Result LaunchKernelEx(const CUlaunchConfig *config, CUfunction hfunc,
                        void **kernelParams, void **extra) const {
    return cuLaunchKernelEx_(config, hfunc, kernelParams, extra);
  }

  Result EventCreate(CUevent *phEvent, unsigned int flags) const {
    return cuEventCreate_(phEvent, flags);
  }
  Result EventDestroy(CUevent hEvent) const { return cuEventDestroy_(hEvent); }
  Result EventRecord(CUevent hEvent, CUstream hStream) const {
    return cuEventRecord_(hEvent, hStream);
  }
  Result EventQuery(CUevent hEvent) const { return cuEventQuery_(hEvent); }
  Result EventSynchronize(CUevent hEvent) const {
    return cuEventSynchronize_(hEvent);
  }
  Result EventElapsedTime(float *pMilliseconds, CUevent hStart,
                          CUevent hEnd) const {
    return cuEventElapsedTime_(pMilliseconds, hStart, hEnd);
  }

  Result GetMemInfo(size_t *free, size_t *total) const {
    return cuMemGetInfo_v2_(free, total);
  }

  static ErrorOr<const CUDADriver &> get();

private:
  // Shared library name for the CUDA driver.
  static const char lib_name[];

  // Once created, the CUDADriver object cannot be deleted.
  CUDADriver() = default;
  ~CUDADriver() = default;

  // Driver is only initialized from internal code.
  Result Init(unsigned int flags) const { return cuInit_(flags); }
  using tcuInit = Result(unsigned int);
  tcuInit *cuInit_;

  using tcuGetErrorName = Result(Result, const char **);
  tcuGetErrorName *cuGetErrorName_;

  using tcuGetErrorString = Result(Result, const char **);
  tcuGetErrorString *cuGetErrorString_;

  using tcuDriverGetVersion = Result(int *);
  tcuDriverGetVersion *cuDriverGetVersion_;

  using tcuDeviceGetAttribute = Result(int *, CUdevice_attribute, CUdevice);
  tcuDeviceGetAttribute *cuDeviceGetAttribute_;

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

  using tcuMemAlloc_v2 = Result(CUdeviceptr *, size_t);
  tcuMemAlloc_v2 *cuMemAlloc_v2_;

  using tcuMemFree = Result(CUdeviceptr);
  tcuMemFree *cuMemFree_;

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

  using tcuDeviceCanAccessPeer = Result(int *, CUdevice, CUdevice);
  tcuDeviceCanAccessPeer *cuDeviceCanAccessPeer_;

  using tcuCtxEnablePeerAccess = Result(CUcontext, unsigned int);
  tcuCtxEnablePeerAccess *cuCtxEnablePeerAccess_;

  using tcuCtxDisablePeerAccess = Result(CUcontext);
  tcuCtxDisablePeerAccess *cuCtxDisablePeerAccess_;

  using tcuMemcpyHtoD_v2 = Result(CUdeviceptr, const void *, size_t);
  tcuMemcpyHtoD_v2 *cuMemcpyHtoD_v2_;

  using tcuMemcpyDtoH_v2 = Result(void *, CUdeviceptr, size_t);
  tcuMemcpyDtoH_v2 *cuMemcpyDtoH_v2_;

  using tcuMemcpyDtoD_v2 = Result(CUdeviceptr, CUdeviceptr, size_t);
  tcuMemcpyDtoD_v2 *cuMemcpyDtoD_v2_;

  using tcuMemcpyPeer = Result(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext,
                               size_t);
  tcuMemcpyPeer *cuMemcpyPeer_;

  using tcuMemsetD8Async = Result(CUdeviceptr, unsigned char, size_t, CUstream);
  tcuMemsetD8Async *cuMemsetD8Async_;

  using tcuMemsetD16Async = Result(CUdeviceptr, unsigned short, size_t,
                                   CUstream);
  tcuMemsetD16Async *cuMemsetD16Async_;

  using tcuMemsetD32Async = Result(CUdeviceptr, unsigned int, size_t, CUstream);
  tcuMemsetD32Async *cuMemsetD32Async_;

  using tcuMemsetD8_v2 = Result(CUdeviceptr, unsigned char, size_t);
  tcuMemsetD8_v2 *cuMemsetD8_v2_;

  using tcuMemsetD16_v2 = Result(CUdeviceptr, unsigned short, size_t);
  tcuMemsetD16_v2 *cuMemsetD16_v2_;

  using tcuMemsetD32_v2 = Result(CUdeviceptr, unsigned int, size_t);
  tcuMemsetD32_v2 *cuMemsetD32_v2_;

  using tcuModuleLoadData = Result(CUmodule *, const void *);
  tcuModuleLoadData *cuModuleLoadData_;

  using tcuModuleGetGlobal_v2 = Result(CUdeviceptr *, size_t *, CUmodule,
                                       const char *);
  tcuModuleGetGlobal_v2 *cuModuleGetGlobal_v2_;

  using tcuModuleLoadDataEx = Result(CUmodule *, const void *, unsigned int,
                                     CUjit_option *, void **);
  tcuModuleLoadDataEx *cuModuleLoadDataEx_;

  using tcuModuleUnload = Result(CUmodule);
  tcuModuleUnload *cuModuleUnload_;

  using tcuModuleGetFunction = Result(CUfunction *, CUmodule, const char *);
  tcuModuleGetFunction *cuModuleGetFunction_;

  using tcuFuncGetAttribute = Result(int *pi, CUfunction_attribute attrib,
                                     CUfunction hfunc);
  tcuFuncGetAttribute *cuFuncGetAttribute_;

  using tcuFuncSetAttribute = Result(CUfunction, CUfunction_attribute, int);
  tcuFuncSetAttribute *cuFuncSetAttribute_;

  using tcuFuncSetCacheConfig = Result(CUfunction hfunc, CUfunc_cache config);
  tcuFuncSetCacheConfig *cuFuncSetCacheConfig_;

  using tcuLaunchKernel = Result(CUfunction, unsigned int, unsigned int,
                                 unsigned int, unsigned int, unsigned int,
                                 unsigned int, unsigned int, CUstream, void **,
                                 void **);
  tcuLaunchKernel *cuLaunchKernel_;

  using tcuLaunchKernelEx = Result(const CUlaunchConfig *config,
                                   CUfunction hfunc, void **kernelParams,
                                   void **extra);
  tcuLaunchKernelEx *cuLaunchKernelEx_;

  using tcuEventCreate = Result(CUevent *, unsigned int);
  tcuEventCreate *cuEventCreate_;

  using tcuEventDestroy = Result(CUevent);
  tcuEventDestroy *cuEventDestroy_;

  using tcuEventRecord = Result(CUevent, CUstream);
  tcuEventRecord *cuEventRecord_;

  using tcuEventQuery = Result(CUevent);
  tcuEventQuery *cuEventQuery_;

  using tcuEventSynchronize = Result(CUevent);
  tcuEventSynchronize *cuEventSynchronize_;

  using tcuEventElapsedTime = Result(float *, CUevent, CUevent);
  tcuEventElapsedTime *cuEventElapsedTime_;

  using tcuMemGetInfo_v2 = Result(size_t *, size_t *);
  tcuMemGetInfo_v2 *cuMemGetInfo_v2_;

  static ErrorOr<const CUDADriver &> allocAndInitDriver();
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_CUDA_CUDADRIVER_H
