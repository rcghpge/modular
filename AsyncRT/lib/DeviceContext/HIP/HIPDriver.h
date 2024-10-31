//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the HIP driver interface.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_LIB_DEVICECONTEXT_HIP_HIPDRIVER_H
#define ASYNCRT_LIB_DEVICECONTEXT_HIP_HIPDRIVER_H

#include "Support/ErrorOr.h"

#include <stddef.h>

namespace M::AsyncRT {

using hipCtx_t = struct ihipCtx_t *;
using hipDevice_t = int;
using hipDeviceptr_t = void *;
using hipFunction_t = struct ihipModuleSymbol_t *;
using hipModule_t = struct ihipModule_t *;
using hipStream_t = struct ihipStream_t *;
using hipEvent_t = struct ihipEvent_t *;

enum class hipDeviceAttribute_t {
  COMPUTE_CAPABILITY_MAJOR = 75,
  COMPUTE_CAPABILITY_MINOR = 76,
};

enum class hipJitOption {
  JIT_MAX_REGISTERS = 0,
  JIT_THREADS_PER_BLOCK = 1,
  JIT_OPTIMIZATION_LEVEL = 7,
  JIT_GENERATE_DEBUG_INFO = 11,
  JIT_GENERATE_LINE_INFO = 13,
  JIT_CACHE_MODE = 14,
};

enum class hipFunction_attribute {
  FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
};

enum class hipFuncCache_t {
  FUNC_CACHE_PREFER_NONE = 0,
  FUNC_CACHE_PREFER_SHARED = 1,
  FUNC_CACHE_PREFER_L1 = 2,
  FUNC_CACHE_PREFER_EQUAL = 3,
};

class HIPDriver {
public:
  enum Result {
    SUCCESS = 0,
  };

  Result GetErrorName(Result error, const char **pStr) const {
    return hipDrvGetErrorName_(error, pStr);
  }

  Result GetErrorString(Result error, const char **pStr) const {
    return hipDrvGetErrorString_(error, pStr);
  }

  Result DriverGetVersion(int *version) const {
    return hipDriverGetVersion_(version);
  }

  Result DeviceGetAttribute(int *pi, hipDeviceAttribute_t attrib,
                            hipDevice_t dev) const {
    return hipDeviceGetAttribute_(pi, attrib, dev);
  }

  Result DeviceGetCount(int *count) const { return hipGetDeviceCount_(count); }

  Result DeviceGetName(char *name, int len, hipDevice_t dev) const {
    return hipDeviceGetName_(name, len, dev);
  }

  Result CtxCreate(hipCtx_t *pctx, unsigned int flags, hipDevice_t dev) const {
    return hipCtxCreate_(pctx, flags, dev);
  }
  Result CtxDestroy(hipCtx_t ctx) const { return hipCtxDestroy_(ctx); }
  Result CtxSetCurrent(hipCtx_t ctx) const { return hipCtxSetCurrent_(ctx); }

  Result StreamCreate(hipStream_t *phStream, unsigned int flags) const {
    return hipStreamCreate_(phStream, flags);
  }
  Result StreamDestroy(hipStream_t hStream) const {
    return hipStreamDestroy_(hStream);
  }
  Result StreamSynchronize(hipStream_t hStream) const {
    return hipStreamSynchronize_(hStream);
  }

  Result MemAllocHost(void **pp, size_t bytesize) const {
    return hipHostMalloc_(pp, bytesize);
  }
  Result MemFreeHost(void *p) const { return hipHostFree_(p); }

  Result MemAllocAsync(hipDeviceptr_t *pdptr, size_t bytesize,
                       hipStream_t hStream) const {
    return hipMallocAsync_(pdptr, bytesize, hStream);
  }
  Result MemFreeAsync(hipDeviceptr_t dptr, hipStream_t hStream) const {
    return hipFreeAsync_(dptr, hStream);
  }

  Result MemAlloc(hipDeviceptr_t *pdptr, size_t bytesize) const {
    return hipMalloc_(pdptr, bytesize);
  }
  Result MemFree(hipDeviceptr_t dptr) const { return hipFree_(dptr); }

  Result MemcpyHtoDAsync(hipDeviceptr_t dstDevice, const void *srcHost,
                         size_t bytesize, hipStream_t hStream) const {
    return hipMemcpyHtoDAsync_(dstDevice, srcHost, bytesize, hStream);
  }
  Result MemcpyDtoHAsync(void *dstHost, hipDeviceptr_t srcDevice,
                         size_t bytesize, hipStream_t hStream) const {
    return hipMemcpyDtoHAsync_(dstHost, srcDevice, bytesize, hStream);
  }
  Result MemcpyDtoDAsync(hipDeviceptr_t dstDevice, hipDeviceptr_t srcDevice,
                         size_t bytesize, hipStream_t hStream) const {
    return hipMemcpyDtoDAsync_(dstDevice, srcDevice, bytesize, hStream);
  }

  Result MemcpyHtoD(hipDeviceptr_t dstDevice, const void *srcHost,
                    size_t bytesize) const {
    return hipMemcpyHtoD_(dstDevice, srcHost, bytesize);
  }
  Result MemcpyDtoH(void *dstHost, hipDeviceptr_t srcDevice,
                    size_t bytesize) const {
    return hipMemcpyDtoH_(dstHost, srcDevice, bytesize);
  }
  Result MemcpyDtoD(hipDeviceptr_t dstDevice, hipDeviceptr_t srcDevice,
                    size_t bytesize) const {
    return hipMemcpyDtoD_(dstDevice, srcDevice, bytesize);
  }

  Result MemsetD8Async(hipDeviceptr_t dstDevice, unsigned char uc, size_t N,
                       hipStream_t hStream) const {
    return hipMemsetD8Async_(dstDevice, uc, N, hStream);
  }
  Result MemsetD16Async(hipDeviceptr_t dstDevice, unsigned short us, size_t N,
                        hipStream_t hStream) const {
    return hipMemsetD16Async_(dstDevice, us, N, hStream);
  }
  Result MemsetD32Async(hipDeviceptr_t dstDevice, unsigned int ui, size_t N,
                        hipStream_t hStream) const {
    return hipMemsetD32Async_(dstDevice, ui, N, hStream);
  }

  Result MemsetD8(hipDeviceptr_t dstDevice, unsigned char uc, size_t N) const {
    return hipMemsetD8_(dstDevice, uc, N);
  }
  Result MemsetD16(hipDeviceptr_t dstDevice, unsigned short us,
                   size_t N) const {
    return hipMemsetD16_(dstDevice, us, N);
  }
  Result MemsetD32(hipDeviceptr_t dstDevice, unsigned int ui, size_t N) const {
    return hipMemsetD32_(dstDevice, ui, N);
  }

  Result ModuleLoadData(hipModule_t *module, const void *image) const {
    return hipModuleLoadData_(module, image);
  }
  Result ModuleLoadDataEx(hipModule_t *module, const void *image,
                          unsigned int numOptions, hipJitOption *options,
                          void **optionValues) const {
    return hipModuleLoadDataEx_(module, image, numOptions, options,
                                optionValues);
  }
  Result ModuleGetGlobal(hipDeviceptr_t *dptr, size_t *sizeptr,
                         hipModule_t hmod, const char *name) const {
    return hipModuleGetGlobal_(dptr, sizeptr, hmod, name);
  }
  Result ModuleUnload(hipModule_t hmod) const { return hipModuleUnload_(hmod); }
  Result ModuleGetFunction(hipFunction_t *hfunc, hipModule_t hmod,
                           const char *name) const {
    return hipModuleGetFunction_(hfunc, hmod, name);
  }
  Result FuncGetAttribute(int *pi, hipFunction_attribute attrib,
                          hipFunction_t hfunc) const {
    return hipFuncGetAttribute_(pi, attrib, hfunc);
  }

  Result ModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX,
                            unsigned int gridDimY, unsigned int gridDimZ,
                            unsigned int blockDimX, unsigned int blockDimY,
                            unsigned int blockDimZ, unsigned int sharedMemBytes,
                            hipStream_t hStream, void **kernelParams,
                            void **extra) const {
    return hipModuleLaunchKernel_(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                                  blockDimY, blockDimZ, sharedMemBytes, hStream,
                                  kernelParams, extra);
  }

  Result EventCreate(hipEvent_t *phEvent, unsigned int flags) const {
    return hipEventCreateWithFlags_(phEvent, flags);
  }
  Result EventDestroy(hipEvent_t hEvent) const {
    return hipEventDestroy_(hEvent);
  }
  Result EventRecord(hipEvent_t hEvent, hipStream_t hStream) const {
    return hipEventRecord_(hEvent, hStream);
  }
  Result EventQuery(hipEvent_t hEvent) const { return hipEventQuery_(hEvent); }
  Result EventSynchronize(hipEvent_t hEvent) const {
    return hipEventSynchronize_(hEvent);
  }
  Result EventElapsedTime(float *pMilliseconds, hipEvent_t hStart,
                          hipEvent_t hEnd) const {
    return hipEventElapsedTime_(pMilliseconds, hStart, hEnd);
  }

  Result GetMemInfo(size_t *free, size_t *total) const {
    return hipMemGetInfo_(free, total);
  }

  static ErrorOr<const HIPDriver &> get();

private:
  // Shared library name for the HIP driver.
  static const char lib_name[];

  // Once created, the HIPDriver object cannot be deleted.
  HIPDriver() = default;
  ~HIPDriver() = default;

  // Driver is only initialized from internal code.
  Result Init(unsigned int flags) const { return hipInit_(flags); }
  using thipInit = Result(unsigned int);
  thipInit *hipInit_;

  using thipDrvGetErrorName = Result(Result, const char **);
  thipDrvGetErrorName *hipDrvGetErrorName_;

  using thipDrvGetErrorString = Result(Result, const char **);
  thipDrvGetErrorString *hipDrvGetErrorString_;

  using thipDriverGetVersion = Result(int *);
  thipDriverGetVersion *hipDriverGetVersion_;

  using thipDeviceGetAttribute = Result(int *, hipDeviceAttribute_t,
                                        hipDevice_t);
  thipDeviceGetAttribute *hipDeviceGetAttribute_;

  using thipGetDeviceCount = Result(int *);
  thipGetDeviceCount *hipGetDeviceCount_;

  using thipDeviceGetName = Result(char *, int, hipDevice_t);
  thipDeviceGetName *hipDeviceGetName_;

  using thipCtxCreate = Result(hipCtx_t *, unsigned int, hipDevice_t);
  thipCtxCreate *hipCtxCreate_;

  using thipCtxDestroy = Result(hipCtx_t);
  thipCtxDestroy *hipCtxDestroy_;

  using thipCtxSetCurrent = Result(hipCtx_t);
  thipCtxSetCurrent *hipCtxSetCurrent_;

  using thipStreamCreate = Result(hipStream_t *, unsigned int);
  thipStreamCreate *hipStreamCreate_;

  using thipStreamDestroy = Result(hipStream_t);
  thipStreamDestroy *hipStreamDestroy_;

  using thipStreamSynchronize = Result(hipStream_t);
  thipStreamSynchronize *hipStreamSynchronize_;

  using thipHostMalloc = Result(void **, size_t);
  thipHostMalloc *hipHostMalloc_;

  using thipHostFree = Result(void *);
  thipHostFree *hipHostFree_;

  using thipMallocAsync = Result(hipDeviceptr_t *, size_t, hipStream_t);
  thipMallocAsync *hipMallocAsync_;

  using thipFreeAsync = Result(hipDeviceptr_t, hipStream_t);
  thipFreeAsync *hipFreeAsync_;

  using thipMalloc = Result(hipDeviceptr_t *, size_t);
  thipMalloc *hipMalloc_;

  using thipFree = Result(hipDeviceptr_t);
  thipFree *hipFree_;

  using thipMemcpyHtoDAsync = Result(hipDeviceptr_t, const void *, size_t,
                                     hipStream_t);
  thipMemcpyHtoDAsync *hipMemcpyHtoDAsync_;

  using thipMemcpyDtoHAsync = Result(void *, hipDeviceptr_t, size_t,
                                     hipStream_t);
  thipMemcpyDtoHAsync *hipMemcpyDtoHAsync_;

  using thipMemcpyDtoDAsync = Result(hipDeviceptr_t, hipDeviceptr_t, size_t,
                                     hipStream_t);
  thipMemcpyDtoDAsync *hipMemcpyDtoDAsync_;

  using thipMemcpyHtoD = Result(hipDeviceptr_t, const void *, size_t);
  thipMemcpyHtoD *hipMemcpyHtoD_;

  using thipMemcpyDtoH = Result(void *, hipDeviceptr_t, size_t);
  thipMemcpyDtoH *hipMemcpyDtoH_;

  using thipMemcpyDtoD = Result(hipDeviceptr_t, hipDeviceptr_t, size_t);
  thipMemcpyDtoD *hipMemcpyDtoD_;

  using thipMemsetD8Async = Result(hipDeviceptr_t, unsigned char, size_t,
                                   hipStream_t);
  thipMemsetD8Async *hipMemsetD8Async_;

  using thipMemsetD16Async = Result(hipDeviceptr_t, unsigned short, size_t,
                                    hipStream_t);
  thipMemsetD16Async *hipMemsetD16Async_;

  using thipMemsetD32Async = Result(hipDeviceptr_t, unsigned int, size_t,
                                    hipStream_t);
  thipMemsetD32Async *hipMemsetD32Async_;

  using thipMemsetD8 = Result(hipDeviceptr_t, unsigned char, size_t);
  thipMemsetD8 *hipMemsetD8_;

  using thipMemsetD16 = Result(hipDeviceptr_t, unsigned short, size_t);
  thipMemsetD16 *hipMemsetD16_;

  using thipMemsetD32 = Result(hipDeviceptr_t, unsigned int, size_t);
  thipMemsetD32 *hipMemsetD32_;

  using thipModuleLoadData = Result(hipModule_t *, const void *);
  thipModuleLoadData *hipModuleLoadData_;

  using thipModuleGetGlobal = Result(hipDeviceptr_t *, size_t *, hipModule_t,
                                     const char *);
  thipModuleGetGlobal *hipModuleGetGlobal_;

  using thipModuleLoadDataEx = Result(hipModule_t *, const void *, unsigned int,
                                      hipJitOption *, void **);
  thipModuleLoadDataEx *hipModuleLoadDataEx_;

  using thipModuleUnload = Result(hipModule_t);
  thipModuleUnload *hipModuleUnload_;

  using thipModuleGetFunction = Result(hipFunction_t *, hipModule_t,
                                       const char *);
  thipModuleGetFunction *hipModuleGetFunction_;

  using thipFuncGetAttribute = Result(int *pi, hipFunction_attribute attrib,
                                      hipFunction_t hfunc);
  thipFuncGetAttribute *hipFuncGetAttribute_;

  using thipModuleLaunchKernel = Result(hipFunction_t, unsigned int,
                                        unsigned int, unsigned int,
                                        unsigned int, unsigned int,
                                        unsigned int, unsigned int, hipStream_t,
                                        void **, void **);
  thipModuleLaunchKernel *hipModuleLaunchKernel_;

  using thipEventCreateWithFlags = Result(hipEvent_t *, unsigned int);
  thipEventCreateWithFlags *hipEventCreateWithFlags_;

  using thipEventDestroy = Result(hipEvent_t);
  thipEventDestroy *hipEventDestroy_;

  using thipEventRecord = Result(hipEvent_t, hipStream_t);
  thipEventRecord *hipEventRecord_;

  using thipEventQuery = Result(hipEvent_t);
  thipEventQuery *hipEventQuery_;

  using thipEventSynchronize = Result(hipEvent_t);
  thipEventSynchronize *hipEventSynchronize_;

  using thipEventElapsedTime = Result(float *, hipEvent_t, hipEvent_t);
  thipEventElapsedTime *hipEventElapsedTime_;

  using thipMemGetInfo = Result(size_t *, size_t *);
  thipMemGetInfo *hipMemGetInfo_;

  static ErrorOr<const HIPDriver &> allocAndInitDriver();
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_HIP_HIPDRIVER_H
