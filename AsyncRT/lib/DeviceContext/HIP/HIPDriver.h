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

// COPIED FROM hip_runtime_api.h

// clang-format off
enum class hipDeviceAttribute_t {
    hipDeviceAttributeCudaCompatibleBegin = 0,

    hipDeviceAttributeEccEnabled = hipDeviceAttributeCudaCompatibleBegin, ///< Whether ECC support is enabled.
    hipDeviceAttributeAccessPolicyMaxWindowSize,        ///< Cuda only. The maximum size of the window policy in bytes.
    hipDeviceAttributeAsyncEngineCount,                 ///< Asynchronous engines number.
    hipDeviceAttributeCanMapHostMemory,                 ///< Whether host memory can be mapped into device address space
    hipDeviceAttributeCanUseHostPointerForRegisteredMem,///< Device can access host registered memory
                                                        ///< at the same virtual address as the CPU
    hipDeviceAttributeClockRate,                        ///< Peak clock frequency in kilohertz.
    hipDeviceAttributeComputeMode,                      ///< Compute mode that device is currently in.
    hipDeviceAttributeComputePreemptionSupported,       ///< Device supports Compute Preemption.
    hipDeviceAttributeConcurrentKernels,                ///< Device can possibly execute multiple kernels concurrently.
    hipDeviceAttributeConcurrentManagedAccess,          ///< Device can coherently access managed memory concurrently with the CPU
    hipDeviceAttributeCooperativeLaunch,                ///< Support cooperative launch
    hipDeviceAttributeCooperativeMultiDeviceLaunch,     ///< Support cooperative launch on multiple devices
    hipDeviceAttributeDeviceOverlap,                    ///< Device can concurrently copy memory and execute a kernel.
                                                        ///< Deprecated. Use instead asyncEngineCount.
    hipDeviceAttributeDirectManagedMemAccessFromHost,   ///< Host can directly access managed memory on
                                                        ///< the device without migration
    hipDeviceAttributeGlobalL1CacheSupported,           ///< Device supports caching globals in L1
    hipDeviceAttributeHostNativeAtomicSupported,        ///< Link between the device and the host supports native atomic operations
    hipDeviceAttributeIntegrated,                       ///< Device is integrated GPU
    hipDeviceAttributeIsMultiGpuBoard,                  ///< Multiple GPU devices.
    hipDeviceAttributeKernelExecTimeout,                ///< Run time limit for kernels executed on the device
    hipDeviceAttributeL2CacheSize,                      ///< Size of L2 cache in bytes. 0 if the device doesn't have L2 cache.
    hipDeviceAttributeLocalL1CacheSupported,            ///< caching locals in L1 is supported
    hipDeviceAttributeLuid,                             ///< 8-byte locally unique identifier in 8 bytes. Undefined on TCC and non-Windows platforms
    hipDeviceAttributeLuidDeviceNodeMask,               ///< Luid device node mask. Undefined on TCC and non-Windows platforms
    hipDeviceAttributeComputeCapabilityMajor,           ///< Major compute capability version number.
    hipDeviceAttributeManagedMemory,                    ///< Device supports allocating managed memory on this system
    hipDeviceAttributeMaxBlocksPerMultiProcessor,       ///< Max block size per multiprocessor
    hipDeviceAttributeMaxBlockDimX,                     ///< Max block size in width.
    hipDeviceAttributeMaxBlockDimY,                     ///< Max block size in height.
    hipDeviceAttributeMaxBlockDimZ,                     ///< Max block size in depth.
    hipDeviceAttributeMaxGridDimX,                      ///< Max grid size  in width.
    hipDeviceAttributeMaxGridDimY,                      ///< Max grid size  in height.
    hipDeviceAttributeMaxGridDimZ,                      ///< Max grid size  in depth.
    hipDeviceAttributeMaxSurface1D,                     ///< Maximum size of 1D surface.
    hipDeviceAttributeMaxSurface1DLayered,              ///< Cuda only. Maximum dimensions of 1D layered surface.
    hipDeviceAttributeMaxSurface2D,                     ///< Maximum dimension (width, height) of 2D surface.
    hipDeviceAttributeMaxSurface2DLayered,              ///< Cuda only. Maximum dimensions of 2D layered surface.
    hipDeviceAttributeMaxSurface3D,                     ///< Maximum dimension (width, height, depth) of 3D surface.
    hipDeviceAttributeMaxSurfaceCubemap,                ///< Cuda only. Maximum dimensions of Cubemap surface.
    hipDeviceAttributeMaxSurfaceCubemapLayered,         ///< Cuda only. Maximum dimension of Cubemap layered surface.
    hipDeviceAttributeMaxTexture1DWidth,                ///< Maximum size of 1D texture.
    hipDeviceAttributeMaxTexture1DLayered,              ///< Maximum dimensions of 1D layered texture.
    hipDeviceAttributeMaxTexture1DLinear,               ///< Maximum number of elements allocatable in a 1D linear texture.
                                                        ///< Use cudaDeviceGetTexture1DLinearMaxWidth() instead on Cuda.
    hipDeviceAttributeMaxTexture1DMipmap,               ///< Maximum size of 1D mipmapped texture.
    hipDeviceAttributeMaxTexture2DWidth,                ///< Maximum dimension width of 2D texture.
    hipDeviceAttributeMaxTexture2DHeight,               ///< Maximum dimension hight of 2D texture.
    hipDeviceAttributeMaxTexture2DGather,               ///< Maximum dimensions of 2D texture if gather operations  performed.
    hipDeviceAttributeMaxTexture2DLayered,              ///< Maximum dimensions of 2D layered texture.
    hipDeviceAttributeMaxTexture2DLinear,               ///< Maximum dimensions (width, height, pitch) of 2D textures bound to pitched memory.
    hipDeviceAttributeMaxTexture2DMipmap,               ///< Maximum dimensions of 2D mipmapped texture.
    hipDeviceAttributeMaxTexture3DWidth,                ///< Maximum dimension width of 3D texture.
    hipDeviceAttributeMaxTexture3DHeight,               ///< Maximum dimension height of 3D texture.
    hipDeviceAttributeMaxTexture3DDepth,                ///< Maximum dimension depth of 3D texture.
    hipDeviceAttributeMaxTexture3DAlt,                  ///< Maximum dimensions of alternate 3D texture.
    hipDeviceAttributeMaxTextureCubemap,                ///< Maximum dimensions of Cubemap texture
    hipDeviceAttributeMaxTextureCubemapLayered,         ///< Maximum dimensions of Cubemap layered texture.
    hipDeviceAttributeMaxThreadsDim,                    ///< Maximum dimension of a block
    hipDeviceAttributeMaxThreadsPerBlock,               ///< Maximum number of threads per block.
    hipDeviceAttributeMaxThreadsPerMultiProcessor,      ///< Maximum resident threads per multiprocessor.
    hipDeviceAttributeMaxPitch,                         ///< Maximum pitch in bytes allowed by memory copies
    hipDeviceAttributeMemoryBusWidth,                   ///< Global memory bus width in bits.
    hipDeviceAttributeMemoryClockRate,                  ///< Peak memory clock frequency in kilohertz.
    hipDeviceAttributeComputeCapabilityMinor,           ///< Minor compute capability version number.
    hipDeviceAttributeMultiGpuBoardGroupID,             ///< Unique ID of device group on the same multi-GPU board
    hipDeviceAttributeMultiprocessorCount,              ///< Number of multiprocessors on the device.
    hipDeviceAttributeUnused1,                          ///< Previously hipDeviceAttributeName
    hipDeviceAttributePageableMemoryAccess,             ///< Device supports coherently accessing pageable memory
                                                        ///< without calling hipHostRegister on it
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables, ///< Device accesses pageable memory via the host's page tables
    hipDeviceAttributePciBusId,                         ///< PCI Bus ID.
    hipDeviceAttributePciDeviceId,                      ///< PCI Device ID.
    hipDeviceAttributePciDomainID,                      ///< PCI Domain ID.
    hipDeviceAttributePersistingL2CacheMaxSize,         ///< Maximum l2 persisting lines capacity in bytes
    hipDeviceAttributeMaxRegistersPerBlock,             ///< 32-bit registers available to a thread block. This number is shared
                                                        ///< by all thread blocks simultaneously resident on a multiprocessor.
    hipDeviceAttributeMaxRegistersPerMultiprocessor,    ///< 32-bit registers available per block.
    hipDeviceAttributeReservedSharedMemPerBlock,        ///< Shared memory reserved by CUDA driver per block.
    hipDeviceAttributeMaxSharedMemoryPerBlock,          ///< Maximum shared memory available per block in bytes.
    hipDeviceAttributeSharedMemPerBlockOptin,           ///< Maximum shared memory per block usable by special opt in.
    hipDeviceAttributeSharedMemPerMultiprocessor,       ///< Shared memory available per multiprocessor.
    hipDeviceAttributeSingleToDoublePrecisionPerfRatio, ///< Cuda only. Performance ratio of single precision to double precision.
    hipDeviceAttributeStreamPrioritiesSupported,        ///< Whether to support stream priorities.
    hipDeviceAttributeSurfaceAlignment,                 ///< Alignment requirement for surfaces
    hipDeviceAttributeTccDriver,                        ///< Cuda only. Whether device is a Tesla device using TCC driver
    hipDeviceAttributeTextureAlignment,                 ///< Alignment requirement for textures
    hipDeviceAttributeTexturePitchAlignment,            ///< Pitch alignment requirement for 2D texture references bound to pitched memory;
    hipDeviceAttributeTotalConstantMemory,              ///< Constant memory size in bytes.
    hipDeviceAttributeTotalGlobalMem,                   ///< Global memory available on devicice.
    hipDeviceAttributeUnifiedAddressing,                ///< Cuda only. An unified address space shared with the host.
    hipDeviceAttributeUnused2,                          ///< Previously hipDeviceAttributeUuid
    hipDeviceAttributeWarpSize,                         ///< Warp size in threads.
    hipDeviceAttributeMemoryPoolsSupported,             ///< Device supports HIP Stream Ordered Memory Allocator
    hipDeviceAttributeVirtualMemoryManagementSupported, ///< Device supports HIP virtual memory management
    hipDeviceAttributeHostRegisterSupported,            ///< Can device support host memory registration via hipHostRegister
    hipDeviceAttributeMemoryPoolSupportedHandleTypes,   ///< Supported handle mask for HIP Stream Ordered Memory Allocator

    hipDeviceAttributeCudaCompatibleEnd = 9999,
    hipDeviceAttributeAmdSpecificBegin = 10000,

    hipDeviceAttributeClockInstructionRate = hipDeviceAttributeAmdSpecificBegin,  ///< Frequency in khz of the timer used by the device-side "clock*"
    hipDeviceAttributeUnused3,                                  ///< Previously hipDeviceAttributeArch
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,         ///< Maximum Shared Memory PerMultiprocessor.
    hipDeviceAttributeUnused4,                                  ///< Previously hipDeviceAttributeGcnArch
    hipDeviceAttributeUnused5,                                  ///< Previously hipDeviceAttributeGcnArchName
    hipDeviceAttributeHdpMemFlushCntl,                          ///< Address of the HDP_MEM_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeHdpRegFlushCntl,                          ///< Address of the HDP_REG_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,      ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched functions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,   ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched grid dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,  ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched block dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem, ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched shared memories
    hipDeviceAttributeIsLargeBar,                               ///< Whether it is LargeBar
    hipDeviceAttributeAsicRevision,                             ///< Revision of the GPU in this device
    hipDeviceAttributeCanUseStreamWaitValue,                    ///< '1' if Device supports hipStreamWaitValue32() and
                                                                ///< hipStreamWaitValue64(), '0' otherwise.
    hipDeviceAttributeImageSupport,                             ///< '1' if Device supports image, '0' otherwise.
    hipDeviceAttributePhysicalMultiProcessorCount,              ///< All available physical compute
                                                                ///< units for the device
    hipDeviceAttributeFineGrainSupport,                         ///< '1' if Device supports fine grain, '0' otherwise
    hipDeviceAttributeWallClockRate,                            ///< Constant frequency of wall clock in kilohertz.

    hipDeviceAttributeAmdSpecificEnd = 19999,
    hipDeviceAttributeVendorSpecificBegin = 20000,
    // Extended attributes for vendors
};
// clang-format on

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

  Result SetDevice(hipDevice_t dev) const { return hipSetDevice_(dev); }
  Result GetDevice(int *deviceId) const { return hipGetDevice_(deviceId); }
  Result DeviceGetName(char *name, int len, hipDevice_t dev) const {
    return hipDeviceGetName_(name, len, dev);
  }

  Result DeviceCanAccessPeer(int *canAccessPeer, int deviceId,
                             int peerDeviceId) const {
    return hipDeviceCanAccessPeer_(canAccessPeer, deviceId, peerDeviceId);
  }
  Result DeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) const {
    return hipDeviceEnablePeerAccess_(peerDeviceId, flags);
  }

  Result StreamCreate(hipStream_t *phStream, unsigned int flags) const {
    return hipStreamCreate_(phStream, flags);
  }
  Result StreamDestroy(hipStream_t hStream) const {
    return hipStreamDestroy_(hStream);
  }
  Result StreamSynchronize(hipStream_t hStream) const {
    return hipStreamSynchronize_(hStream);
  }
  Result StreamWaitEvent(hipStream_t hStream, hipEvent_t hEvent,
                         unsigned int flags) const {
    return hipStreamWaitEvent_(hStream, hEvent, flags);
  }

  Result MemAllocHost(void **pp, size_t bytesize) const {
    return hipHostMalloc_(pp, bytesize, 0);
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
  Result Memset2DAsync(void *dst, size_t pitch, int value, size_t width,
                       size_t height, hipStream_t stream) const {
    return hipMemset2DAsync_(dst, pitch, value, width, height, stream);
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
  Result Memset2D(void *dst, size_t pitch, int value, size_t width,
                  size_t height) const {
    return hipMemset2D_(dst, pitch, value, width, height);
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

  using thipSetDevice = Result(hipDevice_t);
  thipSetDevice *hipSetDevice_;

  using thipGetDevice = Result(int *);
  thipGetDevice *hipGetDevice_;

  using thipDeviceGetName = Result(char *, int, hipDevice_t);
  thipDeviceGetName *hipDeviceGetName_;

  using thipDeviceCanAccessPeer = Result(int *canAccessPeer, int deviceId,
                                         int peerDeviceId);
  thipDeviceCanAccessPeer *hipDeviceCanAccessPeer_;

  using thipDeviceEnablePeerAccess = Result(int peerDeviceId,
                                            unsigned int flags);
  thipDeviceEnablePeerAccess *hipDeviceEnablePeerAccess_;

  using thipStreamCreate = Result(hipStream_t *, unsigned int);
  thipStreamCreate *hipStreamCreate_;

  using thipStreamDestroy = Result(hipStream_t);
  thipStreamDestroy *hipStreamDestroy_;

  using thipStreamSynchronize = Result(hipStream_t);
  thipStreamSynchronize *hipStreamSynchronize_;
  using thipStreamWaitEvent = Result(hipStream_t, hipEvent_t, unsigned int);
  thipStreamWaitEvent *hipStreamWaitEvent_;

  using thipHostMalloc = Result(void **, size_t, unsigned int flags);
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

  using thipMemset2DAsync = Result(void *dst, size_t pitch, int value,
                                   size_t width, size_t height,
                                   hipStream_t stream);
  thipMemset2DAsync *hipMemset2DAsync_;

  using thipMemsetD8 = Result(hipDeviceptr_t, unsigned char, size_t);
  thipMemsetD8 *hipMemsetD8_;

  using thipMemsetD16 = Result(hipDeviceptr_t, unsigned short, size_t);
  thipMemsetD16 *hipMemsetD16_;

  using thipMemsetD32 = Result(hipDeviceptr_t, unsigned int, size_t);
  thipMemsetD32 *hipMemsetD32_;

  using thipMemset2D = Result(void *dst, size_t pitch, int value, size_t width,
                              size_t height);
  thipMemset2D *hipMemset2D_;

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
