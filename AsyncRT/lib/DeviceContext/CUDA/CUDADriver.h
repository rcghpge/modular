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
#include "llvm/Support/raw_ostream.h"

#ifdef MODULAR_LINUX
#include "nvPTXCompiler.h"
#endif // MODULAR_LINUX

#include <sstream>
#include <stddef.h>
#include <thread>

namespace M::AsyncRT {

using CUcontext = struct CUctx_st *;
using CUdevice = int;
using CUdeviceptr = void *;
using CUfunction = struct CUfunc_st *;
using CUmemGenericAllocationHandle = unsigned long long;
using CUmemoryPool = struct CUmemPoolHandle_st *;
using CUmodule = struct CUmod_st *;
using CUstream = struct CUstream_st *;
using CUevent = struct CUevent_st *;
using CUhostFn = void(void *);
using CUlinkState = void *;

using nvPTXCompilerHandle = struct nvPTXCompiler *;

enum class CUdevice_attribute {
  TEXTURE_ALIGNMENT = 14,
  UNIFIED_ADDRESSING = 41,
  COMPUTE_CAPABILITY_MAJOR = 75,
  COMPUTE_CAPABILITY_MINOR = 76,
  MULTICAST_SUPPORTED = 132,
};

enum class CUdevice_P2PAttribute {
  CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 0x01,
  CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 0x02,
  CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 0x03,
  CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = 0x04,
  CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = 0x04
};

enum class CUevent_flags {
  CU_EVENT_DEFAULT = 0x0,
  CU_EVENT_BLOCKING_SYNC = 0x1,
  CU_EVENT_DISABLE_TIMING = 0x2,
  CU_EVENT_INTERPROCESS = 0x4,
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
  FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14
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

using CUmemorytype = enum CUmemorytype_enum {
  CU_MEMORYTYPE_HOST = 0x01,
  CU_MEMORYTYPE_DEVICE = 0x02,
  CU_MEMORYTYPE_ARRAY = 0x03,
  CU_MEMORYTYPE_UNIFIED = 0x04
};

using CUpointer_attribute = enum CUpointer_attribute_enum {
  CU_POINTER_ATTRIBUTE_CONTEXT = 1,
  CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
  CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
  CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
  CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
  CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
  CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,
  CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
  CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
  CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10,
  CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
  CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
  CU_POINTER_ATTRIBUTE_MAPPED = 13,
  CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,
  CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15,
  CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,
  CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17,
  CU_POINTER_ATTRIBUTE_MAPPING_SIZE = 18,
  CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR = 19,
  CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID = 20,
  CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE = 21
};

enum class CUtensorMapDataType {
  CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,
  CU_TENSOR_MAP_DATA_TYPE_UINT16,
  CU_TENSOR_MAP_DATA_TYPE_UINT32,
  CU_TENSOR_MAP_DATA_TYPE_INT32,
  CU_TENSOR_MAP_DATA_TYPE_UINT64,
  CU_TENSOR_MAP_DATA_TYPE_INT64,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ,
  CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
  CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ,
};

enum class CUtensorMapFloatOOBfill {
  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0,
  CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA,
};

enum class CUtensorMapInterleave {
  CU_TENSOR_MAP_INTERLEAVE_NONE = 0,
  CU_TENSOR_MAP_INTERLEAVE_16B,
  CU_TENSOR_MAP_INTERLEAVE_32B,
};

enum class CUtensorMapL2promotion {
  CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
  CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
  CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
  CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
};

enum class CUtensorMapSwizzle {
  CU_TENSOR_MAP_SWIZZLE_NONE = 0,
  CU_TENSOR_MAP_SWIZZLE_32B,
  CU_TENSOR_MAP_SWIZZLE_64B,
  CU_TENSOR_MAP_SWIZZLE_128B,
};

enum class CUjitInputType {
  CU_JIT_INPUT_CUBIN = 0,
  CU_JIT_INPUT_PTX = 1,
  CU_JIT_INPUT_FATBINARY = 2,
  CU_JIT_INPUT_OBJECT = 3,
  CU_JIT_INPUT_LIBRARY = 4,
  CU_JIT_INPUT_NVVM = 5,
  CU_JIT_NUM_INPUT_TYPES = 6
};

enum class CUmemAccess_flags {
  CU_MEM_ACCESS_FLAGS_PROT_NONE = 0x0,
  CU_MEM_ACCESS_FLAGS_PROT_READ = 0x1,
  CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3,
  CU_MEM_ACCESS_FLAGS_PROT_MAX = 0x7FFFFFFF
};

enum class CUmemLocationType {
  CU_MEM_LOCATION_TYPE_INVALID = 0x0,
  CU_MEM_LOCATION_TYPE_DEVICE = 0x1,
  CU_MEM_LOCATION_TYPE_HOST = 0x2,
  CU_MEM_LOCATION_TYPE_HOST_NUMA = 0x3,
  CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT = 0x4,
  CU_MEM_LOCATION_TYPE_MAX = 0x7FFFFFFF
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

using CUmemLocation = struct CUmemLocation_st {
  CUmemLocationType type;
  int id;
};

using CUmemAccessDesc = struct CUmemAccessDesc_st {
  CUmemLocation location;
  CUmemAccess_flags flags;
};

enum class CUmemAllocationType {
  INVALID = 0x0,
  PINNED = 0x1,
  MAX = 0x7FFFFFFF,
};

enum class CUmemAllocationHandleType {
  NONE = 0x0,
  POSIX_FILE_DESCRIPTOR = 0x1,
  WIN32 = 0x2,
  WIN32_KMT = 0x4,
  FABRIC = 0x8,
  MAX = 0x7FFFFFFF,
};

using CUmemAllocationProp = struct CUmemAllocationProp_st {
  CUmemAllocationType type;
  CUmemAllocationHandleType requestedHandleTypes;
  CUmemLocation location;
  void *win32HandleMetaData;
  struct {
    unsigned char compressionType;
    unsigned char gpuDirectRDMACapable;
    unsigned short usage;
    unsigned char reserved[4];
  } allocFlags;
};

enum class CUmemAllocationGranularity_flags {
  GRANULARITY_MINIMUM = 0x0,
  GRANULARITY_RECOMMENDED = 0x1,
};

enum class CUmulticastGranularity_flags {
  GRANULARITY_MINIMUM = 0x0,
  GRANULARITY_RECOMMENDED = 0x1,
};

using CUmulticastObjectProp = struct CUmulticastObjectProp_st {
  unsigned int numDevices;
  size_t size;
  unsigned long long handleTypes;
  unsigned long long flags;
};

#define CUDA_DRIVER_CALL_LOG()                                                 \
  if (do_call_log) {                                                           \
    /* Make sure we emit a single entry per log call. */                       \
    std::ostringstream buffer;                                                 \
    buffer << "THR-" << std::this_thread::get_id()                             \
           << ": CUDADriver::" << __FUNCTION__ << "\n";                        \
    llvm::errs() << buffer.str();                                              \
  }

class CUDADriver {

public:
  enum Result : unsigned int {
    SUCCESS = 0,
    NOT_READY = 600,
  };

  Result GetErrorName(Result error, const char **pStr) const {
    CUDA_DRIVER_CALL_LOG();
    return cuGetErrorName_(error, pStr);
  }

  Result GetErrorString(Result error, const char **pStr) const {
    CUDA_DRIVER_CALL_LOG();
    return cuGetErrorString_(error, pStr);
  }

  Result DriverGetVersion(int *version) const {
    CUDA_DRIVER_CALL_LOG();
    return cuDriverGetVersion_(version);
  }

  Result DeviceGet(CUdevice *device, int ordinal) const {
    CUDA_DRIVER_CALL_LOG();
    return cuDeviceGet_(device, ordinal);
  }

  Result DeviceGetAttribute(int *pi, CUdevice_attribute attrib,
                            CUdevice dev) const {
    CUDA_DRIVER_CALL_LOG();
    return cuDeviceGetAttribute_(pi, attrib, dev);
  }

  Result DeviceGetCount(int *count) const {
    CUDA_DRIVER_CALL_LOG();
    return cuDeviceGetCount_(count);
  }

  Result DeviceGetName(char *name, int len, CUdevice dev) const {
    CUDA_DRIVER_CALL_LOG();
    return cuDeviceGetName_(name, len, dev);
  }
  Result DeviceCanAccessPeer(int *canAccessPeer, CUdevice dev,
                             CUdevice peerDev) const {
    CUDA_DRIVER_CALL_LOG();
    return cuDeviceCanAccessPeer_(canAccessPeer, dev, peerDev);
  }

  Result DeviceGetP2PAttribute(int *value, CUdevice_P2PAttribute attrib,
                               CUdevice srcDevice, CUdevice dstDevice) const {
    CUDA_DRIVER_CALL_LOG();
    return cuDeviceGetP2PAttribute_(value, attrib, srcDevice, dstDevice);
  }

  Result DevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) const {
    CUDA_DRIVER_CALL_LOG();
    return cuDevicePrimaryCtxRetain_(pctx, dev);
  }
  Result DevicePrimaryCtxRelease(CUdevice dev) const {
    CUDA_DRIVER_CALL_LOG();
    return cuDevicePrimaryCtxRelease_v2_(dev);
  }

  Result CtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) const {
    CUDA_DRIVER_CALL_LOG();
    return cuCtxCreate_v2_(pctx, flags, dev);
  }
  Result CtxDestroy(CUcontext ctx) const {
    CUDA_DRIVER_CALL_LOG();
    return cuCtxDestroy_(ctx);
  }
  Result CtxSetCurrent(CUcontext ctx) const {
    CUDA_DRIVER_CALL_LOG();
    return cuCtxSetCurrent_(ctx);
  }
  Result CtxGetCurrent(CUcontext *pctx) const {
    CUDA_DRIVER_CALL_LOG();
    return cuCtxGetCurrent_(pctx);
  }
  Result CtxPushCurrent(CUcontext ctx) const {
    CUDA_DRIVER_CALL_LOG();
    return cuCtxPushCurrent_(ctx);
  }
  Result CtxPopCurrent(CUcontext *ctx) const {
    CUDA_DRIVER_CALL_LOG();
    return cuCtxPopCurrent_(ctx);
  }
  Result CtxEnablePeerAccess(CUcontext peerContext, unsigned int flags) const {
    CUDA_DRIVER_CALL_LOG();
    return cuCtxEnablePeerAccess_(peerContext, flags);
  }

  Result StreamCreate(CUstream *phStream, unsigned int flags) const {
    CUDA_DRIVER_CALL_LOG();
    return cuStreamCreate_(phStream, flags);
  }
  Result StreamCreateWithPriority(CUstream *phStream, unsigned int flags,
                                  int priority) const {
    CUDA_DRIVER_CALL_LOG();
    return cuStreamCreateWithPriority_(phStream, flags, priority);
  }
  Result StreamDestroy(CUstream hStream) const {
    CUDA_DRIVER_CALL_LOG();
    return cuStreamDestroy_(hStream);
  }
  Result StreamSynchronize(CUstream hStream) const {
    CUDA_DRIVER_CALL_LOG();
    return cuStreamSynchronize_(hStream);
  }
  Result StreamWaitEvent(CUstream hStream, CUevent hEvent,
                         unsigned int flags) const {
    CUDA_DRIVER_CALL_LOG();
    return cuStreamWaitEvent_(hStream, hEvent, flags);
  }
  Result CtxGetStreamPriorityRange(int *leastPriority,
                                   int *greatestPriority) const {
    CUDA_DRIVER_CALL_LOG();
    return cuCtxGetStreamPriorityRange_(leastPriority, greatestPriority);
  }

  Result MemAllocHost(void **pp, size_t bytesize) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemAllocHost_v2_(pp, bytesize);
  }
  Result MemFreeHost(void *p) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemFreeHost_(p);
  }

  Result MemAllocAsync(CUdeviceptr *pdptr, size_t bytesize,
                       CUstream hStream) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemAllocAsync_(pdptr, bytesize, hStream);
  }
  Result MemFreeAsync(CUdeviceptr dptr, CUstream hStream) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemFreeAsync_(dptr, hStream);
  }

  Result MemAlloc(CUdeviceptr *pdptr, size_t bytesize) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemAlloc_v2_(pdptr, bytesize);
  }
  Result MemFree(CUdeviceptr dptr) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemFree_v2_(dptr);
  }

  Result MemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
                         size_t bytesize, CUstream hStream) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemcpyHtoDAsync_v2_(dstDevice, srcHost, bytesize, hStream);
  }
  Result MemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t bytesize,
                         CUstream hStream) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemcpyDtoHAsync_v2_(dstHost, srcDevice, bytesize, hStream);
  }
  Result MemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                         size_t bytesize, CUstream hStream) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemcpyDtoDAsync_v2_(dstDevice, srcDevice, bytesize, hStream);
  }
  Result MemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                         CUdeviceptr srcDevice, CUcontext srcContext,
                         size_t bytesize, CUstream hStream) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemcpyPeerAsync_(dstDevice, dstContext, srcDevice, srcContext,
                              bytesize, hStream);
  }

  Result MemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost,
                    size_t bytesize) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemcpyHtoD_v2_(dstDevice, srcHost, bytesize);
  }
  Result MemcpyDtoH(void *dstHost, CUdeviceptr srcDevice,
                    size_t bytesize) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemcpyDtoH_v2_(dstHost, srcDevice, bytesize);
  }
  Result MemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                    size_t bytesize) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemcpyDtoD_v2_(dstDevice, srcDevice, bytesize);
  }
  Result MemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                    CUdeviceptr srcDevice, CUcontext srcContext,
                    size_t bytesize) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemcpyPeer_(dstDevice, dstContext, srcDevice, srcContext,
                         bytesize);
  }

  Result MemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N,
                       CUstream hStream) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemsetD8Async_(dstDevice, uc, N, hStream);
  }
  Result MemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N,
                        CUstream hStream) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemsetD16Async_(dstDevice, us, N, hStream);
  }
  Result MemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N,
                        CUstream hStream) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemsetD32Async_(dstDevice, ui, N, hStream);
  }
  Result MemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch,
                          unsigned int ui, size_t Width, size_t Height,
                          CUstream hStream) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemsetD2D32Async_(dstDevice, dstPitch, ui, Width, Height, hStream);
  }

  Result MemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemsetD8_v2_(dstDevice, uc, N);
  }
  Result MemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemsetD16_v2_(dstDevice, us, N);
  }
  Result MemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemsetD32_v2_(dstDevice, ui, N);
  }
  Result MemsetD2D32(CUdeviceptr dstDevice, unsigned int dstPitch,
                     unsigned int ui, size_t Width, size_t Height) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemsetD2D32_v2_(dstDevice, dstPitch, ui, Width, Height);
  }

  Result LinkCreate(unsigned int numOptions, CUjit_option *options,
                    void **optionValues, CUlinkState *stateOut) const {
    CUDA_DRIVER_CALL_LOG();
    return cuLinkCreate_v2_(numOptions, options, optionValues, stateOut);
  }

  Result LinkAddData(CUlinkState state, CUjitInputType type, void *data,
                     size_t size, const char *name, unsigned int numOptions,
                     CUjit_option *options, void **optionValues) const {
    CUDA_DRIVER_CALL_LOG();
    return cuLinkAddData_v2_(state, type, data, size, name, numOptions, options,
                             optionValues);
  }

  Result LinkComplete(CUlinkState state, void **cubinOut,
                      size_t *sizeOut) const {
    CUDA_DRIVER_CALL_LOG();
    return cuLinkComplete_(state, cubinOut, sizeOut);
  }

  Result LinkDestroy(CUlinkState state) const {
    CUDA_DRIVER_CALL_LOG();
    return cuLinkDestroy_(state);
  }

  Result ModuleLoadDataEx(CUmodule *module, const void *image,
                          unsigned int numOptions, CUjit_option *options,
                          void **optionValues) const {
    CUDA_DRIVER_CALL_LOG();
    return cuModuleLoadDataEx_(module, image, numOptions, options,
                               optionValues);
  }
  Result ModuleGetGlobal(CUdeviceptr *dptr, size_t *sizeptr, CUmodule hmod,
                         const char *name) const {
    CUDA_DRIVER_CALL_LOG();
    return cuModuleGetGlobal_v2_(dptr, sizeptr, hmod, name);
  }
  Result ModuleUnload(CUmodule hmod) const {
    CUDA_DRIVER_CALL_LOG();
    return cuModuleUnload_(hmod);
  }
  Result ModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                           const char *name) const {
    CUDA_DRIVER_CALL_LOG();
    return cuModuleGetFunction_(hfunc, hmod, name);
  }
  Result FuncGetAttribute(int *pi, CUfunction_attribute attrib,
                          CUfunction hfunc) const {
    CUDA_DRIVER_CALL_LOG();
    return cuFuncGetAttribute_(pi, attrib, hfunc);
  }
  Result FuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib,
                          int value) const {
    CUDA_DRIVER_CALL_LOG();
    return cuFuncSetAttribute_(hfunc, attrib, value);
  }

  Result LaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData) const {
    CUDA_DRIVER_CALL_LOG();
    return cuLaunchHostFunc_(hStream, fn, userData);
  }
  Result LaunchKernel(CUfunction f, unsigned int gridDimX,
                      unsigned int gridDimY, unsigned int gridDimZ,
                      unsigned int blockDimX, unsigned int blockDimY,
                      unsigned int blockDimZ, unsigned int sharedMemBytes,
                      CUstream hStream, void **kernelParams,
                      void **extra) const {
    CUDA_DRIVER_CALL_LOG();
    return cuLaunchKernel_(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                           blockDimY, blockDimZ, sharedMemBytes, hStream,
                           kernelParams, extra);
  }
  Result LaunchKernelEx(const CUlaunchConfig *config, CUfunction hfunc,
                        void **kernelParams, void **extra) const {
    CUDA_DRIVER_CALL_LOG();
    return cuLaunchKernelEx_(config, hfunc, kernelParams, extra);
  }

  Result
  EventCreate(CUevent *phEvent,
              CUevent_flags flags = CUevent_flags::CU_EVENT_DEFAULT) const {
    CUDA_DRIVER_CALL_LOG();
    return cuEventCreate_(phEvent, static_cast<unsigned int>(flags));
  }
  Result EventDestroy(CUevent hEvent) const {
    CUDA_DRIVER_CALL_LOG();
    return cuEventDestroy_(hEvent);
  }
  Result EventRecord(CUevent hEvent, CUstream hStream) const {
    CUDA_DRIVER_CALL_LOG();
    return cuEventRecord_(hEvent, hStream);
  }
  Result EventQuery(CUevent hEvent) const {
    CUDA_DRIVER_CALL_LOG();
    return cuEventQuery_(hEvent);
  }
  Result EventSynchronize(CUevent hEvent) const {
    CUDA_DRIVER_CALL_LOG();
    return cuEventSynchronize_(hEvent);
  }
  Result EventElapsedTime(float *pMilliseconds, CUevent hStart,
                          CUevent hEnd) const {
    CUDA_DRIVER_CALL_LOG();
    return cuEventElapsedTime_(pMilliseconds, hStart, hEnd);
  }

  Result DeviceGetMemPool(CUmemoryPool *pool, CUdevice device) const {
    CUDA_DRIVER_CALL_LOG();
    return cuDeviceGetMemPool_(pool, device);
  }

  Result MemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *desc,
                          int count) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemPoolSetAccess_(pool, desc, count);
  }
  Result MemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemPoolTrimTo_(pool, minBytesToKeep);
  }

  Result GetMemInfo(size_t *free, size_t *total) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemGetInfo_v2_(free, total);
  }

  Result
  MemGetAllocationGranularity(size_t *granularity,
                              const CUmemAllocationProp *prop,
                              CUmemAllocationGranularity_flags option) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemGetAllocationGranularity_(granularity, prop, option);
  }

  Result MemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                   const CUmemAllocationProp *prop,
                   unsigned long long flags) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemCreate_(handle, size, prop, flags);
  }

  Result MemRelease(CUmemGenericAllocationHandle handle) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemRelease_(handle);
  }

  Result MemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                           CUdeviceptr addr, unsigned long long flags) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemAddressReserve_(ptr, size, alignment, addr, flags);
  }

  Result MemAddressFree(CUdeviceptr ptr, size_t size) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemAddressFree_(ptr, size);
  }

  Result MemMap(CUdeviceptr ptr, size_t size, size_t offset,
                CUmemGenericAllocationHandle handle,
                unsigned long long flags) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemMap_(ptr, size, offset, handle, flags);
  }

  Result MemUnmap(CUdeviceptr ptr, size_t size) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemUnmap_(ptr, size);
  }

  Result MemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc,
                      size_t count) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMemSetAccess_(ptr, size, desc, count);
  }

  Result MulticastCreate(CUmemGenericAllocationHandle *mcHandle,
                         const CUmulticastObjectProp *prop) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMulticastCreate_(mcHandle, prop);
  }

  Result MulticastGetGranularity(size_t *granularity,
                                 const CUmulticastObjectProp *prop,
                                 CUmulticastGranularity_flags option) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMulticastGetGranularity_(granularity, prop, option);
  }

  Result MulticastAddDevice(CUmemGenericAllocationHandle mcHandle,
                            CUdevice dev) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMulticastAddDevice_(mcHandle, dev);
  }

  Result MulticastBindMem(CUmemGenericAllocationHandle mcHandle,
                          size_t mcOffset,
                          CUmemGenericAllocationHandle memHandle,
                          size_t memOffset, size_t size,
                          unsigned long long flags) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMulticastBindMem_(mcHandle, mcOffset, memHandle, memOffset, size,
                               flags);
  }

  Result MulticastUnbind(CUmemGenericAllocationHandle mcHandle, CUdevice dev,
                         size_t mcOffset, size_t size) const {
    CUDA_DRIVER_CALL_LOG();
    return cuMulticastUnbind_(mcHandle, dev, mcOffset, size);
  }

  Result PointerGetAttributes(unsigned int numAttributes,
                              CUpointer_attribute *attributes, void **data,
                              CUdeviceptr ptr) const {
    CUDA_DRIVER_CALL_LOG();
    return cuPointerGetAttributes_(numAttributes, attributes, data, ptr);
  }

  Result TensorMapEncodeTiled(
      void *tensorMap, CUtensorMapDataType tensorDataType, uint32_t tensorRank,
      void *globalAddress, const uint64_t *globalDim,
      const uint64_t *globalStrides, const uint32_t *boxDim,
      const uint32_t *elementStrides, CUtensorMapInterleave interleave,
      CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
      CUtensorMapFloatOOBfill oobFill) const {
    CUDA_DRIVER_CALL_LOG();
    return cuTensorMapEncodeTiled_(tensorMap, tensorDataType, tensorRank,
                                   globalAddress, globalDim, globalStrides,
                                   boxDim, elementStrides, interleave, swizzle,
                                   l2Promotion, oobFill);
  }

#ifdef MODULAR_LINUX
  Result NVPTXCompilerGetVersion(unsigned int *major,
                                 unsigned int *minor) const {
    CUDA_DRIVER_CALL_LOG();
    return static_cast<Result>(nvPTXCompilerGetVersion(major, minor));
  }

  Result NVPTXCompilerCompile(nvPTXCompilerHandle compiler,
                              int numCompileOptions,
                              const char *const *compileOptions) const {

    CUDA_DRIVER_CALL_LOG();
    return static_cast<Result>(
        nvPTXCompilerCompile(compiler, numCompileOptions, compileOptions));
  }

  Result NVPTXCompilerCreate(nvPTXCompilerHandle *compiler, size_t ptxCodeLen,
                             const char *ptxCode) const {

    CUDA_DRIVER_CALL_LOG();
    return static_cast<Result>(
        nvPTXCompilerCreate(compiler, ptxCodeLen, ptxCode));
  }

  Result NVPTXCompilerDestroy(nvPTXCompilerHandle *compiler) const {

    CUDA_DRIVER_CALL_LOG();
    return static_cast<Result>(nvPTXCompilerDestroy(compiler));
  }

  Result NVPTXCompilerGetCompiledProgram(nvPTXCompilerHandle compiler,
                                         void *binaryImage) const {

    CUDA_DRIVER_CALL_LOG();
    return static_cast<Result>(
        nvPTXCompilerGetCompiledProgram(compiler, binaryImage));
  }

  Result NVPTXCompilerGetCompiledProgramSize(nvPTXCompilerHandle compiler,
                                             size_t *binaryImageSize) const {

    CUDA_DRIVER_CALL_LOG();
    return static_cast<Result>(
        nvPTXCompilerGetCompiledProgramSize(compiler, binaryImageSize));
  }

  Result NVPTXCompilerGetErrorLogSize(nvPTXCompilerHandle compiler,
                                      size_t *errorLogSize) const {

    CUDA_DRIVER_CALL_LOG();
    return static_cast<Result>(
        nvPTXCompilerGetErrorLogSize(compiler, errorLogSize));
  }

  Result NVPTXCompilerGetErrorLog(nvPTXCompilerHandle compiler,
                                  char *errorLog) const {
    CUDA_DRIVER_CALL_LOG();
    return static_cast<Result>(nvPTXCompilerGetErrorLog(compiler, errorLog));
  }

  Result NVPTXCompilerGetInfoLog(nvPTXCompilerHandle compiler,
                                 char *infoLog) const {
    CUDA_DRIVER_CALL_LOG();
    return static_cast<Result>(nvPTXCompilerGetInfoLog(compiler, infoLog));
  }

  Result NVPTXCompilerGetInfoLogSize(nvPTXCompilerHandle compiler,
                                     size_t *infoLogSize) const {
    CUDA_DRIVER_CALL_LOG();
    return static_cast<Result>(
        nvPTXCompilerGetInfoLogSize(compiler, infoLogSize));
  }
#endif // MODULAR_LINUX

  static ErrorOr<const CUDADriver &> get();

private:
  // Enable in-depth logging at build time.
  static constexpr bool do_call_log = false;

  // Shared library name for the CUDA driver.
  static const char lib_name[];
  static const char lib_nvptx_name[];
  static const char lib_nvml_name[];

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

  using tcuDeviceGet = Result(CUdevice *, int);
  tcuDeviceGet *cuDeviceGet_;

  using tcuDeviceGetMemPool = Result(CUmemoryPool *, CUdevice);
  tcuDeviceGetMemPool *cuDeviceGetMemPool_;

  using tcuDeviceGetAttribute = Result(int *, CUdevice_attribute, CUdevice);
  tcuDeviceGetAttribute *cuDeviceGetAttribute_;

  using tcuDeviceGetCount = Result(int *);
  tcuDeviceGetCount *cuDeviceGetCount_;

  using tcuDeviceGetName = Result(char *, int, CUdevice);
  tcuDeviceGetName *cuDeviceGetName_;

  using tcuDeviceCanAccessPeer = Result(int *, CUdevice, CUdevice);
  tcuDeviceCanAccessPeer *cuDeviceCanAccessPeer_;

  using tcuDeviceGetP2PAttribute = Result(int *, CUdevice_P2PAttribute,
                                          CUdevice, CUdevice);
  tcuDeviceGetP2PAttribute *cuDeviceGetP2PAttribute_;

  using tcuDevicePrimaryCtxRetain = Result(CUcontext *, CUdevice);
  tcuDevicePrimaryCtxRetain *cuDevicePrimaryCtxRetain_;

  using tcuDevicePrimaryCtxRelease_v2 = Result(CUdevice);
  tcuDevicePrimaryCtxRelease_v2 *cuDevicePrimaryCtxRelease_v2_;

  using tcuCtxCreate_v2 = Result(CUcontext *, unsigned int, CUdevice);
  tcuCtxCreate_v2 *cuCtxCreate_v2_;

  using tcuCtxDestroy = Result(CUcontext);
  tcuCtxDestroy *cuCtxDestroy_;

  using tcuCtxSetCurrent = Result(CUcontext);
  tcuCtxSetCurrent *cuCtxSetCurrent_;

  using tcuCtxGetCurrent = Result(CUcontext *);
  tcuCtxGetCurrent *cuCtxGetCurrent_;

  using tcuCtxPushCurrent = Result(CUcontext);
  tcuCtxPushCurrent *cuCtxPushCurrent_;

  using tcuCtxPopCurrent = Result(CUcontext *);
  tcuCtxPopCurrent *cuCtxPopCurrent_;

  using tcuCtxEnablePeerAccess = Result(CUcontext, unsigned int);
  tcuCtxEnablePeerAccess *cuCtxEnablePeerAccess_;

  using tcuStreamCreate = Result(CUstream *, unsigned int);
  tcuStreamCreate *cuStreamCreate_;

  using tcuStreamDestroy = Result(CUstream);
  tcuStreamDestroy *cuStreamDestroy_;

  using tcuStreamSynchronize = Result(CUstream);
  tcuStreamSynchronize *cuStreamSynchronize_;

  using tcuStreamWaitEvent = Result(CUstream, CUevent, unsigned int);
  tcuStreamWaitEvent *cuStreamWaitEvent_;

  using tcuStreamCreateWithPriority = Result(CUstream *, unsigned int, int);
  tcuStreamCreateWithPriority *cuStreamCreateWithPriority_;

  using tcuCtxGetStreamPriorityRange = Result(int *, int *);
  tcuCtxGetStreamPriorityRange *cuCtxGetStreamPriorityRange_;

  using tcuMemAllocHost_v2 = Result(void **, size_t);
  tcuMemAllocHost_v2 *cuMemAllocHost_v2_;

  using tcuMemFreeHost = Result(void *);
  tcuMemFreeHost *cuMemFreeHost_;

  using tcuMemAllocAsync = Result(CUdeviceptr *, size_t, CUstream);
  tcuMemAllocAsync *cuMemAllocAsync_;

  using tcuMemFreeAsync = Result(CUdeviceptr, CUstream);
  tcuMemFreeAsync *cuMemFreeAsync_;

  using tcuMemPoolSetAccess = Result(CUmemoryPool, const CUmemAccessDesc *,
                                     size_t);
  tcuMemPoolSetAccess *cuMemPoolSetAccess_;

  using tcuMemPoolTrimTo = Result(CUmemoryPool, size_t);
  tcuMemPoolTrimTo *cuMemPoolTrimTo_;

  using tcuMemAlloc_v2 = Result(CUdeviceptr *, size_t);
  tcuMemAlloc_v2 *cuMemAlloc_v2_;

  using tcuMemFree_v2 = Result(CUdeviceptr);
  tcuMemFree_v2 *cuMemFree_v2_;

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

  using tcuMemsetD2D32Async = Result(CUdeviceptr, size_t, unsigned int, size_t,
                                     size_t, CUstream);
  tcuMemsetD2D32Async *cuMemsetD2D32Async_;

  using tcuMemsetD8_v2 = Result(CUdeviceptr, unsigned char, size_t);
  tcuMemsetD8_v2 *cuMemsetD8_v2_;

  using tcuMemsetD16_v2 = Result(CUdeviceptr, unsigned short, size_t);
  tcuMemsetD16_v2 *cuMemsetD16_v2_;

  using tcuMemsetD32_v2 = Result(CUdeviceptr, unsigned int, size_t);
  tcuMemsetD32_v2 *cuMemsetD32_v2_;

  using tcuMemsetD2D32_v2 = Result(CUdeviceptr, unsigned int, unsigned int,
                                   size_t, size_t);
  tcuMemsetD2D32_v2 *cuMemsetD2D32_v2_;

  using tcuLinkCreate_v2 = Result(unsigned int numOptions,
                                  CUjit_option *options, void **optionValues,
                                  CUlinkState *stateOut);
  tcuLinkCreate_v2 *cuLinkCreate_v2_;

  using tcuLinkAddData_v2 = Result(CUlinkState state, CUjitInputType type,
                                   void *data, size_t size, const char *name,
                                   unsigned int numOptions,
                                   CUjit_option *options, void **optionValues);
  tcuLinkAddData_v2 *cuLinkAddData_v2_;

  using tcuLinkComplete = Result(CUlinkState state, void **cubinOut,
                                 size_t *sizeOut);
  tcuLinkComplete *cuLinkComplete_;

  using tcuLinkDestroy = Result(CUlinkState state);
  tcuLinkDestroy *cuLinkDestroy_;

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

  using tcuLaunchHostFunc = Result(CUstream, CUhostFn fn, void *);
  tcuLaunchHostFunc *cuLaunchHostFunc_;

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

  using tcuMemGetAllocationGranularity = Result(
      size_t *, const CUmemAllocationProp *, CUmemAllocationGranularity_flags);
  tcuMemGetAllocationGranularity *cuMemGetAllocationGranularity_;

  using tcuMemCreate = Result(CUmemGenericAllocationHandle *handle, size_t size,
                              const CUmemAllocationProp *prop,
                              unsigned long long flags);
  tcuMemCreate *cuMemCreate_;

  using tcuMemRelease = Result(CUmemGenericAllocationHandle handle);
  tcuMemRelease *cuMemRelease_;

  using tcuMemAddressReserve = Result(CUdeviceptr *ptr, size_t size,
                                      size_t alignment, CUdeviceptr addr,
                                      unsigned long long flags);
  tcuMemAddressReserve *cuMemAddressReserve_;

  using tcuMemAddressFree = Result(CUdeviceptr ptr, size_t size);
  tcuMemAddressFree *cuMemAddressFree_;

  using tcuMemMap = Result(CUdeviceptr ptr, size_t size, size_t offset,
                           CUmemGenericAllocationHandle handle,
                           unsigned long long flags);
  tcuMemMap *cuMemMap_;

  using tcuMemUnmap = Result(CUdeviceptr ptr, size_t size);
  tcuMemUnmap *cuMemUnmap_;

  using tcuMemSetAccess = Result(CUdeviceptr ptr, size_t size,
                                 const CUmemAccessDesc *desc, size_t count);
  tcuMemSetAccess *cuMemSetAccess_;

  using tcuMulticastCreate = Result(CUmemGenericAllocationHandle *,
                                    const CUmulticastObjectProp *);
  tcuMulticastCreate *cuMulticastCreate_;

  using tcuMulticastGetGranularity = Result(size_t *,
                                            const CUmulticastObjectProp *,
                                            CUmulticastGranularity_flags);
  tcuMulticastGetGranularity *cuMulticastGetGranularity_;

  using tcuMulticastAddDevice = Result(CUmemGenericAllocationHandle, CUdevice);
  tcuMulticastAddDevice *cuMulticastAddDevice_;

  using tcuMulticastBindMem = Result(CUmemGenericAllocationHandle mcHandle,
                                     size_t mcOffset,
                                     CUmemGenericAllocationHandle memHandle,
                                     size_t memOffset, size_t size,
                                     unsigned long long flags);
  tcuMulticastBindMem *cuMulticastBindMem_;

  using tcuMulticastUnbind = Result(CUmemGenericAllocationHandle mcHandle,
                                    CUdevice dev, size_t mcOffset, size_t size);
  tcuMulticastUnbind *cuMulticastUnbind_;

  using tcuPointerGetAttributes = Result(unsigned int numAttributes,
                                         CUpointer_attribute *attributes,
                                         void **data, CUdeviceptr ptr);
  tcuPointerGetAttributes *cuPointerGetAttributes_;

  using tcuTensorMapEncodeTiled = Result(
      void *tensorMap, CUtensorMapDataType tensorDataType, uint32_t tensorRank,
      void *globalAddress, const uint64_t *globalDim,
      const uint64_t *globalStrides, const uint32_t *boxDim,
      const uint32_t *elementStrides, CUtensorMapInterleave interleave,
      CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
      CUtensorMapFloatOOBfill oobFill);
  tcuTensorMapEncodeTiled *cuTensorMapEncodeTiled_;

  static ErrorOr<const CUDADriver &> allocAndInitDriver();
};

#undef CUDA_DRIVER_CALL_LOG

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_CUDA_CUDADRIVER_H
