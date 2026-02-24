//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the CUDA driver interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDADRIVER_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDADRIVER_H

#include "Support/ErrorOr.h"
#include "Support/Preprocessor.h"
#include "Support/Profiling/Tracy.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <thread>

namespace M::Driver {

using CUcontext = struct CUctx_st *;
using CUdevice = int;
using CUdeviceptr = void *;
using CUfunction = struct CUfunc_st *;
using CUmemGenericAllocationHandle = unsigned long long;
using CUmemoryPool = struct CUmemPoolHandle_st *;
using CUmodule = struct CUmod_st *;
using CUstream = struct CUstream_st *;
using CUevent = struct CUevent_st *;
using CUgraph = struct CUgraph_st *;
using CUgraphExec = struct CUgraphExec_st *;
using CUgraphNode = struct CUgraphNode_st *;
using CUhostFn = void(void *);

enum class CUstreamCaptureMode {
  CU_STREAM_CAPTURE_MODE_GLOBAL = 0,
  CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
  CU_STREAM_CAPTURE_MODE_RELAXED = 2,
};

enum class CUstreamCaptureStatus {
  CU_STREAM_CAPTURE_STATUS_NONE = 0,
  CU_STREAM_CAPTURE_STATUS_ACTIVE = 1,
  CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2,
};

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

/// Graph node types for CUDA graph introspection.
/// These match the CUDA driver API CUgraphNodeType enum values.
enum class CUgraphNodeType {
  CU_GRAPH_NODE_TYPE_KERNEL = 0,
  CU_GRAPH_NODE_TYPE_MEMCPY = 1,
  CU_GRAPH_NODE_TYPE_MEMSET = 2,
  CU_GRAPH_NODE_TYPE_HOST = 3,
  CU_GRAPH_NODE_TYPE_GRAPH = 4,
  CU_GRAPH_NODE_TYPE_EMPTY = 5,
  CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6,
  CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7,
  CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8,
  CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9,
  CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10,
  CU_GRAPH_NODE_TYPE_MEM_FREE = 11,
  CU_GRAPH_NODE_TYPE_BATCH_MEM_OP = 12,
};

enum class CUgraphInstantiateFlags : unsigned long long {
  CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 1,
  CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD = 2,
  CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH = 4,
  CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = 8,
};

// Result codes from cuGraphInstantiateWithParams
enum class CUgraphInstantiateResult {
  CUDA_GRAPH_INSTANTIATE_SUCCESS = 0,
  CUDA_GRAPH_INSTANTIATE_ERROR = 1,
  CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE = 2,
  CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED = 3,
  CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED = 4,
};

/// Convert CUgraphInstantiateResult to a human-readable string.
inline llvm::StringLiteral toString(CUgraphInstantiateResult result) {
  switch (result) {
  case CUgraphInstantiateResult::CUDA_GRAPH_INSTANTIATE_SUCCESS:
    return "success";
  case CUgraphInstantiateResult::CUDA_GRAPH_INSTANTIATE_ERROR:
    return "generic error";
  case CUgraphInstantiateResult::CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE:
    return "invalid structure";
  case CUgraphInstantiateResult::
      CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED:
    return "node operation not supported";
  case CUgraphInstantiateResult::
      CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED:
    return "multiple contexts not supported";
  }
  return "unknown result";
}

// Parameters structure for cuGraphInstantiateWithParams
// This matches CUDA_GRAPH_INSTANTIATE_PARAMS from cuda.h
struct CUDA_GRAPH_INSTANTIATE_PARAMS {
  uint64_t flags;           // Instantiation flags (CUgraphInstantiateFlags)
  CUstream hUploadStream;   // Upload stream (can be nullptr)
  CUgraphNode hErrNode_out; // Output: node that caused error
  CUgraphInstantiateResult result_out; // Output: detailed error result
};

enum class CUgraphDotFlags : unsigned int {
  None = 0,
  Verbose = 0x01, // 1: Output all debug data

  // Detailed parameter flags
  KernelNodeParams = 0x04,         // 4: Grid/Block dims, shared mem, etc.
  MemcpyNodeParams = 0x08,         // 8: Src/Dst addrs, width/height/pitch
  MemsetNodeParams = 0x10,         // 16: Dst addr, value, width/height/pitch
  HostNodeParams = 0x20,           // 32: Host function details
  EventNodeParams = 0x40,          // 64: Event record/wait details
  ExtSemasSignalNodeParams = 0x80, // 128: External semaphore signal details
  ExtSemasWaitNodeParams = 0x100,  // 256: External semaphore wait details
  KernelNodeAttributes = 0x200,    // 512: Kernel attributes
  Handles = 0x400                  // 1024: Internal handles for nodes/kernels
};

/// Kernel node parameters structure for CUDA graph introspection.
/// This matches the CUDA_KERNEL_NODE_PARAMS_v2 structure from the CUDA driver.
struct CUDA_KERNEL_NODE_PARAMS_v2 {
  CUfunction func;             // Kernel function
  unsigned int gridDimX;       // Grid X dimension
  unsigned int gridDimY;       // Grid Y dimension
  unsigned int gridDimZ;       // Grid Z dimension
  unsigned int blockDimX;      // Block X dimension
  unsigned int blockDimY;      // Block Y dimension
  unsigned int blockDimZ;      // Block Z dimension
  unsigned int sharedMemBytes; // Dynamic shared memory size
  void **kernelParams;         // Array of pointers to kernel parameters
  void **extra;                // Extra options (usually nullptr)
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

/// Check if CUDA API logging is enabled at runtime via MLRT_CUDA_DEBUG env var.
inline bool isCUDADebugLoggingEnabled() {
  static const bool enabled =
      llvm::sys::Process::GetEnv("MLRT_CUDA_DEBUG").has_value();
  return enabled;
}

/// Runtime logging macro for CUDA driver API calls.
#define CUDA_DRIVER_CALL_LOG()                                                 \
  if (isCUDADebugLoggingEnabled()) {                                           \
    llvm::errs() << "[CUDA] " << __FUNCTION__ << "\n";                         \
  }

class CUDADriver {

public:
  enum Result : unsigned int {
    SUCCESS = 0,
    NOT_READY = 600,
  };

// Define wrapper member function with comprehensive logging.
// The macro generates a wrapper that:
// 1. Logs entry with function name when MLRT_CUDA_DEBUG is set
// 2. Adds Tracy profiling zone
// 3. Calls the actual CUDA driver function
// 4. Logs exit with result code when MLRT_CUDA_DEBUG is set
#define CUDA_FUNC(name, ret, ...)                                              \
  ret name(EXPAND_PARAMS(__VA_ARGS__)) const {                                 \
    CUDA_DRIVER_CALL_LOG();                                                    \
    TRACY_ZONE_SCOPED_NC(#name, TRACY_COLOR_GREEN);                            \
    ret result = name##_(EXPAND_ARGS(__VA_ARGS__));                            \
    if (isCUDADebugLoggingEnabled()) {                                         \
      llvm::errs() << "[CUDA] " << #name << " -> " << result << "\n";          \
    }                                                                          \
    return result;                                                             \
  }

#include "CUDADriver.inc"
#undef CUDA_FUNC

  // Hand-written wrapper for cuTensorMapEncodeIm2col (exceeds macro limit).
  Result cuTensorMapEncodeIm2col(
      void *arg1, CUtensorMapDataType arg2, uint32_t arg3, void *arg4,
      const uint64_t *arg5, const uint64_t *arg6, const int *arg7,
      const int *arg8, uint32_t arg9, uint32_t arg10, const uint32_t *arg11,
      CUtensorMapInterleave arg12, CUtensorMapSwizzle arg13,
      CUtensorMapL2promotion arg14, CUtensorMapFloatOOBfill arg15) const {
    CUDA_DRIVER_CALL_LOG();
    TRACY_ZONE_SCOPED_NC("cuTensorMapEncodeIm2col", TRACY_COLOR_GREEN);
    return cuTensorMapEncodeIm2col_(arg1, arg2, arg3, arg4, arg5, arg6, arg7,
                                    arg8, arg9, arg10, arg11, arg12, arg13,
                                    arg14, arg15);
  }

  static ErrorOr<const CUDADriver &> get();

  // CUDADriver object lifetime is tied to a static unique_ptr.
  CUDADriver() = default;
  ~CUDADriver() = default;

private:
  // Shared library name for the CUDA driver.
  static const char lib_name[];

  // Driver is only initialized from internal code.
  Result Init(unsigned int flags) const { return cuInit_(flags); }

#define CUDA_FUNC(name, ret, ...)                                              \
  using t##name = ret(__VA_ARGS__);                                            \
  t##name *name##_;

#include "CUDADriver.inc"

#undef CUDA_FUNC

  // Hand-written because cuTensorMapEncodeIm2col has 15 parameters,
  // exceeding the EXPAND_PARAMS macro limit (12).
  using tcuTensorMapEncodeIm2col =
      Result(void *, CUtensorMapDataType, uint32_t, void *, const uint64_t *,
             const uint64_t *, const int *, const int *, uint32_t, uint32_t,
             const uint32_t *, CUtensorMapInterleave, CUtensorMapSwizzle,
             CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  tcuTensorMapEncodeIm2col *cuTensorMapEncodeIm2col_;

  static ErrorOr<std::unique_ptr<CUDADriver>> allocAndInitDriver();
};

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDADRIVER_H
