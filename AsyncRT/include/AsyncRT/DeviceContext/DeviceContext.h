//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the M::AsyncRT::DeviceContext interface.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_DEVICECONTEXT_DEVICECONTEXT_H
#define ASYNCRT_DEVICECONTEXT_DEVICECONTEXT_H

#include "Support/Buffer.h"
#include "Support/Configuration.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"

#include "llvm/ADT/StringRef.h"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <stddef.h>
#include <string>
#include <vector>

namespace M::AsyncRT {

class BufferCache;
class Device;
class DeviceBuffer;
class DeviceContext;
class DeviceEvent;
class DeviceFunction;
class DeviceFunctionCacheEntry;
class DeviceMulticastBuffer;
class DeviceStream;
class DeviceTimer;

using DeviceRef = RCRef<const Device>;
using DeviceFunctionCacheEntryRef = RCRef<const DeviceFunctionCacheEntry>;
using DeviceBufferRef = RCRef<const DeviceBuffer>;
using DeviceContextRef = RCRef<const DeviceContext>;
using DeviceEventRef = RCRef<const DeviceEvent>;
using DeviceFunctionRef = RCRef<const DeviceFunction>;
using DeviceMulticastBufferRef = RCRef<const DeviceMulticastBuffer>;
using DeviceStreamRef = RCRef<const DeviceStream>;
using DeviceTimerRef = RCRef<const DeviceTimer>;

size_t getPreferredAlignment(std::string label);

class Device : public ReferenceCounted<Device> {
public:
  enum class Attribute;

  // API is a constexpr StringLiteral for fast comparison, while being
  // human-readable at the same time.
  using API = StringLiteral;
  static constexpr API cpuAPI = "cpu";   // Generic CPU device
  static constexpr API gpuAPI = "gpu";   // Generic GPU device
  static constexpr API cudaAPI = "cuda"; // NVIDIA CUDA device
  static constexpr API hipAPI = "hip";   // AMDGPU HIP device

  virtual ~Device();

  const std::string &name() const;
  int id() const;

  API api() const;
  StringRef vendor() const;

  virtual ErrorOr<int> getDriverVersion() const = 0;
  virtual ErrorOrSuccess isCompatible() const = 0;
  virtual ErrorOr<int> computeCapability() const = 0;

  size_t allocationAlignment() const;

  virtual ErrorOr<bool> canAccess(const Device &peer) const = 0;
  virtual ErrorOr<int> getAttribute(Attribute attr) const = 0;

  virtual ErrorOr<bool> supportsMulticast() const;

  virtual int deviceCount() const = 0;

  virtual ErrorOr<DeviceContextRef>
  createContext(size_t bufferCacheSize) const = 0;

  virtual ErrorOr<DeviceStreamRef>
  createExternalStream(void *externalStream) const = 0;

  static int numberOfDevices(StringRef api);
  static size_t preferredAlignment(StringRef api);

  static ErrorOr<DeviceRef> create(StringRef api, int id = 0);

  /// Returns the vendor and compute_capability of the accelerator in the form
  /// of `vendor:compute_capability`. For example, an A100 GPU would return
  /// `nvidia:80`.
  static const std::string &getAcceleratorArchOrEmpty();

  // Internal implementation helpers for DeviceContexts.
  DeviceFunctionCacheEntryRef findInFunctionCache(StringRef key) const;
  DeviceFunctionCacheEntryRef
  insertInFunctionCache(StringRef key,
                        const DeviceFunctionCacheEntry &entry) const;
  void clearFunctionCache() const;

protected:
  Device(API api, std::string name, int id, size_t allocAligment);

private:
  std::string name_;
  int id_;
  API api_;
  size_t allocAlignment_;

  // Mutex to protect mutable data.
  mutable std::mutex mu_;
  // The function cache holds the lower-level function references which do not
  // contain a device context reference to avoid reference counting cycles.
  mutable llvm::StringMap<DeviceFunctionCacheEntryRef> funcMap_;
};

class DeviceBuffer final : public ReferenceCounted<DeviceBuffer> {
public:
  ~DeviceBuffer();

  const DeviceContext &context() const;
  const DeviceBuffer *parent() const;

  // Returns the raw device pointer, which is not safe to access on the host
  void *ptr() const;
  // TODO: Remove the V1 compatibility.
  void *releasePtr() const;

  size_t byteSize() const;

  bool isHost() const;

  ErrorOrSuccess copyTo(const DeviceBuffer &dst) const;
  ErrorOrSuccess copyTo(void *dst) const;

  ErrorOrSuccess copyFrom(const DeviceBuffer &src) const;
  ErrorOrSuccess copyFrom(const void *src) const;

  ErrorOrSuccess reassignOwnershipTo(const DeviceContext &ctx) const;

  ErrorOr<DeviceBufferRef> createSubBuffer(size_t byteOffset,
                                           size_t byteSize) const;

private:
  friend class DeviceContext;

  mutable DeviceContextRef context_;
  DeviceBufferRef parent_;
  mutable void *ptr_;
  size_t byteSize_;
  bool isHost_;
  // TODO: Remove the V1 compatibility.
  mutable bool owning_ = true;

  DeviceBuffer(const DeviceContext &context, void *ptr, size_t byteSize,
               bool isHost);
  DeviceBuffer(const DeviceContext &context, const DeviceBuffer &parent,
               void *ptr, size_t byteSize, bool isHost);
  // TODO: Remove the V1 compatibility.
  DeviceBuffer(const DeviceContext &context, void *ptr, size_t byteSize,
               bool isHost, bool owning);
};

class DeviceMulticastBuffer : public ReferenceCounted<DeviceMulticastBuffer> {
public:
  virtual ~DeviceMulticastBuffer();

  virtual ErrorOr<DeviceBufferRef>
  multicastBuffer(const DeviceContext &ctx) const = 0;
  virtual ErrorOr<DeviceBufferRef>
  unicastBufferFor(const DeviceContext &ctx) const = 0;

  size_t byteSize() const;

  virtual size_t numberOfDevices() const = 0;

  static ErrorOr<DeviceMulticastBufferRef>
  allocate(const std::vector<DeviceContextRef> &contexts, size_t byteSize);

protected:
  DeviceMulticastBuffer(size_t byteSize);

private:
  size_t byteSize_;
};

class DeviceEvent : public ReferenceCounted<DeviceEvent> {
public:
  virtual ~DeviceEvent() = default;
  const Device &device() const;

  virtual ErrorOrSuccess synchronize() const = 0;

protected:
  DeviceEvent(const Device &device);

private:
  DeviceRef device_;
};

class DeviceStream : public ReferenceCounted<DeviceStream> {
public:
  virtual ~DeviceStream() = default;
  const Device &device() const;

  virtual ErrorOrSuccess synchronize() const = 0;

  virtual ErrorOr<DeviceEventRef> enqueueEvent() const = 0;

  virtual ErrorOrSuccess
  enqueueWaitForEvent(const DeviceEvent &event) const = 0;

  // Raw device driver pointer to enable passing to PyTorch and similar APIs.
  virtual void *ptr() const = 0;

protected:
  DeviceStream(const Device &device);

private:
  DeviceRef device_;
};

class DeviceFunction : public ReferenceCounted<DeviceFunction> {
public:
  enum ArgumentType : uint8_t;
  class SignatureBuilder;
  class Arguments;

  using ArgumentsRef = RCRef<const Arguments>;

  virtual ~DeviceFunction() = default;

  const DeviceContext &context() const;

  virtual const std::string &name() const = 0;

  virtual ErrorOrSuccess copyToNamedConstant(const char *name, const void *data,
                                             size_t byteSize) const = 0;

  virtual ErrorOr<int32_t> getAttribute(int32_t code) const = 0;

  template <typename... Ts>
  ArgumentsRef buildArguments(const Ts &...ts) const;

protected:
  DeviceFunction(const DeviceContext &context,
                 const DeviceFunction::SignatureBuilder &signature);

  virtual Arguments *allocateArguments(const DeviceFunction &fn) const = 0;

private:
  DeviceContextRef context_;
  std::vector<ArgumentType> signature_;

  ErrorOrSuccess checkSignatureAt(int index, ArgumentType argType) const;
  static const char *typeName(ArgumentType argType);
};

class DeviceFunction::SignatureBuilder {
public:
  SignatureBuilder() = default;
  ~SignatureBuilder() = default;

  SignatureBuilder &addUInt64();
  SignatureBuilder &addBuffer();

private:
  friend class DeviceFunction;

  std::vector<ArgumentType> types_;
};

class DeviceFunction::Arguments
    : public ReferenceCounted<DeviceFunction::Arguments> {
public:
  virtual ~Arguments() = default;

  bool hasError() const;
  const ErrorOrSuccess &getError() const;

protected:
  Arguments(const DeviceFunction &func);

  // The implementations of these methods are being called based on the
  // arguments being passed to the DeviceFunction::buildArguments() method.
  // TODO: Add argument types incrementally as needed.
  virtual void pushArg(int index, const uint64_t &val) = 0;
  virtual void pushArg(int index, const DeviceBuffer &val) = 0;

private:
  friend class DeviceFunction;

  DeviceFunctionRef fn_;
  ErrorOrSuccess err_;

  template <typename... Ts>
  void build(const Ts &...ts);

  template <int index, typename T0, typename... T1s>
  void pushArgs(const T0 &&t0, const T1s &&...t1s);

  template <int index, typename T0>
  void pushArgs(const T0 &&t0);

  // TODO: Add argument types incrementally as needed.
  void checkArg(int index, const uint64_t &val);
  void checkArg(int index, const DeviceBuffer &val);
};

class DeviceTimer : public ReferenceCounted<DeviceTimer> {
public:
  virtual ~DeviceTimer() = default;

  const DeviceContext &context() const;

protected:
  DeviceTimer(const DeviceContext &context);

private:
  DeviceContextRef context_;
};

class DeviceContext : public ReferenceCounted<DeviceContext> {
public:
  struct MemInfo {
    MemInfo() : total(0), free(0) {}
    size_t total;
    size_t free;
  };

  enum class AllocMode : uint8_t {
    HOST = 1,
    SYNC = 2,
    ASYNC = 3,
  };

  virtual ~DeviceContext();

  const Device &device() const;

  /// Returns true if this device can access the `peer` device.
  ErrorOr<bool> canAccess(const DeviceContext &peer) const {
    return device().canAccess(peer.device());
  };

  // DeviceBuffer allocation.
  ErrorOr<DeviceBufferRef> enqueueCreateBuffer(size_t byteSize) const;
  ErrorOr<DeviceBufferRef> createBuffer(size_t byteSize) const;

  // Host memory allocation.
  ErrorOr<DeviceBufferRef> createHostBuffer(size_t byteSize) const;

  // TODO: Remove the V1 compatibility.
  DeviceBufferRef createBufferOwning(void *ptr, size_t byteSize,
                                     bool owning) const;
  DeviceBufferRef createHostBufferOwning(void *ptr, size_t byteSize,
                                         bool owning) const;

  // DeviceBuffer copies.
  virtual ErrorOrSuccess enqueueCopyToDevice(const DeviceBuffer &dst,
                                             const void *src) const = 0;

  virtual ErrorOrSuccess
  enqueueCopyFromDevice(void *dst, const DeviceBuffer &src) const = 0;

  ErrorOrSuccess enqueueCopyDeviceToDevice(const DeviceBuffer &dst,
                                           const DeviceBuffer &src) const;

  virtual ErrorOrSuccess enqueueSetMemory(const DeviceBuffer &dst, uint64_t val,
                                          size_t valSize) const = 0;

  // DeviceFunction compiling.
  virtual ErrorOr<M::BufferRef>
  compileFunction(StringRef ir, StringRef debugLevel,
                  int32_t optimizationLevel) const;

  // DeviceFunction loading and execution.
  virtual ErrorOr<DeviceFunctionRef>
  loadFunction(StringRef moduleName, StringRef functionName,
               const DeviceFunction::SignatureBuilder &signature,
               StringRef code, int32_t maxDynamicSharedBytes = -1,
               StringRef debugLevel = "",
               int32_t optimizationLevel = -1) const = 0;
  virtual ErrorOrSuccess
  enqueueFunctionExec(const DeviceFunction &func, uint32_t gridX,
                      uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                      uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
                      const DeviceFunction::Arguments &args) const = 0;
  virtual ErrorOrSuccess
  enqueueFunctionExecDirect(const DeviceFunction &func, uint32_t gridX,
                            uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                            uint32_t blockY, uint32_t blockZ,
                            uint32_t sharedMemBytes, void *attrs,
                            uint32_t numAttrs, void **args) const = 0;

  // Synchronize asynchronous operations.
  virtual ErrorOr<DeviceStreamRef> stream() const = 0;

  virtual ErrorOrSuccess synchronize() const = 0;

  virtual ErrorOr<DeviceEventRef> enqueueEvent() const = 0;
  virtual ErrorOrSuccess enqueueWaitFor(const DeviceEvent &event) const = 0;
  virtual ErrorOrSuccess enqueueWaitFor(const DeviceContext &ctx) const = 0;
  ErrorOrSuccess enqueueWaitForEvent(const DeviceEvent &event) const {
    return enqueueWaitFor(event);
  }

  virtual ErrorOr<DeviceTimerRef> startTimer() const = 0;
  virtual ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) const = 0;

  virtual ErrorOrSuccess enablePeerAccess(const DeviceContext &peer) const = 0;

  virtual ErrorOr<MemInfo> getMemoryInfo() const = 0;

  BufferCache &bufferCache() const;

  static ErrorOr<DeviceContextRef>
  create(StringRef api, int id = 0,
         // By default allow upto 1TB of bufferCacheSize.
         size_t bufferCacheSize = 1024ULL * 1024ULL * 1024ULL * 1024ULL);

  static ErrorOr<DeviceContextRef>
  getOrCreate(StringRef api, int id = 0,
              // By default allow upto 1TB of bufferCacheSize.
              size_t bufferCacheSize = 1024ULL * 1024ULL * 1024ULL * 1024ULL);

  // Access configuration flags from the "device-context" section.
  static bool configAsBool(StringRef key, bool defaultValue);
  static int64_t configAsInt(StringRef key, int64_t defaultValue);
  static StringRef configAsString(StringRef key, StringRef defaultValue);

  // Error helpers.
  static Error formatError(const char *file, int line, const char *func,
                           const llvm::Twine &msg);

protected:
  DeviceContext(const Device &device, std::unique_ptr<BufferCache> bufferCache);

private:
  friend class BufferCache;
  friend class DeviceBuffer;

  DeviceRef device_;

  std::unique_ptr<BufferCache> bufferCache_;

  // Explicitly mark copy and assignment as unsupported.
  DeviceContext(const DeviceContext &other) = delete;
  DeviceContext(DeviceContext &&other) = delete;
  DeviceContext &operator=(const DeviceContext &) = delete;
  DeviceContext &operator=(DeviceContext &&other) = delete;

  // DeviceBuffer allocation.
  virtual ErrorOr<DeviceBufferRef> allocBuffer(size_t byteSize,
                                               AllocMode mode) const;

  // Low-level uncached allocation of memory.
  // REQUIRES: byteSize > 0
  virtual ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                                  AllocMode mode) const = 0;
  virtual ErrorOrSuccess memFree(void *ptr, bool isHost) const = 0;

  // Low-level copies.
  virtual ErrorOrSuccess memCopyDtoD(const DeviceBuffer &dst,
                                     const DeviceBuffer &src) const = 0;

  // Shared configuration object.
  // REQUIRES: `DeviceContext::mu_` must be held to get and access the shared
  // Config object.
  static ErrorOr<Config> &getConfig();
};

// TODO: Delete any that seem too CUDA-specific
enum class Device::Attribute : int {
  // Maximum number of threads per block
  MAX_THREADS_PER_BLOCK = 1,
  // Maximum block dimension X
  MAX_BLOCK_DIM_X = 2,
  // Maximum block dimension Y
  MAX_BLOCK_DIM_Y = 3,
  // Maximum block dimension Z
  MAX_BLOCK_DIM_Z = 4,
  // Maximum grid dimension X
  MAX_GRID_DIM_X = 5,
  // Maximum grid dimension Y
  MAX_GRID_DIM_Y = 6,
  // Maximum grid dimension Z
  MAX_GRID_DIM_Z = 7,
  // Maximum shared memory available per block in bytes
  MAX_SHARED_MEMORY_PER_BLOCK = 8,
  // Deprecated, use alias MAX_SHARED_MEMORY_PER_BLOCK
  SHARED_MEMORY_PER_BLOCK = 8,
  // Memory available on device for __constant__ variables in a CUDA C kernel
  // in bytes
  TOTAL_CONSTANT_MEMORY = 9,
  // Warp size in threads
  WARP_SIZE = 10,
  // Maximum pitch in bytes allowed by memory copies
  MAX_PITCH = 11,
  // Maximum number of 32-bit registers available per block
  MAX_REGISTERS_PER_BLOCK = 12,
  // Deprecated, use alias MAX_REGISTERS_PER_BLOCK
  REGISTERS_PER_BLOCK = 12,
  // Typical clock frequency in kilohertz
  CLOCK_RATE = 13,
  // Alignment requirement for textures
  TEXTURE_ALIGNMENT = 14,
  // Device can possibly copy memory and execute a kernel concurrently.
  // Deprecated. Use instead alias ASYNC_ENGINE_COUNT.)
  GPU_OVERLAP = 15,
  // Number of multiprocessors on device
  MULTIPROCESSOR_COUNT = 16,
  // Specifies whether there is a run time limit on kernels
  KERNEL_EXEC_TIMEOUT = 17,
  // Device is integrated with host memory
  INTEGRATED = 18,
  // Device can map host memory into CUDA address space
  CAN_MAP_HOST_MEMORY = 19,
  // Compute mode (See ::CUcomputemode for details))
  COMPUTE_MODE = 20,
  // Maximum 1D texture width
  MAXIMUM_TEXTURE1D_WIDTH = 21,
  // Maximum 2D texture width
  MAXIMUM_TEXTURE2D_WIDTH = 22,
  // Maximum 2D texture height
  MAXIMUM_TEXTURE2D_HEIGHT = 23,
  // Maximum 3D texture width
  MAXIMUM_TEXTURE3D_WIDTH = 24,
  // Maximum 3D texture height
  MAXIMUM_TEXTURE3D_HEIGHT = 25,
  // Maximum 3D texture depth
  MAXIMUM_TEXTURE3D_DEPTH = 26,
  // Maximum 2D layered texture width
  MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
  // Maximum 2D layered texture height
  MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
  // Maximum layers in a 2D layered texture
  MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
  // Deprecated, use alias MAXIMUM_TEXTURE2D_LAYERED_WIDTH
  MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,
  // Deprecated, use alias MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
  MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,
  // Deprecated, use alias MAXIMUM_TEXTURE2D_LAYERED_LAYERS
  MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29,
  // Alignment requirement for surfaces
  SURFACE_ALIGNMENT = 30,
  // Device can possibly execute multiple kernels concurrently
  CONCURRENT_KERNELS = 31,
  // Device has ECC support enabled
  ECC_ENABLED = 32,
  // PCI bus ID of the device
  PCI_BUS_ID = 33,
  // PCI device ID of the device
  PCI_DEVICE_ID = 34,
  // Device is using TCC driver model
  TCC_DRIVER = 35,
  // Peak memory clock frequency in kilohertz
  MEMORY_CLOCK_RATE = 36,
  // Global memory bus width in bits
  GLOBAL_MEMORY_BUS_WIDTH = 37,
  // Size of L2 cache in bytes
  L2_CACHE_SIZE = 38,
  // Maximum resident threads per multiprocessor
  MAX_THREADS_PER_MULTIPROCESSOR = 39,
  // Number of asynchronous engines
  ASYNC_ENGINE_COUNT = 40,
  // Device shares a unified address space with the host
  UNIFIED_ADDRESSING = 41,
  // Maximum 1D layered texture width
  MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
  // Maximum layers in a 1D layered texture
  MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
  // Deprecated, do not use.)
  CAN_TEX2D_GATHER = 44,
  // Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set
  MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
  // Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set
  MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
  // Alternate maximum 3D texture width
  MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
  // Alternate maximum 3D texture height
  MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
  // Alternate maximum 3D texture depth
  MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
  // PCI domain ID of the device
  PCI_DOMAIN_ID = 50,
  // Pitch alignment requirement for textures
  TEXTURE_PITCH_ALIGNMENT = 51,
  // Maximum cubemap texture width/height
  MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
  // Maximum cubemap layered texture width/height
  MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
  // Maximum layers in a cubemap layered texture
  MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
  // Maximum 1D surface width
  MAXIMUM_SURFACE1D_WIDTH = 55,
  // Maximum 2D surface width
  MAXIMUM_SURFACE2D_WIDTH = 56,
  // Maximum 2D surface height
  MAXIMUM_SURFACE2D_HEIGHT = 57,
  // Maximum 3D surface width
  MAXIMUM_SURFACE3D_WIDTH = 58,
  // Maximum 3D surface height
  MAXIMUM_SURFACE3D_HEIGHT = 59,
  // Maximum 3D surface depth
  MAXIMUM_SURFACE3D_DEPTH = 60,
  // Maximum 1D layered surface width
  MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
  // Maximum layers in a 1D layered surface
  MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
  // Maximum 2D layered surface width
  MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
  // Maximum 2D layered surface height
  MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
  // Maximum layers in a 2D layered surface
  MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
  // Maximum cubemap surface width
  MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
  // Maximum cubemap layered surface width
  MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
  // Maximum layers in a cubemap layered surface
  MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
  // Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or
  // cuDeviceGetTexture1DLinearMaxWidth() instead.)
  MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
  // Maximum 2D linear texture width
  MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
  // Maximum 2D linear texture height
  MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
  // Maximum 2D linear texture pitch in bytes
  MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
  // Maximum mipmapped 2D texture width
  MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
  // Maximum mipmapped 2D texture height
  MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
  // Major compute capability version number
  COMPUTE_CAPABILITY_MAJOR = 75,
  // Minor compute capability version number
  COMPUTE_CAPABILITY_MINOR = 76,
  // Maximum mipmapped 1D texture width
  MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
  // Device supports stream priorities
  STREAM_PRIORITIES_SUPPORTED = 78,
  // Device supports caching globals in L1
  GLOBAL_L1_CACHE_SUPPORTED = 79,
  // Device supports caching locals in L1
  LOCAL_L1_CACHE_SUPPORTED = 80,
  // Maximum shared memory available per multiprocessor in bytes
  MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
  // Maximum number of 32-bit registers available per multiprocessor
  MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
  // Device can allocate managed memory on this system
  MANAGED_MEMORY = 83,
  // Device is on a multi-GPU board
  MULTI_GPU_BOARD = 84,
  // Unique id for a group of devices on the same multi-GPU board
  MULTI_GPU_BOARD_GROUP_ID = 85,
  // Link between the device and the host supports native atomic operations
  // (this is a placeholder attribute, and is not supported on any current
  // hardware).
  HOST_NATIVE_ATOMIC_SUPPORTED = 86,
  // Ratio of single precision performance (in floating-point operations per
  // second) to double precision performance.
  SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
  // Device supports coherently accessing pageable memory without calling
  // cudaHostRegister on it.
  PAGEABLE_MEMORY_ACCESS = 88,
  // Device can coherently access managed memory concurrently with the CPU
  CONCURRENT_MANAGED_ACCESS = 89,
  // Device supports compute preemption.
  COMPUTE_PREEMPTION_SUPPORTED = 90,
  // Device can access host registered memory at the same virtual address as
  // the CPU
  CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
  // Deprecated, along with v1 MemOps API, ::cuStreamBatchMemOp and related
  // APIs are supported.
  CAN_USE_STREAM_MEM_OPS_V1 = 92,
  // Deprecated, along with v1 MemOps API, 64-bit operations are supported in
  // ::cuStreamBatchMemOp and related APIs.
  CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
  // Deprecated, along with v1 MemOps API, ::CU_STREAM_WAIT_VALUE_NOR is
  // supported.
  CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
  // Device supports launching cooperative kernels via
  // ::cuLaunchCooperativeKernel
  COOPERATIVE_LAUNCH = 95,
  // Deprecated, ::cuLaunchCooperativeKernelMultiDevice is deprecated.)
  COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
  // Maximum optin shared memory per block
  MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
  // The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the
  // ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device.
  // See \ref CUDA_MEMOP for additional details.
  CAN_FLUSH_REMOTE_WRITES = 98,
  // Device supports host memory registration via ::cudaHostRegister.
  HOST_REGISTER_SUPPORTED = 99,
  // Device accesses pageable memory via the host's page tables.
  PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
  // The host can directly access managed memory on the device without
  // migration.
  DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
  // Deprecated, Use alias VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
  VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
  // Device supports virtual memory management APIs like
  // ::cuMemAddressReserve, ::cuMemCreate, ::cuMemMap and related APIs
  VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,
  // Device supports exporting memory to a posix file descriptor with
  // ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
  HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
  // Device supports exporting memory to a Win32 NT handle with
  // ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
  HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
  // Device supports exporting memory to a Win32 KMT handle with
  // ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
  HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
  // Maximum number of blocks per multiprocessor
  MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
  // Device supports compression of memory
  GENERIC_COMPRESSION_SUPPORTED = 107,
  // Maximum L2 persisting lines capacity setting in bytes.
  MAX_PERSISTING_L2_CACHE_SIZE = 108,
  // Maximum value of CUaccessPolicyWindow::num_bytes.
  MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
  // Device supports specifying the GPUDirect RDMA flag with ::cuMemCreate
  GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
  // Shared memory reserved by CUDA driver per block in bytes
  RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
  // Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays
  SPARSE_CUDA_ARRAY_SUPPORTED = 112,
  // Device supports using the ::cuMemHostRegister flag
  // ::CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as
  // read-only to the GPU
  READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
  // External timeline semaphore interop is supported on the device
  TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
  // Device supports using the ::cuMemAllocAsync and ::cuMemPool family of
  // APIs
  MEMORY_POOLS_SUPPORTED = 115,
  // Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see
  // https://docs.nvidia.com/cuda/gpudirect-rdma for more information)
  GPU_DIRECT_RDMA_SUPPORTED = 116,
  // The returned attribute shall be interpreted as a bitmask, where the
  // individual bits are described by the ::CUflushGPUDirectRDMAWritesOptions
  // enum
  GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
  // GPUDirect RDMA writes to the device do not need to be flushed for
  // consumers within the scope indicated by the returned attribute. See
  // ::CUGPUDirectRDMAWritesOrdering for the numerical values returned here.
  GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
  // Handle types supported with mempool based IPC
  MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
  // Indicates device supports cluster launch
  CLUSTER_LAUNCH = 120,
  // Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays
  DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
  // 64-bit operations are supported in ::cuStreamBatchMemOp and related MemOp
  // APIs.
  CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
  // ::CU_STREAM_WAIT_VALUE_NOR is supported by MemOp APIs.
  CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
  // Device supports buffer sharing with dma_buf mechanism.
  DMA_BUF_SUPPORTED = 124,
  // Device supports IPC Events.
  IPC_EVENT_SUPPORTED = 125,
  // Number of memory domains the device supports.
  MEM_SYNC_DOMAIN_COUNT = 126,
  // Device supports accessing memory using Tensor Map.
  TENSOR_MAP_ACCESS_SUPPORTED = 127,
  // Device supports unified function pointers.
  UNIFIED_FUNCTION_POINTERS = 129,
  // Device supports switch multicast and reduction operations.
  MULTICAST_SUPPORTED = 132,
};

//
// Internal implementation details follow.
//

class DeviceFunctionCacheEntry
    : public ReferenceCounted<DeviceFunctionCacheEntry> {
public:
  DeviceFunctionCacheEntry(const std::string name);

  const std::string &name() const;

private:
  const std::string name_;
};

//
// Private template definitions used by external code.
//
template <typename... Ts>
DeviceFunction::ArgumentsRef
DeviceFunction::buildArguments(const Ts &...ts) const {
  Arguments *args = allocateArguments(*this);
  args->build(std::forward<const Ts>(ts)...);
  return ArgumentsRef::take(args);
}

template <typename... Ts>
void DeviceFunction::Arguments::build(const Ts &...ts) {
  pushArgs<0>(std::forward<const Ts>(ts)...);
}

template <int index, typename T0, typename... T1s>
void DeviceFunction::Arguments::pushArgs(const T0 &&t0, const T1s &&...t1s) {
  checkArg(index, t0);
  pushArgs<index + 1, const T1s...>(std::forward<const T1s>(t1s)...);
}

template <int index, typename T0>
void DeviceFunction::Arguments::pushArgs(const T0 &&t0) {
  checkArg(index, t0);
}

} // namespace M::AsyncRT

#endif // ASYNCRT_DEVICECONTEXT_DEVICECONTEXT_H
