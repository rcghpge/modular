//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the M::AsyncRT::DeviceContext interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_DEVICECONTEXT_DEVICECONTEXT_H
#define MLRT_DRIVER_DEVICECONTEXT_DEVICECONTEXT_H

#include "Support/Buffer.h"
#include "Support/Configuration.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"

#include "llvm/ADT/DenseMap.h"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <stddef.h>
#include <string>
#include <vector>

namespace M::AsyncRT {

class BufferCache;
class BufferCacheFreeList;
class CompilationDevice;
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

// Virtual device mode control functions.
// When virtual device mode is enabled (count > 0), Device::create() will return
// VirtualDevice instances instead of real hardware devices, and
// Device::numberOfDevices() will return the virtual count. This allows creating
// devices for GPU configurations that don't match the current hardware.
// These functions are thread-safe and the count applies globally to all
// threads.
void setVirtualDeviceCount(int count);
int getVirtualDeviceCount();
bool isVirtualDeviceMode();

// Virtual device API control functions.
// These functions set and get the API (e.g., "cuda", "hip", "metal") for
// virtual devices created in compile-only mode. The API must be set before
// creating virtual devices. Thread-safe.
void setVirtualDeviceAPI(StringRef api);
std::string getVirtualDeviceAPI();

// Virtual device target architecture control functions.
// These functions set and get the target GPU architecture (e.g., "sm_80",
// "sm_90") for virtual devices created in compile-only mode. The architecture
// must be set before creating virtual devices. Thread-safe.
void setVirtualDeviceTargetArch(StringRef arch);
std::string getVirtualDeviceTargetArch();

class DriverVersion {
public:
  explicit DriverVersion(std::string version);
  int major() const;
  int minor() const;
  int patch() const;
  std::string toString() const;

private:
  std::string version_;
  int major_;
  int minor_;
  int patch_;
};

class DeviceContextScope {
public:
  DeviceContextScope(const DeviceContext &ctx);
  ~DeviceContextScope();

  ErrorOrSuccess err() const;

private:
  DeviceRef device_;
  void *prevCtx_;
  ErrorOrSuccess err_;
};

class Device : public ReferenceCounted<Device> {
public:
  enum class Attribute;

  // API is a constexpr StringLiteral for fast comparison, while being
  // human-readable at the same time.
  using API = StringLiteral;
  static constexpr API cpuAPI = "cpu";     // Generic CPU device
  static constexpr API gpuAPI = "gpu";     // Generic GPU device
  static constexpr API cudaAPI = "cuda";   // NVIDIA CUDA device
  static constexpr API hipAPI = "hip";     // AMDGPU HIP device
  static constexpr API metalAPI = "metal"; // Apple Metal device

  // AllocMode specifies the kind of memory requested during allocation and
  // free and is used to implement internal APIs.
  enum class AllocMode : uint8_t {
    HOST = 1,
    SYNC = 2,
    ASYNC = 3,
  };

  struct MemInfo {
    MemInfo() : total(0), free(0) {}
    size_t total;
    size_t free;
  };

  virtual ~Device();

  const std::string &name() const;

  virtual ErrorOr<const char *> getArchName() const;

  int id() const;

  API api() const;

  virtual ErrorOr<int> getApiVersion() const = 0;
  virtual ErrorOr<DriverVersion &> getDriverVersion() const = 0;
  virtual ErrorOrSuccess isCompatible() const = 0;
  virtual ErrorOr<int> computeCapability() const = 0;

  size_t allocationAlignment() const;

  virtual ErrorOr<bool> canAccess(const Device &peer) const = 0;

  // Enable peer access between devices.
  virtual ErrorOrSuccess enablePeerAccess(const Device &peer) const;

  virtual ErrorOr<int> getAttribute(Attribute attr) const = 0;

  virtual ErrorOr<bool> supportsMulticast() const;

  virtual int deviceCount() const = 0;

  BufferCache &bufferCache() const;
  BufferCache &hostBufferCache() const;

  // Low-level uncached synchronous allocation of memory to support the
  // BufferCache.
  // REQUIRES: byteSize > 0
  virtual ErrorOrSuccess chunkAlloc(void **ptr, size_t byteSize,
                                    bool isHost) const = 0;
  virtual ErrorOrSuccess chunkFree(void *ptr, bool isHost) const = 0;

  virtual ErrorOr<Device::MemInfo> getMemoryInfo() const = 0;

  virtual ErrorOr<DeviceContextRef> createContext() const = 0;

  virtual ErrorOr<DeviceStreamRef>
  createExternalStream(void *externalStream) const = 0;

  /// Get the compilation device for this device, or nullptr if no
  /// compilation device is available.
  virtual const CompilationDevice *compilationDevice() const = 0;

  // Device function compilation.
  virtual ErrorOr<M::BufferRef>
  compileFunction(StringRef ir, StringRef debugLevel, int32_t optimizationLevel,
                  StringRef targetAccelerator) const;

  static int numberOfDevices(StringRef api);
  static size_t preferredAlignment(StringRef api);

  static ErrorOr<DeviceRef> create(StringRef api, int id = 0,
                                   int64_t deviceMemoryLimit = -1);

  /// Returns the vendor and compute_capability of the accelerator in the form
  /// of `vendor:compute_capability`. For example, an A100 GPU would return
  /// `nvidia:80`.
  static const std::string &getAcceleratorArchOrEmpty();

  /// Function cache key for the function cache. This accounts for how the
  /// function is compiled including debug level and other properties.
  struct FunctionCacheKey {
    std::string name;
    std::string debugLevel = "";
    int32_t maxDynamicSharedBytes = -1;
    int32_t optimizationLevel = -1;

    bool operator==(const FunctionCacheKey &other) const {
      return name == other.name && debugLevel == other.debugLevel &&
             maxDynamicSharedBytes == other.maxDynamicSharedBytes &&
             optimizationLevel == other.optimizationLevel;
    }
  };

  // Internal implementation helpers for DeviceContexts.
  DeviceFunctionCacheEntryRef
  findInFunctionCache(const FunctionCacheKey &key) const;
  DeviceFunctionCacheEntryRef
  insertInFunctionCache(const FunctionCacheKey &key,
                        const DeviceFunctionCacheEntry &entry) const;
  void clearFunctionCache() const;

  virtual ErrorOr<void *> getCurrentDriverContext() const = 0;
  virtual ErrorOrSuccess setCurrentDriverContext(void *ctx) const = 0;

protected:
  Device(API api, std::string name, int id, MemInfo memInfo, size_t alignment,
         int64_t deviceMemoryLimit = -1);

private:
  std::string name_;
  int id_;
  API api_;

  std::unique_ptr<BufferCache> bufferCache_;
  std::unique_ptr<BufferCache> hostBufferCache_;

  // Mutex to protect mutable data.
  mutable std::mutex mu_;
  // The function cache holds the lower-level function references which do not
  // contain a device context reference to avoid reference counting cycles.
  mutable DenseMap<FunctionCacheKey, DeviceFunctionCacheEntryRef> funcMap_;
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

  virtual ErrorOr<bool> isReady() const = 0;

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
  enqueueFunctionExecDirect(const DeviceFunction &func, uint32_t gridX,
                            uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                            uint32_t blockY, uint32_t blockZ,
                            uint32_t sharedMemBytes, void *attrs,
                            uint32_t numAttrs, void **args) const = 0;
  virtual ErrorOrSuccess eventRecord(const DeviceEvent &event) const = 0;
  virtual ErrorOrSuccess waitEvent(const DeviceEvent &event,
                                   unsigned int flags) const = 0;

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

  virtual ~DeviceFunction() = default;

  const DeviceContext &context() const;

  virtual const std::string &name() const = 0;

  virtual ErrorOrSuccess copyToNamedConstant(const char *name, const void *data,
                                             size_t byteSize) const = 0;

  virtual ErrorOr<int32_t> getAttribute(int32_t code) const = 0;

  size_t argumentCount() const;

protected:
  DeviceFunction(const DeviceContext &context,
                 const DeviceFunction::SignatureBuilder &signature);

private:
  DeviceContextRef context_;
  std::vector<ArgumentType> signature_;
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
  virtual ~DeviceContext();

  const Device &device() const;

  /// Returns true if this device can access the `peer` device.
  ErrorOr<bool> canAccess(const DeviceContext &peer) const {
    return device().canAccess(peer.device());
  }

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

  virtual ErrorOrSuccess enqueueCopyToDeviceSized(void *dst, void *src,
                                                  size_t byteSize) const = 0;

  virtual ErrorOrSuccess enqueueCopyFromDeviceSized(void *dst, void *src,
                                                    size_t byteSize) const = 0;

  ErrorOrSuccess enqueueCopyDeviceToDevice(const DeviceBuffer &dst,
                                           const DeviceBuffer &src) const;

  virtual ErrorOrSuccess enqueueSetMemory(const DeviceBuffer &dst, uint64_t val,
                                          size_t valSize) const = 0;

  ErrorOr<std::pair<AsyncRT::DeviceBufferRef,
                    std::optional<llvm::sys::fs::mapped_file_region>>>
  memmapFile(const std::filesystem::path &filename) const;

  // DeviceFunction loading and execution.
  virtual ErrorOr<DeviceFunctionRef>
  loadFunction(StringRef moduleName, StringRef functionName,
               const DeviceFunction::SignatureBuilder &signature,
               StringRef code, int32_t maxDynamicSharedBytes = -1,
               StringRef debugLevel = "",
               int32_t optimizationLevel = -1) const = 0;
  virtual ErrorOrSuccess enqueueFunctionExecDirect(
      const DeviceFunction &func, uint32_t gridX, uint32_t gridY,
      uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
      uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args,
      const size_t *argSizes = nullptr) const = 0;

  // Synchronize asynchronous operations.
  virtual ErrorOr<DeviceStreamRef> stream() const = 0;
  virtual ErrorOr<DeviceStreamRef> streamCreate(unsigned int flags) const = 0;
  virtual ErrorOr<DeviceStreamRef>
  streamCreateWithPriority(unsigned int flags, int priority) const = 0;
  virtual ErrorOrSuccess streamPriorityRange(int *leastPriority,
                                             int *greatestPriority) const = 0;

  virtual ErrorOrSuccess synchronize() const = 0;

  virtual ErrorOr<DeviceEventRef> enqueueEvent() const = 0;
  virtual ErrorOr<DeviceEventRef> eventCreate(unsigned int flags) const = 0;
  virtual ErrorOrSuccess enqueueWaitFor(const DeviceEvent &event) const = 0;
  virtual ErrorOrSuccess enqueueWaitFor(const DeviceContext &ctx) const = 0;
  ErrorOrSuccess enqueueWaitForEvent(const DeviceEvent &event) const {
    return enqueueWaitFor(event);
  }

  ErrorOrSuccess enqueueEventHandler(std::function<void()> handler) const;

  virtual ErrorOr<DeviceTimerRef> startTimer() const = 0;
  virtual ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) const = 0;

  virtual ErrorOrSuccess enablePeerAccess(const DeviceContext &peer) const = 0;

  ErrorOr<Device::MemInfo> getMemoryInfo() const;

  virtual ErrorOrSuccess occupancyMaxActiveBlocksPerMultiprocessor(
      int *numBlocks, const DeviceFunction &func, int blockSize,
      size_t dynamicSharedMemSize) const = 0;

  // Set this device context as the current driver context
  ErrorOrSuccess setAsCurrent() const;

  BufferCacheFreeList &freeList() const;
  BufferCacheFreeList &hostFreeList() const;

  static ErrorOr<DeviceContextRef> create(StringRef api, int id = 0);

  static ErrorOr<DeviceContextRef> getOrCreate(StringRef api, int id = 0,
                                               int64_t deviceMemoryLimit = -1);

  bool operator==(const DeviceContext &other) const;
  bool operator!=(const DeviceContext &other) const;

  // Access configuration flags from the "device-context" section.
  static bool configAsBool(StringRef key, bool defaultValue,
                           bool *isDefault = nullptr);
  static int64_t configAsInt(StringRef key, int64_t defaultValue,
                             bool *isDefault = nullptr);
  static StringRef configAsString(StringRef key, StringRef defaultValue,
                                  bool *isDefault = nullptr);

  // Error helpers.
  static Error formatError(const char *file, int line, const char *func,
                           const llvm::Twine &msg);

  ErrorOrSuccess checkPendingWork() const;

  // Debugging helpers.
  bool syncMode() const;
  void incSyncMode() const;
  void decSyncMode() const;

protected:
  DeviceContext(const Device &device);

  // Device-specific memory information (raw hardware memory)
  virtual ErrorOr<Device::MemInfo> getDeviceMemoryInfo() const = 0;

  // Low-level uncached allocation of memory.
  // REQUIRES: byteSize > 0
  virtual ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                                  Device::AllocMode mode) const = 0;
  virtual ErrorOrSuccess memFree(void *ptr, bool isHost) const = 0;

  // Low-level copies.
  virtual ErrorOrSuccess memCopyDtoD(const DeviceBuffer &dst,
                                     const DeviceBuffer &src) const = 0;

  using MMapFileImplResult =
      ErrorOr<std::tuple<void *, size_t,
                         std::optional<llvm::sys::fs::mapped_file_region>>>;

  virtual MMapFileImplResult
  memmapFileImpl(const std::filesystem::path &filename) const = 0;

  virtual void *driverContext() const = 0;

  void setPendingError(Error err) const;

private:
  friend class BufferCacheFreeList;
  friend class DeviceBuffer;
  friend class DeviceContextScope;
  // Following line is equivalent to:
  // FRIEND_TEST(GPUDeviceContextTest, DeviceContextScope)
  friend class GPUDeviceContextTest_DeviceContextScope_Test;

  class PendingEventHandler;

  DeviceRef device_;

  std::unique_ptr<BufferCacheFreeList> freeList_;
  std::unique_ptr<BufferCacheFreeList> hostFreeList_;

  mutable std::atomic<int> syncMode_ = {0};

  // hasPendingWork_ does indicate whether pendingError_ and pendingEvents_
  // should be checked. Since pending[Error_ | Events_] are protected by a lock
  // we set an atomic value that can be quickly checked without fully acquiring
  // a mutex lock. Raising errors and enqueueing work across threads is
  // inherently racing, so we are fine with possibly missing an update to
  // hasPendingWork_ from a different thread.
  //
  // To make sure the pendingWork_ is properly set and memory been made visible
  // to other threads, we first need to lock mu_, update the pendingError_ or
  // pendingEvents_, release mu_ and only then atomically set pendingWork_.
  mutable std::atomic<int> hasPendingWork_ = {0};
  mutable std::mutex mu_;
  mutable ErrorOrSuccess pendingError_;
  mutable PendingEventHandler *pendingEvents_ = nullptr;
  mutable PendingEventHandler *pendingEventsLast_ = nullptr;

  // Explicitly mark copy and assignment as unsupported.
  DeviceContext(const DeviceContext &other) = delete;
  DeviceContext(DeviceContext &&other) = delete;
  DeviceContext &operator=(const DeviceContext &) = delete;
  DeviceContext &operator=(DeviceContext &&other) = delete;

  // DeviceBuffer allocation.
  ErrorOr<DeviceBufferRef> allocBuffer(size_t byteSize,
                                       Device::AllocMode mode) const;

  void setPendingEventHandler(PendingEventHandler *h) const;

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
  // Warp size in threads
  WARP_SIZE = 10,
  // Maximum number of 32-bit registers available per block
  MAX_REGISTERS_PER_BLOCK = 12,
  // Typical clock frequency in kilohertz
  CLOCK_RATE = 13,
  // Number of multiprocessors on device
  MULTIPROCESSOR_COUNT = 16,
  // Maximum resident threads per multiprocessor
  MAX_THREADS_PER_MULTIPROCESSOR = 39,
  // Major compute capability version number
  COMPUTE_CAPABILITY_MAJOR = 75,
  // Minor compute capability version number
  COMPUTE_CAPABILITY_MINOR = 76,
  // Maximum shared memory available per multiprocessor in bytes
  MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
  // Maximum number of 32-bit registers available per multiprocessor
  MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
  // Device supports launching cooperative kernels
  COOPERATIVE_LAUNCH = 95,
  // Maximum resident blocks per multiprocessor
  MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
  // CUDA-only: Maximum value of CUaccessPolicyWindow::num_bytes.
  MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
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

} // namespace M::AsyncRT

namespace llvm {
template <>
struct DenseMapInfo<M::AsyncRT::Device::FunctionCacheKey> {
  static inline M::AsyncRT::Device::FunctionCacheKey getEmptyKey() {
    return {""};
  }
  static inline M::AsyncRT::Device::FunctionCacheKey getTombstoneKey() {
    return {""};
  }
  static unsigned
  getHashValue(const M::AsyncRT::Device::FunctionCacheKey &val) {
    return llvm::hash_combine(val.name, val.debugLevel,
                              val.maxDynamicSharedBytes, val.optimizationLevel);
  }
  static bool isEqual(const M::AsyncRT::Device::FunctionCacheKey &lhs,
                      const M::AsyncRT::Device::FunctionCacheKey &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // MLRT_DRIVER_DEVICECONTEXT_DEVICECONTEXT_H
