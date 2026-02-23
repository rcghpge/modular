//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the M::Driver::DeviceContext interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_DEVICECONTEXT_DEVICECONTEXT_H
#define MLRT_DRIVER_DEVICECONTEXT_DEVICECONTEXT_H

#include "ErrorMacros.h"
#include "MLRT/AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "Support/Buffer.h"
#include "Support/Configuration.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"

#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <stddef.h>
#include <string>
#include <vector>

namespace llvm::json {
class Value;
}

namespace M::Driver {

/// Metadata for a trace entry (kernel launch, memcpy, or memset).
struct KernelLaunchMetadata {
  /// Type of operation that produced this trace entry
  enum class OperationKind : uint8_t {
    KernelLaunch,
    Memcpy,
    Memset,
  };

  /// Memcpy direction/kind
  enum class MemcpyKind : uint8_t { None = 0, HtoD, DtoH, DtoD };

  /// The kind of operation this metadata represents
  OperationKind kind = OperationKind::KernelLaunch;

  /// The name of the operation (kernel name, "cuMemcpyHtoD", "cuMemset", etc.)
  std::string kernelName;

  /// The hash value recorded during tracing.
  uint64_t semanticHash = 0;

  // ===== Kernel launch fields (only set for kernel launches) =====
  uint32_t gridX = 0, gridY = 0, gridZ = 0;
  uint32_t blockX = 0, blockY = 0, blockZ = 0;
  uint32_t sharedMemBytes = 0;
  uint32_t numAttrs = 0;
  uint32_t streamIndex = 0;

  // ===== Memcpy fields (only set for memcpy operations) =====
  MemcpyKind memcpyKind = MemcpyKind::None;
  size_t memcpyByteSize = 0;

  // ===== Memset fields (only set for memset operations) =====
  size_t memsetByteSize = 0;
  uint64_t memsetValue = 0;
  size_t memsetValueSize = 0;

  /// Default constructor
  KernelLaunchMetadata() = default;

  /// Constructor for simple operation (just name and hash)
  KernelLaunchMetadata(StringRef name, uint64_t hashValue)
      : kind(OperationKind::KernelLaunch), kernelName(name.str()),
        semanticHash(hashValue) {}

  /// Constructor for kernel launch with full metadata
  KernelLaunchMetadata(StringRef name, uint64_t hashValue, uint32_t gridX,
                       uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                       uint32_t blockY, uint32_t blockZ,
                       uint32_t sharedMemBytes, uint32_t numAttrs,
                       uint32_t streamIndex)
      : kind(OperationKind::KernelLaunch), kernelName(name.str()),
        semanticHash(hashValue), gridX(gridX), gridY(gridY), gridZ(gridZ),
        blockX(blockX), blockY(blockY), blockZ(blockZ),
        sharedMemBytes(sharedMemBytes), numAttrs(numAttrs),
        streamIndex(streamIndex) {}

  /// Constructor for memcpy operation
  KernelLaunchMetadata(StringRef name, uint64_t hashValue, MemcpyKind memcpy,
                       size_t byteSize)
      : kind(OperationKind::Memcpy), kernelName(name.str()),
        semanticHash(hashValue), memcpyKind(memcpy), memcpyByteSize(byteSize) {}

  /// Constructor for memset operation
  KernelLaunchMetadata(StringRef name, uint64_t hashValue, size_t byteSize,
                       uint64_t value, size_t valSize)
      : kind(OperationKind::Memset), kernelName(name.str()),
        semanticHash(hashValue), memsetByteSize(byteSize), memsetValue(value),
        memsetValueSize(valSize) {}

  /// Comparison operator: sort by kernel name first, then by hash
  bool operator<(const KernelLaunchMetadata &other) const {
    if (kernelName != other.kernelName)
      return kernelName < other.kernelName;
    return semanticHash < other.semanticHash;
  }

  /// Equality operator for verification
  bool operator==(const KernelLaunchMetadata &other) const {
    return kernelName == other.kernelName && semanticHash == other.semanticHash;
  }

  bool operator!=(const KernelLaunchMetadata &other) const {
    return !(*this == other);
  }

  /// Format this trace entry as a human-readable string
  std::string format(int count = 1) const;

  /// Format this trace entry as JSON object
  llvm::json::Value formatJSON() const;
};

/// Result of comparing two launch traces.
struct LaunchTraceDiff {
  /// Operations in baseline but missing from current trace, with counts.
  std::map<KernelLaunchMetadata, int> missingOps;

  /// Operations in current trace but not in baseline, with counts.
  std::map<KernelLaunchMetadata, int> unexpectedOps;

  /// Returns true if the traces match (no differences).
  bool empty() const { return missingOps.empty() && unexpectedOps.empty(); }

  /// Format a detailed error message describing the differences.
  std::string formatErrorMessage(size_t graphIdx) const;

  /// Format as JSON object
  llvm::json::Value formatJSON(size_t graphIdx) const;
};

class MemoryManager;
class MemoryManagerFreeList;
class Device;
class DeviceBuffer;
class DeviceContext;
class DeviceEvent;
class DeviceFunction;
class DeviceGraph;
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
using DeviceGraphRef = RCRef<DeviceGraph>;
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
  explicit DeviceContextScope(const DeviceContext &ctx);
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

  virtual ErrorOr<std::string> getPciBusId() const = 0;

  virtual ErrorOr<bool> supportsMulticast() const;

  virtual int deviceCount() const = 0;

  MemoryManager &memoryManager() const;
  MemoryManager &hostMemoryManager() const;

  // Low-level uncached synchronous allocation of memory to support the
  // MemoryManager.
  // REQUIRES: byteSize > 0
  virtual ErrorOrSuccess chunkAlloc(void **ptr, size_t byteSize,
                                    bool isHost) const = 0;
  virtual ErrorOrSuccess chunkFree(void *ptr, bool isHost) const = 0;

  virtual ErrorOr<Device::MemInfo> getMemoryInfo() const = 0;

  virtual ErrorOr<DeviceContextRef> createContext() const = 0;

  static int numberOfDevices(StringRef api);
  static size_t preferredAlignment(StringRef api);

  static ErrorOr<DeviceRef> create(StringRef api, int id = 0);

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
  Device(API api, std::string name, int id, MemInfo memInfo, size_t alignment);

private:
  std::string name_;
  int id_;
  API api_;

  std::unique_ptr<MemoryManager> memoryManager_;
  std::unique_ptr<MemoryManager> hostMemoryManager_;

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
  const DeviceEvent *manualEvent() const;

  // Returns the raw device pointer, which is not safe to access on the host
  void *ptr() const;
  // TODO: Remove the V1 compatibility.
  void *releasePtr() const;

  size_t byteSize() const;

  bool isHost() const;

  inline ErrorOrSuccess copyTo(const DeviceBuffer &dst) const {
    return copyTo(dst, 0, dst.byteSize());
  }
  ErrorOrSuccess copyTo(const DeviceBuffer &dst, size_t srcByteOffset,
                        size_t srcByteSize) const;
  ErrorOrSuccess copyTo(void *dst) const;

  ErrorOrSuccess copyFrom(const void *src) const;

  ErrorOrSuccess reassignOwnershipTo(const DeviceContext &ctx) const;

  ErrorOr<DeviceBufferRef> createSubBuffer(size_t byteOffset,
                                           size_t byteSize) const;

  // Either do a blocking synchronize or setup events with an external stream.
  ErrorOrSuccess synchronize(void *external_stream = nullptr) const;

  ErrorOrSuccess disableAutoSync() const;
  ErrorOrSuccess markAsReady() const;

  void setDestructorFn(llvm::unique_function<void()> destructor) const;

private:
  friend class DeviceContext;

  mutable DeviceContextRef context_;
  mutable DeviceEventRef manualEvent_;
  DeviceBufferRef parent_;
  mutable void *ptr_;
  size_t byteSize_;
  bool isHost_;
  // TODO: Remove the V1 compatibility.
  mutable bool owning_ = true;
  mutable std::optional<llvm::unique_function<void()>> destructorFn_ =
      std::nullopt;

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
  explicit DeviceMulticastBuffer(size_t byteSize);

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
  explicit DeviceEvent(const Device &device);

private:
  DeviceRef device_;
};

/// A collection of device graphs generated during a recording session.
///
/// This class contains all the information needed to replay a multi-device
/// model. Each contained DeviceGraph corresponds to a captured graph for a
/// single execution stream.
///
/// The outputs correspond to the pinned memory allocated by one of the
/// DeviceGraphs during the record session and are used to ensure the backing
/// memory lives as long as the allocating DeviceGraph.
class DeviceGraphCollection {
public:
  explicit DeviceGraphCollection(SmallVector<DeviceGraphRef> &&graphs)
      : graphs(std::move(graphs)) {}

  /// Set the outputs for the device graph collection by taking ownership of
  /// the set of provided async references.
  void setOutputs(ArrayRef<AsyncRT::AnyAsyncValueRef> outputs);

  /// Get the output async values for the graph collection. These correspond
  /// to the outputs of the MEF model, not any particular DeviceGraph.
  SmallVector<AsyncRT::AnyAsyncValueRef> getOutputs() const;

  /// Replay the graph collection by launching all graphs.
  ErrorOrSuccess replay() const;

  /// Execute eagerly and compare launch trace against captured graphs.
  /// For each device graph, this runs the execute callback and verifies
  /// that the resulting trace matches the captured graph.
  ErrorOrSuccess
  debugVerifyReplay(llvm::function_ref<ErrorOrSuccess()> execute) const;

  /// The number of graphs in the collection.
  int64_t numGraphs() const;

private:
  /// A DeviceGraphCollection holds the set of captured device graphs which
  /// correspond to a single capture session of a MAX graph. There will be
  /// one graph per device context used by the graph.
  SmallVector<DeviceGraphRef> graphs;

  /// The set of outputs produced during a capture session of a MAX graph.
  /// The memory allocated for these outputs is stable and has a lifetime tied
  /// to that graph.
  ///
  /// Note: these outputs are the outputs of the entire MAX graph, not for any
  /// particular device.
  SmallVector<AsyncRT::AnyAsyncValueRef> outputs;
};

/// A captured sequence of GPU operations that can be replayed.
class DeviceGraph : public ReferenceCounted<DeviceGraph> {
public:
  virtual ~DeviceGraph() = default;
  const Device &device() const { return *device_; }

  /// Capture a graph and baseline launch trace by executing a callback.
  static ErrorOr<DeviceGraphCollection>
  capture(ArrayRef<DeviceStreamRef> streams,
          llvm::function_ref<ErrorOrSuccess()> execute);

  /// Replay the captured operations on the stored capture stream.
  virtual ErrorOrSuccess replay() const = 0;

  /// Set the stream associated with the device graph.
  void setStream(DeviceStreamRef stream);

  /// Set the launch trace recorded for the device graph.
  void setLaunchTrace(std::vector<KernelLaunchMetadata> baselineLaunchTrace);

  /// Get the baseline launch trace for this graph.
  ArrayRef<KernelLaunchMetadata> baselineLaunchTrace() const {
    return baselineLaunchTrace_;
  }

  /// Check if this graph has a baseline launch trace.
  bool hasBaselineLaunchTrace() const { return hasBaselineLaunchTrace_; }

  /// Get the capture stream for this graph.
  const DeviceStream *captureStream() const {
    return captureStream_.getPointer();
  }

  /// Get a reference to the capture stream for this graph.
  DeviceStreamRef captureStreamRef() const {
    return DeviceStreamRef::copy(captureStream_.getPointer());
  }

  /// Check if this graph has a capture stream.
  bool hasCaptureStream() const {
    return captureStream_.getPointer() != nullptr;
  }

protected:
  explicit DeviceGraph(const Device &device, DeviceStreamRef captureStream = {})
      : device_(DeviceRef::copy(&device)),
        captureStream_(std::move(captureStream)) {}

private:
  DeviceRef device_;
  DeviceStreamRef captureStream_;
  bool hasBaselineLaunchTrace_ = false;
  std::vector<KernelLaunchMetadata> baselineLaunchTrace_;
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

  /// Begin capturing operations on this stream.
  /// Operations enqueued after this call will be recorded into a graph.
  /// Returns error if already capturing or capture not supported.
  virtual ErrorOrSuccess beginCapture() const;

  /// End capture and return the captured graph.
  /// Returns error if not capturing or capture was invalidated.
  virtual ErrorOr<DeviceGraphRef> endCapture() const;

  /// Begin recording a device launch trace. Default is a no-op.
  virtual void beginLaunchTrace() const;

  /// Take and clear the current device launch trace. Default is empty.
  virtual std::vector<KernelLaunchMetadata> takeLaunchTrace() const;

protected:
  explicit DeviceStream(const DeviceContext &context);
  const DeviceContext &context_() const { return *ctx_; }

private:
  DeviceContextRef ctx_;
};

class DeviceFunction : public ReferenceCounted<DeviceFunction> {
public:
  virtual ~DeviceFunction() = default;

  const DeviceContext &context() const;

  virtual const std::string &name() const = 0;

  virtual ErrorOrSuccess copyToNamedConstant(const char *name, const void *data,
                                             size_t byteSize) const = 0;

  virtual ErrorOr<int32_t> getAttribute(int32_t code) const = 0;

  // Returns the number of arguments for this function. Default implementation
  // returns 0 (unknown). Device-specific implementations may override this.
  virtual size_t argumentCount() const;

protected:
  explicit DeviceFunction(const DeviceContext &context);

private:
  DeviceContextRef context_;
};

class DeviceTimer : public ReferenceCounted<DeviceTimer> {
public:
  virtual ~DeviceTimer() = default;

  const DeviceContext &context() const;

protected:
  explicit DeviceTimer(const DeviceContext &context);

private:
  DeviceContextRef context_;
};

class DeviceContext : public ReferenceCounted<DeviceContext> {
public:
  virtual ~DeviceContext();

  const Device &device() const;

  // DeviceBuffer allocation.
  ErrorOr<DeviceBufferRef> enqueueCreateBuffer(size_t byteSize) const;

  // Host memory allocation.
  ErrorOr<DeviceBufferRef> createHostBuffer(size_t byteSize) const;

  // TODO: Remove the V1 compatibility.
  DeviceBufferRef createBufferOwning(void *ptr, size_t byteSize,
                                     bool owning) const;
  DeviceBufferRef createHostBufferOwning(void *ptr, size_t byteSize,
                                         bool owning) const;

  // DeviceBuffer copies.
  virtual ErrorOrSuccess enqueueCopyToDevice(void *dst, const void *src,
                                             size_t byteSize) const = 0;
  inline ErrorOrSuccess enqueueCopyToDevice(const DeviceBuffer &dst,
                                            const void *src) const {
    if (dst.context().device().api() != device().api())
      RETURN_ERROR("API mismatch between buffer and context");
    return enqueueCopyToDevice(dst.ptr(), src, dst.byteSize());
  }

  virtual ErrorOrSuccess enqueueCopyFromDevice(void *dst, const void *src,
                                               size_t byteSize) const = 0;
  inline ErrorOrSuccess enqueueCopyFromDevice(void *dst,
                                              const DeviceBuffer &src) const {
    if (src.context().device().api() != device().api())
      RETURN_ERROR("API mismatch between buffer and context");
    return enqueueCopyFromDevice(dst, src.ptr(), src.byteSize());
  }

  inline ErrorOrSuccess
  enqueueCopyDeviceToDevice(const DeviceBuffer &dst,
                            const DeviceBuffer &src) const {
    return enqueueCopyDeviceToDevice(dst, src, 0, src.byteSize());
  }
  ErrorOrSuccess enqueueCopyDeviceToDevice(const DeviceBuffer &dst,
                                           const DeviceBuffer &src,
                                           size_t srcByteOffset,
                                           size_t srcByteSize) const;

  virtual ErrorOrSuccess enqueueSetMemory(void *dst, size_t byteSize,
                                          uint64_t val,
                                          size_t valSize) const = 0;
  inline ErrorOrSuccess enqueueSetMemory(const DeviceBuffer &dst, uint64_t val,
                                         size_t valSize) const {
    if (dst.context().device().api() != device().api())
      RETURN_ERROR("API mismatch between buffer and context");
    return enqueueSetMemory(dst.ptr(), dst.byteSize(), val, valSize);
  }

  ErrorOr<Driver::DeviceBufferRef>
  memmapFile(const std::filesystem::path &filename) const;

  // DeviceFunction loading and execution.
  virtual ErrorOr<DeviceFunctionRef>
  loadFunction(StringRef moduleName, StringRef functionName, StringRef code,
               int32_t maxDynamicSharedBytes = -1, StringRef debugLevel = "",
               int32_t optimizationLevel = -1) const = 0;
  virtual ErrorOrSuccess enqueueFunctionExecDirect(
      const DeviceFunction &func, uint32_t gridX, uint32_t gridY,
      uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
      uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args,
      const size_t *argSizes = nullptr) const = 0;

  // Synchronize asynchronous operations.
  virtual ErrorOr<DeviceStreamRef> stream() const = 0;
  virtual ErrorOrSuccess streamPriorityRange(int *leastPriority,
                                             int *greatestPriority) const = 0;
  virtual ErrorOr<DeviceStreamRef> createStream(int priority = 0) const = 0;

  virtual ErrorOr<DeviceStreamRef>
  createExternalStream(void *externalStream) const = 0;

  virtual ErrorOrSuccess synchronize() const = 0;

  virtual ErrorOr<DeviceEventRef> enqueueEvent() const = 0;
  virtual ErrorOrSuccess enqueueEvent(const DeviceEvent &event) const = 0;
  virtual ErrorOr<DeviceEventRef> eventCreate(unsigned int flags) const = 0;
  virtual ErrorOrSuccess enqueueWaitFor(const DeviceEvent &event) const = 0;
  virtual ErrorOrSuccess enqueueWaitFor(const DeviceContext &ctx) const = 0;
  ErrorOrSuccess enqueueWaitForEvent(const DeviceEvent &event) const {
    return enqueueWaitFor(event);
  }

  virtual ErrorOr<DeviceTimerRef> startTimer() const = 0;
  virtual ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) const = 0;

  ErrorOr<Device::MemInfo> getMemoryInfo() const;

  virtual ErrorOrSuccess occupancyMaxActiveBlocksPerMultiprocessor(
      int *numBlocks, const DeviceFunction &func, int blockSize,
      size_t dynamicSharedMemSize) const = 0;

  ErrorOrSuccess setAsCurrent() const;

  // Returns true if the device context is bound to the current thread.
  bool isBoundToCurrentThread() const;

  // Binds the device context to the current thread, or no-op if already bound.
  ErrorOrSuccess bindToCurrentThread() const;

  // Clears the bound context for the current thread.
  static void unbindFromCurrentThread();

  MemoryManagerFreeList &freeList() const;
  MemoryManagerFreeList &hostFreeList() const;

  static ErrorOr<DeviceContextRef> create(StringRef api, int id = 0);

  static ErrorOr<DeviceContextRef> getOrCreate(StringRef api, int id = 0);

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
  ErrorOrSuccess checkPendingError() const;

  // Enqueue event handlers.
  ErrorOrSuccess enqueueEventHandler(std::function<void()> handler) const;
  // Poll event handlers when events are ready.
  ErrorOrSuccess pollPendingEvents() const;

  // Debugging helpers.
  bool syncMode() const;
  void incSyncMode() const;
  void decSyncMode() const;
  bool isDebugAllocatorEnabled(StringRef name) const;

protected:
  explicit DeviceContext(const Device &device);

  // Device-specific memory information (raw hardware memory)
  virtual ErrorOr<Device::MemInfo> getDeviceMemoryInfo() const = 0;

  // Low-level uncached allocation of memory.
  // REQUIRES: byteSize > 0
  virtual ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                                  Device::AllocMode mode) const = 0;
  virtual ErrorOrSuccess memFree(void *ptr, bool isHost) const = 0;

  // Low-level copies.
  virtual ErrorOrSuccess memCopyDtoD(const DeviceBuffer &dst,
                                     const DeviceBuffer &src,
                                     size_t srcByteOffset,
                                     size_t srcByteSize) const = 0;

  using MMapFileImplResult =
      ErrorOr<std::tuple<void *, size_t,
                         std::optional<llvm::sys::fs::mapped_file_region>>>;

  virtual MMapFileImplResult
  memmapFileImpl(const std::filesystem::path &filename) const = 0;

  virtual void *driverContext() const = 0;

  void setPendingError(Error err) const;

private:
  friend class MemoryManagerFreeList;
  friend class DeviceBuffer;
  friend class DeviceContextScope;
  // Following line is equivalent to:
  // FRIEND_TEST(GPUDeviceContextTest, DeviceContextScope)
  friend class GPUDeviceContextTest_DeviceContextScope_Test;

  class PendingEventHandler;

  DeviceRef device_;

  std::unique_ptr<MemoryManagerFreeList> freeList_;
  std::unique_ptr<MemoryManagerFreeList> hostFreeList_;

  mutable std::atomic<int> syncMode_ = {0};

  // hasPendingError_ indicates whether pendingError_ should be checked.
  // Since pendingError_ is protected by a lock we set an atomic value that can
  // be quickly checked without fully acquiring a mutex lock. Raising errors
  // across threads is inherently racing, so we are fine with possibly
  // missing an update to hasPendingError_ from a different thread.
  mutable std::atomic<int> hasPendingError_ = {0};
  mutable std::mutex pendingErrorMutex_;
  mutable ErrorOrSuccess pendingError_;

  // hasPendingWork_ indicates whether pendingWork_ should be checked.
  // pendingWorkMutex_ protects pendingEvents_ and pendingEventsLast_.
  mutable std::atomic<int> hasPendingWork_ = {0};
  mutable std::mutex pendingWorkMutex_;
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

  // Is the current execution stream for the device running in graph capture
  // mode?
  bool isGraphCapturing() const;

  // Shared configuration object.
  // REQUIRES: `getGlobalMutex()` must be held to get and access the shared
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
  explicit DeviceFunctionCacheEntry(const std::string name);

  const std::string &name() const;

private:
  const std::string name_;
};

} // namespace M::Driver

namespace llvm {
template <>
struct DenseMapInfo<M::Driver::Device::FunctionCacheKey> {
  static inline M::Driver::Device::FunctionCacheKey getEmptyKey() {
    return {""};
  }
  static inline M::Driver::Device::FunctionCacheKey getTombstoneKey() {
    return {""};
  }
  static unsigned getHashValue(const M::Driver::Device::FunctionCacheKey &val) {
    return llvm::hash_combine(val.name, val.debugLevel,
                              val.maxDynamicSharedBytes, val.optimizationLevel);
  }
  static bool isEqual(const M::Driver::Device::FunctionCacheKey &lhs,
                      const M::Driver::Device::FunctionCacheKey &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // MLRT_DRIVER_DEVICECONTEXT_DEVICECONTEXT_H
