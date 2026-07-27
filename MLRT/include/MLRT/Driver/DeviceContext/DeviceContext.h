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

// Default to the legacy backend when the build does not define the flag
// (including targets compiled with -UMODULAR_HAL_DEVICE_CONTEXT).
#ifndef MODULAR_HAL_DEVICE_CONTEXT
#define MODULAR_HAL_DEVICE_CONTEXT 0
#endif

#if MODULAR_HAL_DEVICE_CONTEXT
// Under --config=hal-device-context the M::Driver device classes are the HAL
// shim over the Mojo HAL DeviceContext; the legacy DeviceContext declarations
// below are fully replaced.
#include "MLRT/Driver/DeviceContext/HALDeviceContext.h"
#else

#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/Globals/VirtualDeviceGlobals.h"
#include "CompilationDevice.h"
#include "ErrorMacros.h"
#include "Support/Buffer.h"
#include "Support/Configuration.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include <dlpack/dlpack.h>

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
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
class CompletionFlag;
class Device;
class DeviceBuffer;
class DeviceContext;
class DeviceEvent;
class DeviceFunction;
class DeviceGraph;
class DeviceGraphBuilder;
class DeviceFunctionCacheEntry;
class DeviceMulticastBuffer;
class DeviceStream;
class DeviceTimer;

using CompletionFlagRef = RCRef<CompletionFlag>;
using DeviceRef = RCRef<Device>;
using DeviceFunctionCacheEntryRef = RCRef<const DeviceFunctionCacheEntry>;
using DeviceBufferRef = RCRef<DeviceBuffer>;
using DeviceContextRef = RCRef<DeviceContext>;
using DeviceEventRef = RCRef<DeviceEvent>;
using DeviceFunctionRef = RCRef<DeviceFunction>;
using DeviceGraphRef = RCRef<DeviceGraph>;
using DeviceGraphBuilderRef = RCRef<DeviceGraphBuilder>;
using AsyncDeviceGraphRef = AsyncValueRef<DeviceGraphRef>;
using DeviceMulticastBufferRef = RCRef<DeviceMulticastBuffer>;
using DeviceStreamRef = RCRef<DeviceStream>;
using DeviceTimerRef = RCRef<DeviceTimer>;

size_t getPreferredAlignment(std::string label);

// Empty = host CPU. Accepts an LLVM CPU name (or "generic"); must be set
// before compiling CPU kernels. Thread-safe.
void setVirtualCpuTarget(StringRef cpu);
std::string getVirtualCpuTarget();

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

  struct GraphMemInfo {
    size_t reserved = 0; // Physical memory reserved by graph pool
    size_t used = 0;     // Physical memory actively mapped by graphs
  };

  virtual ~Device();

  const std::string &name() const;

  virtual ErrorOr<const char *> getArchName() const;

  int id() const;

  API api() const;

  virtual ErrorOr<int> getApiVersion() const = 0;
  virtual ErrorOr<DriverVersion &> getDriverVersion() const = 0;
  virtual ErrorOrSuccess isCompatible() const = 0;

  /// Run lightweight GPU health validation. Checks for hardware throttling,
  /// ECC errors, zombie VRAM, etc. Disable with
  /// MODULAR_DEVICE_CONTEXT_DISABLE_HEALTHCHECK=true.
  virtual ErrorOrSuccess runHealthcheck() const = 0;

  virtual ErrorOr<int> computeCapability() const = 0;

  size_t allocationAlignment() const;

  virtual ErrorOr<bool> canAccess(const Device &peer) const = 0;

  // Enable peer access between devices.
  ErrorOrSuccess enablePeerAccess(const Device &peer);

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
                                    bool isHost) = 0;
  virtual ErrorOrSuccess chunkFree(void *ptr, bool isHost) = 0;

  // ---- CUDA Virtual Memory Management (VMM) support ----
  //
  // These back the MemoryManager's single-arena defragmenting allocator: a
  // large contiguous *virtual* allocation can be assembled from physically-
  // scattered free granules by reserving fresh VA and mapping arbitrary
  // physical handles into it (see MemoryManager::ensureRegionBacked /
  // reclaimMappedGranules). They are only used when
  // `vmmSupported()` is true; backends without VMM (HIP/CPU/Metal) keep the
  // plain chunkAlloc/chunkFree path. Handles are opaque
  // (CUmemGenericAllocationHandle) carried as uint64_t; VA is carried as void*.
  //
  // All sizes passed here must be multiples of `vmmGranularity()`.
  virtual bool vmmSupported() const { return false; }
  // Whether the VMM defragmenting allocator is enabled by default on this
  // backend (i.e. when `memory_manager_vmm` is unset). Backends opt in by
  // overriding this; the base default is off. Only takes effect where
  // `vmmSupported()` is also true.
  virtual bool vmmEnabledByDefault() const { return false; }
  // Physical allocation granularity in bytes (e.g. 2 MiB on B200).
  virtual ErrorOr<size_t> vmmGranularity() const;
  // Reserve / free a contiguous virtual address range (no physical backing).
  virtual ErrorOr<void *> vmmAddressReserve(size_t byteSize);
  virtual ErrorOrSuccess vmmAddressFree(void *va, size_t byteSize);
  // Create / release one physical granule (handle is opaque, 0 == invalid).
  virtual ErrorOr<uint64_t> vmmCreate(size_t byteSize);
  virtual ErrorOrSuccess vmmRelease(uint64_t handle);
  // Map `handle` into [va, va+byteSize) and grant this device R/W access.
  virtual ErrorOrSuccess vmmMapAndSetAccess(void *va, size_t byteSize,
                                            uint64_t handle);
  // Unmap [va, va+byteSize); the handle remains valid for reuse.
  virtual ErrorOrSuccess vmmUnmap(void *va, size_t byteSize);

  virtual ErrorOr<Device::MemInfo> getMemoryInfo() const = 0;

  /// Largest single contiguous device allocation. Defaults to
  /// `getMemoryInfo().total`; Metal overrides to `maxBufferLength`.
  virtual ErrorOr<size_t> maxSingleAllocationSize() const;

  virtual ErrorOr<Device::GraphMemInfo> getGraphMemoryInfo() const;

  virtual ErrorOr<DeviceContextRef> createContext() = 0;

  static int numberOfDevices(StringRef api);
  static size_t preferredAlignment(StringRef api);
  // Enables bidirectional peer access for all devices.
  static ErrorOrSuccess enableAllPeerAccess();
  // Returns true if bidirectional peer access is enabled for all devices.
  static ErrorOr<bool> allPeerAccessEnabled();

  static ErrorOr<DeviceRef> create(StringRef api, int id = 0);

  /// Returns the vendor and compute_capability of the accelerator in the form
  /// of `vendor:compute_capability`. For example, an A100 GPU would return
  /// `nvidia:80`.
  static const std::string &getAcceleratorArchOrEmpty();

  /// Function cache key for the function cache. This accounts for how the
  /// function is compiled including debug level and other properties.
  ///
  /// `name` identifies the loaded *module* (a caller-supplied module name, or a
  /// fallback derived from the PTX/cubin blob when none is given).
  /// `functionName` is the entry point resolved out of that module: it must be
  /// part of the key so that two loads of the same blob with different
  /// entry-point names do not collide and silently return the first function.
  struct FunctionCacheKey {
    std::string name;
    std::string functionName = "";
    std::string debugLevel = "";
    int32_t maxDynamicSharedBytes = -1;
    int32_t optimizationLevel = -1;

    bool operator==(const FunctionCacheKey &other) const {
      return name == other.name && functionName == other.functionName &&
             debugLevel == other.debugLevel &&
             maxDynamicSharedBytes == other.maxDynamicSharedBytes &&
             optimizationLevel == other.optimizationLevel;
    }

    /// Derives the `name` field for a load. Returns `moduleName` when the
    /// caller supplied one; otherwise returns a short stable hash of the
    /// PTX/cubin blob rather than the blob itself, so the cache neither
    /// copies/hashes/compares the whole blob on every lookup nor retains a
    /// duplicate of it for the Device's lifetime. (The Mojo `load_function`
    /// binding supplies no module name, so without this every call would key on
    /// the full blob.)
    static std::string moduleCacheName(StringRef moduleName, StringRef code);
  };

  // Internal implementation helpers for DeviceContexts.
  DeviceFunctionCacheEntryRef
  findInFunctionCache(const FunctionCacheKey &key) const;
  DeviceFunctionCacheEntryRef
  insertInFunctionCache(const FunctionCacheKey &key,
                        const DeviceFunctionCacheEntry &entry);
  void clearFunctionCache() const;

  virtual ErrorOr<void *> getCurrentDriverContext() const = 0;
  virtual ErrorOrSuccess setCurrentDriverContext(void *ctx) = 0;

  /// Returns the DLDevice (type + id) for this device, for use in the DLPack
  /// protocol. The `pinned` argument selects the host-pinned variant when true
  /// (e.g. kDLCUDAHost instead of kDLCUDA).
  ///
  /// Override this in a Device subclass to add DLPack support for new device
  /// types without modifying the core DLPack bridge code.
  virtual DLDevice getDLDevice(bool pinned) const = 0;

protected:
  Device(API api, std::string name, int id, MemInfo memInfo, size_t alignment);

  // Enable peer access between devices.
  virtual ErrorOrSuccess enablePeerAccessImpl(const Device &peer);

  // Releases the device MemoryManager's cached allocations if it has been
  // constructed. Intended for subclass destructors: unlike memoryManager(),
  // it never lazily constructs the manager during teardown.
  void clearMemoryManager();

private:
  std::string name_;
  int id_;
  API api_;

  // The device MemoryManager is constructed lazily on first memoryManager()
  // call rather than in this constructor. Whether the VMM defragmenting
  // allocator is usable depends on the derived Device's vmmSupported()
  // override, which is not yet dispatchable while the base Device constructor
  // runs (the derived vtable is not installed until the subclass constructor
  // completes). Deferring construction to first access guarantees the fully-
  // constructed derived object is in place. memInfo_/allocAlignment_ retain
  // the inputs that deferred construction needs.
  MemInfo memInfo_;
  size_t allocAlignment_;
  mutable std::once_flag memoryManagerOnce_;
  mutable std::unique_ptr<MemoryManager> memoryManager_;
  std::unique_ptr<MemoryManager> hostMemoryManager_;

  // Mutex to protect mutable data.
  mutable std::mutex mu_;
  // The function cache holds the lower-level function references which do not
  // contain a device context reference to avoid reference counting cycles.
  mutable DenseMap<FunctionCacheKey, DeviceFunctionCacheEntryRef> funcMap_;
  // Tracks whether P2P access is enabled to other devices.
  mutable llvm::BitVector canAccessPeer_;
};

/// Raw GPU health metrics, used by evaluateGPUHealth().
/// Separating data gathering from evaluation enables unit testing without
/// requiring a real GPU or mocking NVML.
struct GPUHealthData {
  // Throttle reason bitmask constants.
  // NVIDIA (mirrored from NVML):
  static constexpr unsigned long long THROTTLE_HW_SLOWDOWN = 0x8ULL;
  static constexpr unsigned long long THROTTLE_HW_THERMAL_SLOWDOWN = 0x40ULL;
  static constexpr unsigned long long THROTTLE_HW_POWER_BRAKE = 0x80ULL;
  // AMD (from amd-smi throttle violations):
  static constexpr unsigned long long THROTTLE_PROCHOT = 0x100ULL;
  static constexpr unsigned long long THROTTLE_SOCKET_THERMAL = 0x200ULL;
  static constexpr unsigned long long THROTTLE_VR_THERMAL = 0x400ULL;
  static constexpr unsigned long long THROTTLE_HBM_THERMAL = 0x800ULL;

  int deviceId = 0;
  std::string deviceName;
  unsigned long long throttleReasons = 0;
  bool throttleQueried = false;
  unsigned long long eccErrors = 0;
  bool eccQueried = false;
};

/// Evaluate GPU health from pre-gathered metrics. Returns an error with a
/// descriptive message if the GPU is unhealthy.
ErrorOrSuccess evaluateGPUHealth(const GPUHealthData &data);

class DeviceBuffer final : public ReferenceCounted<DeviceBuffer> {
public:
  ~DeviceBuffer();

  DeviceContext &context();
  const DeviceContext &context() const;
  const DeviceBuffer *parent() const;

  // Returns the raw device pointer, which is not safe to access on the host
  void *ptr() const;
  // TODO: Remove the V1 compatibility.
  void *releasePtr();

  size_t byteSize() const;

  bool isHost() const;

  inline ErrorOrSuccess copyTo(DeviceBuffer &dst) {
    return copyTo(dst, 0, dst.byteSize());
  }
  ErrorOrSuccess copyTo(DeviceBuffer &dst, size_t srcByteOffset,
                        size_t srcByteSize);
  ErrorOrSuccess copyTo(void *dst);

  ErrorOrSuccess copyFrom(const void *src);

  ErrorOrSuccess reassignOwnershipTo(DeviceContext &ctx);

  ErrorOr<DeviceBufferRef> createSubBuffer(size_t byteOffset, size_t byteSize);

  // Either do a blocking synchronize or setup events with an external stream.
  ErrorOrSuccess synchronize(void *external_stream = nullptr);

  void setDestructorFn(llvm::unique_function<void()> destructor);

  // Returns the placeholder/dangling pointer for when a DeviceBuffer
  // is created with size `0`. This is needed as in Mojo, the DeviceBuffer
  // class's `device_ptr` field cannot be NULL, so we use a dangling pointer
  // instead of a null pointer. We can track weather this pointer is valid
  // or not by querying the size of this DeviceBuffer instead of relying on
  // the "nullability" of the device pointer.
  static void *emptyBufferPointer();

private:
  friend class DeviceContext;

  // Records that this host/pinned buffer was used as an async-copy source or
  // destination on `ctx`, a stream other than its owning `context_`, so the
  // destructor can order the deferred host free against `ctx` too. No-op for
  // device buffers and for the owning context; for a sub-buffer, forwards to
  // the root owner that performs the free. Thread-safe.
  void recordUsedOnContext(DeviceContext &ctx);

  DeviceContextRef context_;
  DeviceBufferRef parent_;
  void *ptr_;
  size_t byteSize_;
  bool isHost_;
  // TODO: Remove the V1 compatibility.
  bool owning_ = true;
  std::optional<llvm::unique_function<void()>> destructorFn_ = std::nullopt;

  // Streams other than `context_` on which this host/pinned buffer was used as
  // an async-copy staging buffer. The destructor defers the host free until
  // every one of them completes, not just the owning stream, to avoid a
  // use-after-free. Populated by `recordUsedOnContext`; empty for device and
  // single-stream buffers. Guarded by `usedContextsMu_`.
  std::mutex usedContextsMu_;
  llvm::SmallVector<DeviceContextRef, 0> usedContexts_;

  DeviceBuffer(DeviceContext &context, void *ptr, size_t byteSize, bool isHost);
  DeviceBuffer(DeviceContext &context, DeviceBuffer &parent, void *ptr,
               size_t byteSize, bool isHost);
  // TODO: Remove the V1 compatibility.
  DeviceBuffer(DeviceContext &context, void *ptr, size_t byteSize, bool isHost,
               bool owning);
};

class DeviceMulticastBuffer : public ReferenceCounted<DeviceMulticastBuffer> {
public:
  virtual ~DeviceMulticastBuffer();

  virtual ErrorOr<DeviceBufferRef>
  multicastBuffer(DeviceContext &ctx) const = 0;
  virtual ErrorOr<DeviceBufferRef>
  unicastBufferFor(DeviceContext &ctx) const = 0;

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
  Device &device() const;

  virtual ErrorOrSuccess synchronize() = 0;

  virtual ErrorOr<bool> isReady() const = 0;

  /// Returns the elapsed time in milliseconds between this event and \p end.
  /// Both events must have been recorded with timing enabled.
  virtual ErrorOr<float> elapsedTime(const DeviceEvent &end) const;

protected:
  explicit DeviceEvent(Device &device);

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
  explicit DeviceGraphCollection(SmallVector<AsyncDeviceGraphRef> &&graphs)
      : graphs(std::move(graphs)) {}

  /// Set the outputs for the device graph collection by taking ownership of
  /// the set of provided async references.
  void setOutputs(ArrayRef<AsyncRT::AnyAsyncValueRef> outputs);

  /// Get the output async values for the graph collection. These correspond
  /// to the outputs of the MEF model, not any particular DeviceGraph.
  ArrayRef<AsyncRT::AnyAsyncValueRef> getOutputs() const;

  /// Replay the graph collection by launching all graphs.
  ErrorOrSuccess replay() const;

  /// Execute eagerly and compare launch trace against captured graphs.
  /// For each device graph, this runs the execute callback and verifies
  /// that the resulting trace matches the captured graph.
  ErrorOrSuccess
  debugVerifyReplay(llvm::function_ref<ErrorOrSuccess()> execute) const;

  /// The number of graphs in the collection.
  int64_t numGraphs() const;

  /// Get captured device graphs in stream-capture order.
  ArrayRef<AsyncDeviceGraphRef> getGraphs() const { return graphs; }

private:
  /// A DeviceGraphCollection holds the set of captured device graphs which
  /// correspond to a single capture session of a MAX graph. There will be
  /// one graph per device context used by the graph.
  SmallVector<AsyncDeviceGraphRef> graphs;

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
  Device &device() const { return *device_; }

  /// Capture a graph and baseline launch trace by executing a callback.
  /// Stream capture is ended synchronously for all streams, then graph
  /// instantiation is parallelized across AsyncRT's work queue threads.
  static ErrorOr<DeviceGraphCollection>
  capture(M::AsyncRT::CPUDevice &cpuDevice, ArrayRef<DeviceStreamRef> streams,
          llvm::function_ref<ErrorOrSuccess()> execute);

  /// Replay the captured operations on the stored capture stream.
  virtual ErrorOrSuccess replay() const = 0;

  /// Instantiate the captured graph, completing the two-phase capture process.
  /// Must be called exactly once after endCapture() before replay() is used.
  virtual ErrorOrSuccess instantiate() = 0;

  /// Set the stream associated with the device graph.
  void setStream(DeviceStreamRef stream);

  /// Set the launch trace recorded for the device graph.
  void setLaunchTrace(std::vector<KernelLaunchMetadata> baselineLaunchTrace);

  /// Take ownership of the graph's output async values so their stable backing
  /// memory lives as long as the graph that references it. Transferred in by
  /// `DeviceGraphBuilder::instantiate()`.
  void setOutputs(SmallVector<AsyncRT::AnyAsyncValueRef> outputs) {
    outputs_ = std::move(outputs);
  }

  /// Get the graph's output async values, in graph-signature result order.
  /// Consumed by the `mgp.device_graph.execute` primitive to surface the
  /// graph's outputs as its results after replay.
  ArrayRef<AsyncRT::AnyAsyncValueRef> getOutputs() const { return outputs_; }

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
  explicit DeviceGraph(Device &device, DeviceStreamRef captureStream)
      : device_(DeviceRef::copy(&device)),
        captureStream_(std::move(captureStream)) {}

private:
  DeviceRef device_;
  DeviceStreamRef captureStream_;
  bool hasBaselineLaunchTrace_ = false;
  std::vector<KernelLaunchMetadata> baselineLaunchTrace_;

  /// Outputs recorded on the builder whose stable backing memory must outlive
  /// this graph; transferred in by `DeviceGraphBuilder::instantiate()`.
  SmallVector<AsyncRT::AnyAsyncValueRef> outputs_;
};

class DeviceStream : public ReferenceCounted<DeviceStream> {
public:
  virtual ~DeviceStream() = default;
  Device &device() const;

  virtual ErrorOrSuccess synchronize() = 0;

  virtual ErrorOr<DeviceEventRef> enqueueEvent() const = 0;

  virtual ErrorOrSuccess enqueueFunctionExecDirect(
      const DeviceFunction &func, uint32_t gridX, uint32_t gridY,
      uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
      uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args,
      uint32_t argCount = 0, const size_t *argSizes = nullptr) const = 0;
  virtual ErrorOrSuccess eventRecord(const DeviceEvent &event) const = 0;
  virtual ErrorOrSuccess waitEvent(const DeviceEvent &event,
                                   unsigned int flags) const = 0;

  virtual ErrorOrSuccess
  enqueueWaitForEvent(const DeviceEvent &event) const = 0;

  /// Enqueue a host callback that runs after all preceding work on this
  /// stream completes. `fn` must not call any device APIs (per the
  /// `cuLaunchHostFunc` contract). Currently only implemented for CUDA
  /// streams; other backends return an error.
  ///
  /// TODO: Add AMD GPU (HIP) support via `hipLaunchHostFunc`, which has the
  /// same signature and semantics as `cuLaunchHostFunc`.
  virtual ErrorOrSuccess enqueueHostFunc(void (*fn)(void *),
                                         void *userData) const {
    return Error("enqueueHostFunc is not supported on this device");
  }

  /// Enqueues a wait-value node on this stream: subsequent stream
  /// operations stall until the 64-bit value held by `flag` equals
  /// `value`. The call itself returns immediately after enqueuing
  /// (it does NOT block the caller waiting on the host value). The
  /// flag must have been allocated via the same device's
  /// `DeviceContext::allocateCompletionFlag`. Lets a host producer
  /// (typically calling `flag.signal(value)`) gate stream-side
  /// consumers on a CPU-produced result, without a second stream or
  /// a blocking host callback on the consumer's critical path.
  virtual ErrorOrSuccess enqueueWaitOnHostValue(const CompletionFlag &flag,
                                                uint64_t value) const {
    return Error("enqueueWaitOnHostValue is not supported on this device");
  }

  // Raw device driver pointer to enable passing to PyTorch and similar APIs.
  virtual void *ptr() const = 0;

  /// Begin capturing operations on this stream.
  /// Operations enqueued after this call will be recorded into a graph.
  /// Returns error if already capturing or capture not supported.
  virtual ErrorOrSuccess beginCapture() const;

  /// End capture and return the captured graph.
  /// Returns error if not capturing or capture was invalidated.
  virtual ErrorOr<DeviceGraphRef> endCapture();

  /// Begin recording a device launch trace. Default is a no-op.
  virtual void beginLaunchTrace() const;

  /// Take and clear the current device launch trace. Default is empty.
  virtual std::vector<KernelLaunchMetadata> takeLaunchTrace() const;

protected:
  explicit DeviceStream(DeviceContext &context);
  DeviceContext &context_() { return *ctx_; }
  const DeviceContext &context_() const { return *ctx_; }

private:
  DeviceContextRef ctx_;
};

class DeviceFunction : public ReferenceCounted<DeviceFunction> {
public:
  virtual ~DeviceFunction() = default;

  DeviceContext &context();
  const DeviceContext &context() const;

  virtual const std::string &name() const = 0;

  virtual ErrorOrSuccess copyToNamedConstant(const char *name, const void *data,
                                             size_t byteSize) = 0;

  virtual ErrorOr<int32_t> getAttribute(int32_t code) = 0;

  // Returns the number of arguments for this function. Default implementation
  // returns 0 (unknown). Device-specific implementations may override this.
  virtual size_t argumentCount() const;

protected:
  explicit DeviceFunction(DeviceContext &context);

private:
  DeviceContextRef context_;
};

class DeviceTimer : public ReferenceCounted<DeviceTimer> {
public:
  virtual ~DeviceTimer() = default;

  const DeviceContext &context() const;

protected:
  explicit DeviceTimer(DeviceContext &context);

private:
  DeviceContextRef context_;
};

/// 8-byte completion flag in pinned host memory mapped into the device
/// address space. The CPU side reads / writes the value via
/// `reset()` / `signal()` / `load()`; the GPU side waits for a value
/// via `DeviceStream::enqueueWaitOnHostValue`. Allocated by
/// `DeviceContext::allocateCompletionFlag()`; freed by the destructor
/// (which routes through `DeviceContext::freeCompletionFlag`).
///
/// Refcounted so a single flag can be held simultaneously by, e.g., a
/// Python-side wrapper and a host-function trampoline that needs to keep
/// it alive until its AsyncRT worker signals.
class CompletionFlag final : public ReferenceCounted<CompletionFlag> {
public:
  // Constructed only by `DeviceContext::allocateCompletionFlag`.
  CompletionFlag(DeviceContext &context, void *hostPtr, uint64_t devicePtr);
  ~CompletionFlag();

  CompletionFlag(const CompletionFlag &) = delete;
  CompletionFlag &operator=(const CompletionFlag &) = delete;

  /// Device-visible address (suitable for
  /// `DeviceStream::enqueueWaitOnHostValue`).
  uint64_t devicePtr() const { return devicePtr_; }

  /// Relaxed store of 0 to clear the flag.
  void reset();

  /// Release-ordered store of `value` to the flag.
  void signal(uint64_t value);

  /// Acquire load of the current flag value.
  uint64_t load() const;

  const DeviceContext &context() const { return *context_; }

private:
  DeviceContextRef context_;
  void *hostPtr_;
  uint64_t devicePtr_;
};

class DeviceContext : public ReferenceCounted<DeviceContext> {
public:
  virtual ~DeviceContext();

  Device &device() const;

  // DeviceBuffer allocation.
  ErrorOr<DeviceBufferRef> enqueueCreateBuffer(size_t byteSize);

  // Host memory allocation.
  ErrorOr<DeviceBufferRef> createHostBuffer(size_t byteSize);

  /// Allocates an 8-byte `CompletionFlag` in pinned host memory mapped
  /// into this device's address space (CUDA:
  /// `cuMemHostAlloc(... CU_MEMHOSTALLOC_DEVICEMAP)` +
  /// `cuMemHostGetDevicePointer_v2`). Returns a refcounted handle; the
  /// allocation is freed when the last ref is dropped via
  /// `freeCompletionFlag`.
  ///
  /// Only implemented for CUDA. Other backends return an error.
  virtual ErrorOr<CompletionFlagRef> allocateCompletionFlag() {
    return Error("allocateCompletionFlag is not supported on this device");
  }

  /// Frees the host pointer returned by an earlier
  /// `allocateCompletionFlag`. Called by `CompletionFlag`'s destructor;
  /// not intended for direct use.
  virtual ErrorOrSuccess freeCompletionFlag(void *hostPtr) {
    return Error("freeCompletionFlag is not supported on this device");
  }

  /// Page-locks (pins) an existing host range of `byteSize` bytes at `ptr` so
  /// it is DMA-capable for transfers against this device, registering it with
  /// the underlying driver (CUDA: `cuMemHostRegister`; HIP: `hipHostRegister`).
  /// Unlike `createHostBuffer`, the caller owns the host allocation: the range
  /// must remain mapped and stay registered until `unregisterHostMemory` is
  /// called on the same base pointer.
  ///
  /// Only implemented for GPU backends with a host-register primitive (CUDA,
  /// HIP). Other backends return an error.
  virtual ErrorOrSuccess registerHostMemory(void *ptr, size_t byteSize) {
    return Error("registerHostMemory is not supported on this device");
  }

  /// Unregisters a host range previously pinned by `registerHostMemory`.
  /// `ptr` must be a base pointer that was passed to `registerHostMemory`.
  ///
  /// Only implemented for GPU backends with a host-register primitive (CUDA,
  /// HIP). Other backends return an error.
  virtual ErrorOrSuccess unregisterHostMemory(void *ptr) {
    return Error("unregisterHostMemory is not supported on this device");
  }

  // TODO: Remove the V1 compatibility.
  DeviceBufferRef createBufferOwning(void *ptr, size_t byteSize, bool owning);
  DeviceBufferRef createHostBufferOwning(void *ptr, size_t byteSize,
                                         bool owning);

  /// Returns true if `ptr` is device-resident memory addressable by *this*
  /// context, so a caller-provided pointer can be borrowed zero-copy rather
  /// than staged via a host->device copy. Defaults to false (residency
  /// unknown — caller should stage). Backends that can cheaply query pointer
  /// residency (e.g. CUDA) override this.
  virtual bool ownsDeviceAddress(const void *ptr) const { return false; }

  // DeviceBuffer copies.
  virtual ErrorOrSuccess enqueueCopyToDevice(void *dst, const void *src,
                                             size_t byteSize) = 0;
  inline ErrorOrSuccess enqueueCopyToDevice(const DeviceBuffer &dst,
                                            const void *src) {
    if (dst.context().device().api() != device().api())
      RETURN_ERROR("API mismatch between buffer and context");
    return enqueueCopyToDevice(dst.ptr(), src, dst.byteSize());
  }

  /// Batched copy of equal-length (dst, src, size) pairs on this device.
  /// One driver submit when available; otherwise loops enqueueCopyToDevice.
  /// Callers group by device; mixed-device batches are rejected upstream.
  virtual ErrorOrSuccess
  enqueueBatchCopyToDevice(const std::vector<void *> &dsts,
                           const std::vector<const void *> &srcs,
                           const std::vector<size_t> &sizes) {
    assert(dsts.size() == srcs.size() && dsts.size() == sizes.size());
    for (size_t i = 0, n = dsts.size(); i < n; ++i) {
      if (auto err = enqueueCopyToDevice(dsts[i], srcs[i], sizes[i]))
        return err;
    }
    return success();
  }

  virtual ErrorOrSuccess enqueueCopyFromDevice(void *dst, const void *src,
                                               size_t byteSize) = 0;
  inline ErrorOrSuccess enqueueCopyFromDevice(void *dst,
                                              const DeviceBuffer &src) {
    if (src.context().device().api() != device().api())
      RETURN_ERROR("API mismatch between buffer and context");
    return enqueueCopyFromDevice(dst, src.ptr(), src.byteSize());
  }

  // If `crossStreamSync` is true (the default) and the source and destination
  // are on different streams, the driver inserts the cross-stream
  // synchronization required for a correct device-to-device copy: the
  // destination stream waits for the source stream before the copy, and the
  // source stream waits for the copy to complete afterward. Callers that
  // manage this synchronization themselves (e.g. the graph compiler, which
  // emits explicit synchronization ops) can pass false to skip it.
  inline ErrorOrSuccess enqueueCopyDeviceToDevice(DeviceBuffer &dst,
                                                  DeviceBuffer &src,
                                                  bool crossStreamSync = true) {
    return enqueueCopyDeviceToDevice(dst, src, 0, src.byteSize(),
                                     crossStreamSync);
  }
  ErrorOrSuccess enqueueCopyDeviceToDevice(DeviceBuffer &dst, DeviceBuffer &src,
                                           size_t srcByteOffset,
                                           size_t srcByteSize,
                                           bool crossStreamSync = true);

  virtual ErrorOrSuccess enqueueSetMemory(void *dst, size_t byteSize,
                                          uint64_t val, size_t valSize) = 0;
  inline ErrorOrSuccess enqueueSetMemory(const DeviceBuffer &dst, uint64_t val,
                                         size_t valSize) {
    if (dst.context().device().api() != device().api())
      RETURN_ERROR("API mismatch between buffer and context");
    return enqueueSetMemory(dst.ptr(), dst.byteSize(), val, valSize);
  }

  ErrorOr<Driver::DeviceBufferRef>
  memmapFile(const std::filesystem::path &filename);

  // DeviceFunction loading and execution.
  virtual ErrorOr<DeviceFunctionRef>
  loadFunction(StringRef moduleName, StringRef functionName, StringRef code,
               int32_t maxDynamicSharedBytes = -1, StringRef debugLevel = "",
               int32_t optimizationLevel = -1) = 0;

  /// loadFunction with a user-facing debug name to the loaded
  /// function.
  /// The default implementation drops the name and forwards to
  /// `loadFunction`. Override version can decide what to do with
  /// the debugName.
  virtual ErrorOr<DeviceFunctionRef>
  loadFunctionWithDebugName(StringRef moduleName, StringRef functionName,
                            StringRef code, int32_t maxDynamicSharedBytes,
                            StringRef debugLevel, int32_t optimizationLevel,
                            StringRef debugName) {
    return loadFunction(moduleName, functionName, code, maxDynamicSharedBytes,
                        debugLevel, optimizationLevel);
  }
  virtual ErrorOrSuccess enqueueFunctionExecDirect(
      const DeviceFunction &func, uint32_t gridX, uint32_t gridY,
      uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
      uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args,
      uint32_t argCount = 0, const size_t *argSizes = nullptr) = 0;

  virtual ErrorOrSuccess enqueueHostFunction(void (*resumeFunc)(int8_t *),
                                             void (*destroyFunc)(int8_t *),
                                             int8_t *coroHandle) {
    return Error("Not implemented;");
  }

  virtual ErrorOrSuccess enqueueHostFunctionRange(void (*resumeFunc)(int8_t *),
                                                  void (*destroyFunc)(int8_t *),
                                                  int8_t **coroHandle,
                                                  size_t numHandles) {
    return Error("Not implemented;");
  }

  // Synchronize asynchronous operations.

  /// Returns the stream pointer for use in the DLPack protocol, or nullopt if
  /// this context has no CUDA-style stream model (e.g. Metal, Hexagon).
  ///
  /// Override this in a DeviceContext subclass to expose a stream pointer
  /// through DLPack without modifying the core DLPack bridge code.
  virtual std::optional<intptr_t> getDLPackStream() const {
    return std::nullopt;
  }

  /// Returns the active stream for this context.
  /// This stream is used for the "enqueue" family of method calls.
  virtual ErrorOr<DeviceStreamRef> stream() const = 0;
  /// Returns the number of streams in this context's internal pool.
  /// Backends without a multi-stream model report a single stream.
  virtual unsigned numStreams() const;
  /// Reports the range of stream priorities supported by this context via the
  /// out-parameters. Defaults to a single priority level (0, 0); backends with
  /// real stream priorities (e.g. CUDA, HIP) override this.
  virtual ErrorOrSuccess streamPriorityRange(int *leastPriority,
                                             int *greatestPriority);
  // Create a stream associated with this context. The stream is owned by the
  // context and is counted in numStreams().
  virtual ErrorOr<DeviceStreamRef> createStream(int priority = 0);
  // Convert a device-specific stream handle to a DeviceStreamRef.
  // Note: streams created in this manner are NOT counted in numStreams(),
  // which only counts internally-managed streams.
  virtual ErrorOr<DeviceStreamRef> createExternalStream(void *externalStream);

  // Returns a view of this context whose current stream is `streamId`. The view
  // shares this context's full stream set, driver context, and device memory
  // pool; only the current-stream selector differs. Backends without a
  // multi-stream model return this context, so all work runs on its single
  // stream.
  virtual ErrorOr<DeviceContextRef> selectStream(unsigned streamId);

  virtual ErrorOr<DeviceGraphBuilderRef> createGraphBuilder();

  virtual ErrorOrSuccess synchronize() = 0;

  virtual ErrorOr<DeviceEventRef> enqueueEvent() = 0;
  virtual ErrorOrSuccess enqueueEvent(const DeviceEvent &event) = 0;

  /// Creates a device event. Defaults to a not-supported error; backends with
  /// an event model override this.
  virtual ErrorOr<DeviceEventRef> eventCreate(bool enableTiming);
  virtual ErrorOrSuccess enqueueWaitFor(DeviceEvent &event) = 0;
  virtual ErrorOrSuccess enqueueWaitFor(DeviceContext &ctx) = 0;
  ErrorOrSuccess enqueueWaitForEvent(DeviceEvent &event) {
    return enqueueWaitFor(event);
  }

  virtual ErrorOr<DeviceTimerRef> startTimer() = 0;
  virtual ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) = 0;

  ErrorOr<Device::MemInfo> getMemoryInfo() const;

  // Context-level attribute query. Subclasses override to provide
  // context-specific values (e.g. CpuDeviceContext returns the cpuDevice's
  // parallelism level). Default returns an error for unknown attributes.
  enum class Attribute : int {
    // Number of parallel worker threads for CPU work dispatch.
    PARALLELISM_LEVEL = 1000,
  };
  virtual ErrorOr<int> getAttribute(Attribute attr) const;

  /// Queries the maximum number of active blocks per multiprocessor for a
  /// function. Defaults to a not-supported error; backends with an occupancy
  /// model (e.g. CUDA, HIP) override this.
  virtual ErrorOrSuccess
  occupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
                                            DeviceFunction &func, int blockSize,
                                            size_t dynamicSharedMemSize);

  // Is the current execution stream for the device running in graph capture
  // mode?
  virtual bool isGraphCapturing() const;

  ErrorOrSuccess setAsCurrent() const;

  // Returns true if the device context is bound to the current thread.
  bool isBoundToCurrentThread() const;

  // Binds the device context to the current thread, or no-op if already bound.
  ErrorOrSuccess bindToCurrentThread() const;

  // Clears the bound context for the current thread.
  static void unbindFromCurrentThread();

  MemoryManagerFreeList &freeList() const;
  MemoryManagerFreeList &hostFreeList() const;
  MemoryManagerFreeList &graphFreeList() const;

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

  // Returns true if device sync mode is enabled via either the new
  // `max-debug.device-sync-mode` debug configuration or the legacy
  // `device-context.sync-mode` configuration key.
  static bool syncModeFromConfig();

  // Error helpers.
  static Error formatError(const char *file, int line, const char *func,
                           const llvm::Twine &msg);
  ErrorOrSuccess checkPendingError();

  // Enqueue event handlers.
  ErrorOrSuccess enqueueEventHandler(std::function<void()> handler);
  // Poll event handlers when events are ready.
  ErrorOrSuccess pollPendingEvents();

  // Debugging helpers.
  bool syncMode() const;
  void incSyncMode();
  void decSyncMode();
  bool isDebugAllocatorEnabled(StringRef name) const;

protected:
  explicit DeviceContext(Device &device);

  // Device-specific memory information (raw hardware memory)
  virtual ErrorOr<Device::MemInfo> getDeviceMemoryInfo() const = 0;

  // Low-level uncached allocation of memory.
  // REQUIRES: byteSize > 0
  virtual ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                                  Device::AllocMode mode) = 0;
  virtual ErrorOrSuccess memFree(void *ptr, bool isHost) = 0;

  // Low-level copies. When `crossStreamSync` is true, implementations insert
  // the cross-stream synchronization needed when source and destination are on
  // different streams (see `enqueueCopyDeviceToDevice`).
  virtual ErrorOrSuccess memCopyDtoD(DeviceBuffer &dst, DeviceBuffer &src,
                                     size_t srcByteOffset, size_t srcByteSize,
                                     bool crossStreamSync) = 0;

  using MMapFileImplResult =
      ErrorOr<std::tuple<void *, size_t,
                         std::optional<llvm::sys::fs::mapped_file_region>>>;

  virtual MMapFileImplResult
  memmapFileImpl(const std::filesystem::path &filename) const = 0;

  virtual void *driverContext() const = 0;

  void setPendingError(Error err);

private:
  friend class CompletionFlag;
  friend class MemoryManagerFreeList;
  friend class DeviceBuffer;
  friend class DeviceContextScope;
  // Following line is equivalent to:
  // FRIEND_TEST(GPUDeviceContextTest, DeviceContextScope)
  friend class GPUDeviceContextTest_DeviceContextScope_Test;
  // Following line is equivalent to:
  // FRIEND_TEST(MetalDeviceContextTestBase, MemAllocDonatesRefcountToTracker)
  friend class MetalDeviceContextTestBase_MemAllocDonatesRefcountToTracker_Test;

  class PendingEventHandler;

  DeviceRef device_;

  std::unique_ptr<MemoryManagerFreeList> freeList_;
  std::unique_ptr<MemoryManagerFreeList> hostFreeList_;
  std::unique_ptr<MemoryManagerFreeList> graphFreeList_;

  std::atomic<int> syncMode_ = {0};

  // hasPendingError_ indicates whether pendingError_ should be checked.
  // Since pendingError_ is protected by a lock we set an atomic value that can
  // be quickly checked without fully acquiring a mutex lock. Raising errors
  // across threads is inherently racing, so we are fine with possibly
  // missing an update to hasPendingError_ from a different thread.
  std::atomic<int> hasPendingError_ = {0};
  std::mutex pendingErrorMutex_;
  ErrorOrSuccess pendingError_ = success();

  // hasPendingWork_ indicates whether pendingWork_ should be checked.
  // pendingWorkMutex_ protects pendingEvents_ and pendingEventsLast_.
  std::atomic<int> hasPendingWork_ = {0};
  std::mutex pendingWorkMutex_;
  PendingEventHandler *pendingEvents_ = nullptr;
  PendingEventHandler *pendingEventsLast_ = nullptr;

  // Explicitly mark copy and assignment as unsupported.
  DeviceContext(const DeviceContext &other) = delete;
  DeviceContext(DeviceContext &&other) = delete;
  DeviceContext &operator=(const DeviceContext &) = delete;
  DeviceContext &operator=(DeviceContext &&other) = delete;

  // DeviceBuffer allocation.
  ErrorOr<DeviceBufferRef> allocBuffer(size_t byteSize, Device::AllocMode mode);

  void setPendingEventHandler(PendingEventHandler *h);

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
  // Maximum opt-in shared memory per block (CUDA:
  // cudaDevAttrMaxSharedMemoryPerBlockOptin)
  MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
  // Maximum resident blocks per multiprocessor
  MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
  // CUDA-only: Maximum value of CUaccessPolicyWindow::num_bytes.
  MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
  // Metal-only: Disable `print` to improve kernel's execution time.
  METAL_DISABLE_LOGGING = 110,
};

//
// Internal implementation details follow.
//

class DeviceFunctionCacheEntry
    : public ReferenceCounted<DeviceFunctionCacheEntry> {
public:
  explicit DeviceFunctionCacheEntry(const std::string name);

  // Must be virtual: `ReferenceCounted::destroy()` deletes through a
  // `DeviceFunctionCacheEntry *`, so without this the derived destructor
  // (CUDA/HIP/Metal `ModuleFunction`, which release GPU pipeline/module
  // handles) would never run -- undefined behavior when a cache entry is
  // evicted.
  virtual ~DeviceFunctionCacheEntry() = default;

  const std::string &name() const;

private:
  const std::string name_;
};

} // namespace M::Driver

namespace llvm {
template <>
struct DenseMapInfo<M::Driver::Device::FunctionCacheKey> {
  // Empty and tombstone must be distinct from each other and from any real
  // key. A real key always carries a non-empty `name` — `moduleCacheName`
  // returns either the caller's module name or a hash of the code, never an
  // empty string. Both sentinels leave `name` empty (and differ from each other
  // in `functionName`), so neither can alias a real entry.
  static inline M::Driver::Device::FunctionCacheKey getEmptyKey() {
    return {/*name=*/"", /*functionName=*/""};
  }
  static inline M::Driver::Device::FunctionCacheKey getTombstoneKey() {
    return {/*name=*/"", /*functionName=*/"\x01__tombstone__"};
  }
  static unsigned getHashValue(const M::Driver::Device::FunctionCacheKey &val) {
    return llvm::hash_combine(val.name, val.functionName, val.debugLevel,
                              val.maxDynamicSharedBytes, val.optimizationLevel);
  }
  static bool isEqual(const M::Driver::Device::FunctionCacheKey &lhs,
                      const M::Driver::Device::FunctionCacheKey &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // MODULAR_HAL_DEVICE_CONTEXT

#endif // MLRT_DRIVER_DEVICECONTEXT_DEVICECONTEXT_H
