//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Metal device context.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_METAL_METALDEVICECONTEXT_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_METAL_METALDEVICECONTEXT_H

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "llvm/ADT/SmallVector.h"

#include <functional>
#include <map>
#include <mutex>

#if MODULAR_MACOS
#include "MetalDriver.h"

namespace M::Driver {

class MetalDevice;
class MetalDeviceContext;
class MetalDeviceEvent;
class MetalDeviceTimer;
class MetalModuleFunction;
class MetalDeviceFunction;
class MetalDeviceFunctionArguments;

using MetalDeviceRef = RCRef<MetalDevice>;
using MetalDeviceContextRef = RCRef<MetalDeviceContext>;
using MetalDeviceEventRef = RCRef<MetalDeviceEvent>;
using MetalDeviceTimerRef = RCRef<MetalDeviceTimer>;
using MetalModuleFunctionRef = RCRef<const MetalModuleFunction>;
using MetalDeviceFunctionRef = RCRef<MetalDeviceFunction>;

class MetalDevice : public Device {
public:
  static constexpr API api = Device::metalAPI;

  // Result type for buffer lookups - returns buffer AND offset within it
  struct BufferLookupResult {
    MTL::Buffer *buffer = nullptr;
    size_t offset = 0;
    bool isValid() const { return buffer != nullptr; }
  };

  // Allocation tracking with size info for range-based lookup
  struct BufferAllocation {
    uintptr_t baseAddress;
    size_t size;
    MTL::Buffer *buffer;
  };

  MetalDevice(int id, NS::SharedPtr<MTL::Device> device, std::string name,
              Device::MemInfo memInfo, size_t allocAlignment);

  ErrorOr<int> getApiVersion() const override;
  ErrorOr<DriverVersion &> getDriverVersion() const override;
  ErrorOrSuccess isCompatible() const override;
  ErrorOrSuccess runHealthcheck() const override { return success(); }
  ErrorOr<int> computeCapability() const override;

  ErrorOr<bool> canAccess(const Device &peer) const override;
  ErrorOr<int> getAttribute(Attribute attr) const override;

  ErrorOr<std::string> getPciBusId() const override;

  int deviceCount() const override;

  ErrorOrSuccess chunkAlloc(void **ptr, size_t byteSize, bool isHost) override;
  ErrorOrSuccess chunkFree(void *ptr, bool isHost) override;

  ErrorOr<void *> getCurrentDriverContext() const override;
  ErrorOrSuccess setCurrentDriverContext(void *ctx) override;
  ErrorOr<DeviceContextRef> createContext() override;
  ErrorOr<Device::MemInfo> getMemoryInfo() const override;

  ErrorOr<size_t> maxSingleAllocationSize() const override;

  static ErrorOr<Device::MemInfo> getMemInfo(MTL::Device *metalDev = nullptr);

  static const MetalDevice &cast(const Device &device);
  static MetalDevice &cast(Device &device);
  static ErrorOr<MetalDevice &> get(int id);
  static ErrorOr<int> numberOfGPUs();

  ErrorOr<const char *> getArchName() const override;
  static ErrorOr<std::string> getArchName(MTL::Device *d = nullptr);

  DLDevice getDLDevice(bool pinned) const override {
    // Metal has no pinned-memory DLPack variant; pinned buffers are plain CPU.
    return pinned ? DLDevice{kDLCPU, 0} : DLDevice{kDLMetal, id()};
  }

  MTL::Device *metalDevice() const { return device_.get(); }

  // Buffer tracking methods
  void trackBuffer(void *ptr, MTL::Buffer *buffer) const;
  void untrackBuffer(void *ptr) const;
  MTL::Buffer *getTrackedBuffer(const void *ptr) const;
  bool isTrackedBuffer(const void *ptr) const;
  std::map<uintptr_t, BufferAllocation> getTrackedBuffers() const;
  // Invokes `fn` for every tracked buffer, in ascending base-address order,
  // with `bufferMapMutex_` held for the whole traversal. Equivalent to
  // iterating `getTrackedBuffers()` but without copying the allocation map --
  // the residency sweep runs this once per launch, so the per-launch
  // `std::map` value-copy is measurable (GEX-3761). `fn` must not re-enter the
  // buffer-tracking methods (they take the same lock).
  void forEachTrackedBuffer(
      const std::function<void(const BufferAllocation &)> &fn) const;
  BufferLookupResult findContainingBuffer(const void *ptr) const;

private:
  NS::SharedPtr<MTL::Device> device_;
  std::string name_;
  Device::MemInfo memInfo_;
  mutable DriverVersion driverVersion_;

  // Lazily-populated cache of the architecture name string returned by the
  // virtual `getArchName() const` override. The static `getArchName(MTL::Device
  // *)` form is expensive (autorelease pool + MTL device introspection + Metal4
  // feature detection), so the cache is computed on first call rather than at
  // construction. `mutable` because the override is `const`.
  mutable std::string arch_;
  mutable std::mutex archMutex_;

  // Device-level buffer tracking (sorted by base address for range-based
  // lookup)
  mutable std::mutex bufferMapMutex_;
  mutable std::map<uintptr_t, BufferAllocation> allocations_;
};

// Metal-specific kernel-launch payload threaded through `args[0]` of
// `MetalDeviceContext::enqueueFunctionExecDirect`. Per-argument sizes,
// per-argument kind (device pointer vs scalar), and the set of buffers to
// mark resident travel here rather than on the `DeviceContext` signature so
// the shared interface does not grow Metal-specific concepts. The layout
// must stay in sync with the Mojo-side `MetalEnqueueFunctionArgs` declared
// in `oss/modular/mojo/stdlib/std/gpu/host/device_context.mojo`.
struct MetalEnqueueFunctionArgs {
  void **args;
  const size_t *argSizes;
  // Parallel to `args`/`argSizes`; `argCount`-sized. Entry `i` is `true`
  // iff `args[i]` holds the device address of a `DeviceBuffer`, and `false`
  // for scalar arguments. Captures are scalars and so are `false`.
  bool *argIsDevicePtr;
  DeviceBuffer **buffers;
  int32_t numBuffers;
};

class MetalDeviceContext : public DeviceContext {
public:
  MetalDeviceContext(MetalDevice &device,
                     NS::SharedPtr<MTL::CommandQueue> commandQueue);

  // DeviceBuffer copies.
  ErrorOrSuccess enqueueCopyToDevice(void *dst, const void *src,
                                     size_t byteSize) override;
  ErrorOrSuccess enqueueCopyFromDevice(void *dst, const void *src,
                                       size_t byteSize) override;
  ErrorOrSuccess memCopyDtoD(DeviceBuffer &dst, DeviceBuffer &src,
                             size_t srcByteOffset, size_t srcByteSize,
                             bool crossStreamSync) override;

  ErrorOrSuccess enqueueSetMemory(void *dst, size_t byteSize, uint64_t val,
                                  size_t valSize) override;

  ErrorOr<void *> getHostAccessiblePointer(const void *devicePtr) override;

  // DeviceFunction loading and execution.
  ErrorOr<DeviceFunctionRef>
  loadFunction(StringRef moduleName, StringRef functionName, StringRef code,
               int32_t maxDynamicSharedBytes, StringRef debugLevel,
               int32_t optimizationLevel) override;

  ErrorOrSuccess enqueueFunctionExecDirect(
      const DeviceFunction &func, uint32_t gridX, uint32_t gridY,
      uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
      uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args,
      uint32_t argCount = 0, const size_t *argSizes = nullptr) override;

  // Synchronize asynchronous operations.
  ErrorOr<DeviceStreamRef> stream() const override;

  ErrorOrSuccess synchronize() override;

  // Event management
  ErrorOr<DeviceEventRef> enqueueEvent() override;
  ErrorOrSuccess enqueueEvent(const DeviceEvent &event) override;
  ErrorOrSuccess enqueueWaitFor(DeviceEvent &event) override;
  ErrorOrSuccess enqueueWaitFor(DeviceContext &ctx) override;
  ErrorOr<DeviceEventRef> eventCreate(bool enableTiming) override;

  // Timer management
  ErrorOr<DeviceTimerRef> startTimer() override;
  ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) override;

  // Memory info
  ErrorOr<Device::MemInfo> getDeviceMemoryInfo() const override;

  // File mapping
  ErrorOr<std::tuple<void *, size_t,
                     std::optional<llvm::sys::fs::mapped_file_region>>>
  memmapFileImpl(const std::filesystem::path &filename) const override;

  // Driver context
  void *driverContext() const override;

  // Metal-specific helpers
  ErrorOrSuccess enqueueMetalCopy(MTL::Buffer *src, MTL::Buffer *dst,
                                  size_t srcOffset, size_t dstOffset,
                                  size_t size) const;
  MetalDevice::BufferLookupResult
  getMetalBufferWithOffset(const void *ptr) const;

  // Metal GPU frame capture to a .gputrace for offline replay in Xcode.
  // Reached from Mojo via the AsyncRT_DeviceContext_*MetalTraceCapture
  // bindings, which verified_cast a generic DeviceContext to this type.
  ErrorOrSuccess startMetalTraceCapture(const char *path);
  ErrorOrSuccess stopMetalTraceCapture();
  // Enable or disable Metal GPU print (os_log). Must be disabled for
  // replayable captures (captured buffers cannot carry a log state).
  ErrorOrSuccess setMetalPrintEnabled(bool enabled);

  static const MetalDeviceContext *verified_cast(const DeviceContext *ctx);
  static MetalDeviceContext *verified_cast(DeviceContext *ctx);
  MTL::Device *metalDevice() const {
    return MetalDevice::cast(device()).metalDevice();
  }

private:
  NS::SharedPtr<MTL::CommandQueue> commandQueue_;

  // FIFO ordering spine: every committed command buffer waits on then updates
  // this fence, so buffers finish in commit order (Metal only orders starts).
  NS::SharedPtr<MTL::Fence> orderFence_;
  // Blocks the host until all previously committed work completes.
  ErrorOrSuccess drainQueue() const;

  NS::SharedPtr<MTL::LogState> logState_;
  MTL::LogState *getOrCreateLogState(bool enable = true);
  // When true, skip MTL::LogState on command buffers (no GPU print).
  // Set by setMetalPrintEnabled() or MODULAR_DISABLE_METAL_GPU_PRINT.
  bool printDisabled_ = false;
  // True between startMetalTraceCapture and stopMetalTraceCapture; suppresses
  // MTL::LogState so captured command buffers are replayable.
  bool captureActive_ = false;
  // Memory allocation
  ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                          Device::AllocMode mode) override;
  ErrorOrSuccess memFree(void *ptr, bool isHost) override;
  // Helper methods for buffer management
  bool isMetalBuffer(const void *ptr) const;
  MTL::Buffer *getMetalBuffer(const void *ptr) const;
  MTL::StorageMode getStorageMode(Device::AllocMode mode) const;
  MTL::ResourceOptions getResourceOptions(Device::AllocMode mode) const;
};

class MetalDeviceEvent : public DeviceEvent {
public:
  // waitValue 0 = never recorded (complete, CUDA semantics); tests pass 1 to
  // build a CPU-released queue gate.
  MetalDeviceEvent(MetalDevice &device, NS::SharedPtr<MTL::SharedEvent> event,
                   uint64_t waitValue = 0);

  ErrorOrSuccess synchronize() override;
  ErrorOr<bool> isReady() const override;

  MTL::SharedEvent *sharedEvent() const { return event_.get(); }
  uint64_t waitValue() const;

  static const MetalDeviceEvent &cast(const DeviceEvent &event);
  static MetalDeviceEvent &cast(DeviceEvent &event);

private:
  friend class MetalDeviceContext;
  NS::SharedPtr<MTL::SharedEvent> event_;
  // Recording mutates through the const DeviceEvent& API (as CUDA does), hence
  // mutable; the mutex spans reserve-to-commit so re-records serialize.
  mutable std::mutex recordMutex_;
  mutable uint64_t waitValue_;
};

class MetalDeviceTimer : public DeviceTimer {
public:
  MetalDeviceTimer(MetalDeviceContext &context,
                   std::chrono::time_point<std::chrono::steady_clock> start);
  ~MetalDeviceTimer() override = default;

  std::chrono::time_point<std::chrono::steady_clock> cpu_start() const;

  static const MetalDeviceTimer &cast(const DeviceTimer &timer);

private:
  const std::chrono::time_point<std::chrono::steady_clock> start_;
};

// Cacheable half of a loaded Metal function: owns the compiled compute
// pipeline state (immutable once created, safe to reuse across launches) and
// the reflected argument count. Lives in the Device function cache. It holds no
// DeviceContext reference so the cache does not create a reference cycle with
// the Device (mirrors CUDAModuleFunction / HIPModuleFunction).
class MetalModuleFunction : public DeviceFunctionCacheEntry {
public:
  MetalModuleFunction(std::string name,
                      NS::SharedPtr<MTL::ComputePipelineState> pipelineState,
                      size_t argumentCount);

  MTL::ComputePipelineState *getPipelineState() const {
    return pipelineState_.get();
  }
  size_t argumentCount() const { return argumentCount_; }

  static const MetalModuleFunction *cast(const DeviceFunctionCacheEntry *entry);

private:
  NS::SharedPtr<MTL::ComputePipelineState> pipelineState_;
  size_t argumentCount_;
};

class MetalDeviceFunction : public DeviceFunction {
public:
  MetalDeviceFunction(MetalDeviceContext &context,
                      MetalModuleFunctionRef modFunc);

  const std::string &name() const override;

  ErrorOrSuccess copyToNamedConstant(const char *name, const void *data,
                                     size_t byteSize) override;

  ErrorOr<int32_t> getAttribute(int32_t code) override;

  size_t argumentCount() const override;

  MTL::ComputePipelineState *getPipelineState() const {
    return moduleFunction_->getPipelineState();
  }

  static const MetalDeviceFunction &cast(const DeviceFunction &func);

private:
  // The moduleFunction_ holds the cacheable pipeline state. This layout breaks
  // the reference cycle between the Device's function cache and the cached
  // functions (mirrors CUDADeviceFunction / HIPDeviceFunction).
  MetalModuleFunctionRef moduleFunction_;
};

} // namespace M::Driver

#else // !MODULAR_MACOS

namespace M::Driver {
// Stub declarations for non-macOS platforms
class MetalDevice;
class MetalDeviceContext;
using MetalDeviceRef = RCRef<MetalDevice>;
using MetalDeviceContextRef = RCRef<MetalDeviceContext>;
} // namespace M::Driver

#endif // MODULAR_MACOS
#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_METAL_METALDEVICECONTEXT_H
