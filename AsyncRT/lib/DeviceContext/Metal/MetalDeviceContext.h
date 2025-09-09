//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Metal device context.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_LIB_DEVICECONTEXT_METAL_METALDEVICECONTEXT_H
#define ASYNCRT_LIB_DEVICECONTEXT_METAL_METALDEVICECONTEXT_H

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"

#include <mutex>
#include <unordered_map>

#ifdef MODULAR_MACOS
#include "MetalImpl.hpp"

namespace M::AsyncRT {

class MetalDevice;
class MetalDeviceContext;
class MetalDeviceEvent;
class MetalDeviceTimer;
class MetalDeviceFunction;
class MetalDeviceFunctionArguments;

using MetalDeviceRef = RCRef<const MetalDevice>;
using MetalDeviceContextRef = RCRef<const MetalDeviceContext>;
using MetalDeviceEventRef = RCRef<const MetalDeviceEvent>;
using MetalDeviceTimerRef = RCRef<const MetalDeviceTimer>;
using MetalDeviceFunctionRef = RCRef<const MetalDeviceFunction>;

class MetalDevice : public Device {
public:
  static constexpr API api = Device::metalAPI;

  MetalDevice(int id, MTL::Device *device, std::string name,
              Device::MemInfo memInfo, size_t allocAlignment);
  ~MetalDevice() override;

  ErrorOr<int> getApiVersion() const override;
  ErrorOr<DriverVersion &> getDriverVersion() const override;
  ErrorOrSuccess isCompatible() const override;
  ErrorOr<int> computeCapability() const override;

  ErrorOr<bool> canAccess(const Device &peer) const override;
  ErrorOr<int> getAttribute(Attribute attr) const override;

  int deviceCount() const override;

  ErrorOrSuccess chunkAlloc(void **ptr, size_t byteSize,
                            bool isHost) const override;
  ErrorOrSuccess chunkFree(void *ptr, bool isHost) const override;

  ErrorOr<void *> getCurrentDriverContext() const override;
  ErrorOrSuccess setCurrentDriverContext(void *ctx) const override;
  ErrorOr<DeviceContextRef> createContext() const override;
  ErrorOr<Device::MemInfo> getMemoryInfo() const override;

  static ErrorOr<Device::MemInfo> getMemInfo();

  ErrorOr<DeviceStreamRef>
  createExternalStream(void *externalStream) const override;

  static const MetalDevice &cast(const Device &device);
  static ErrorOr<const MetalDevice &> get(int id);
  static ErrorOr<int> numberOfGPUs();

  MTL::Device *metalDevice() const { return device_; }

  // Buffer tracking methods
  void trackBuffer(void *ptr, MTL::Buffer *buffer) const;
  void untrackBuffer(void *ptr) const;
  MTL::Buffer *getTrackedBuffer(const void *ptr) const;
  bool isTrackedBuffer(const void *ptr) const;

private:
  MTL::Device *device_;
  std::string name_;
  Device::MemInfo memInfo_;
  mutable DriverVersion driverVersion_;

  // Device-level buffer tracking
  mutable std::mutex bufferMapMutex_;
  mutable std::unordered_map<void *, MTL::Buffer *> ptrToBuffer_;
};

class MetalDeviceContext : public DeviceContext {
public:
  MetalDeviceContext(const MetalDevice &device,
                     MTL::CommandQueue *commandQueue);
  ~MetalDeviceContext() override;

  // DeviceBuffer copies.
  ErrorOrSuccess enqueueCopyToDevice(const DeviceBuffer &dst,
                                     const void *src) const override;
  ErrorOrSuccess enqueueCopyFromDevice(void *dst,
                                       const DeviceBuffer &src) const override;
  ErrorOrSuccess enqueueCopyToDeviceSized(void *dst, void *src,
                                          size_t byteSize) const override;
  ErrorOrSuccess enqueueCopyFromDeviceSized(void *dst, void *src,
                                            size_t byteSize) const override;
  ErrorOrSuccess memCopyDtoD(const DeviceBuffer &dst,
                             const DeviceBuffer &src) const override;

  ErrorOrSuccess enqueueSetMemory(const DeviceBuffer &dst, uint64_t val,
                                  size_t valSize) const override;

  // DeviceFunction loading and execution.
  ErrorOr<DeviceFunctionRef>
  loadFunction(StringRef moduleName, StringRef functionName,
               const DeviceFunction::SignatureBuilder &signature,
               StringRef code, int32_t maxDynamicSharedBytes,
               StringRef debugLevel, int32_t optimizationLevel) const override;

  ErrorOrSuccess
  enqueueFunctionExec(const DeviceFunction &func, uint32_t gridX,
                      uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                      uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
                      const DeviceFunction::Arguments &args) const override;

  ErrorOrSuccess enqueueFunctionExecDirect(
      const DeviceFunction &func, uint32_t gridX, uint32_t gridY,
      uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
      uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args,
      const size_t *argSizes = nullptr) const override;

  // Synchronize asynchronous operations.
  ErrorOr<DeviceStreamRef> stream() const override;
  ErrorOr<DeviceStreamRef> streamCreate(unsigned int flags) const override;
  ErrorOr<DeviceStreamRef>
  streamCreateWithPriority(unsigned int flags, int priority) const override;
  ErrorOrSuccess streamPriorityRange(int *leastPriority,
                                     int *greatestPriority) const override;
  ErrorOrSuccess synchronize() const override;

  // Event management
  ErrorOr<DeviceEventRef> enqueueEvent() const override;
  ErrorOr<DeviceEventRef> eventCreate(unsigned int flags) const override;
  ErrorOrSuccess enqueueWaitFor(const DeviceEvent &event) const override;
  ErrorOrSuccess enqueueWaitFor(const DeviceContext &ctx) const override;

  // Timer management
  ErrorOr<DeviceTimerRef> startTimer() const override;
  ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) const override;

  // Memory info
  ErrorOr<Device::MemInfo> getDeviceMemoryInfo() const override;

  // Peer access
  ErrorOrSuccess enablePeerAccess(const DeviceContext &peer) const override;

  ErrorOrSuccess occupancyMaxActiveBlocksPerMultiprocessor(
      int *numBlocks, const DeviceFunction &func, int blockSize,
      size_t dynamicSharedMemSize) const override;

  // File mapping
  ErrorOr<std::tuple<void *, size_t,
                     std::optional<llvm::sys::fs::mapped_file_region>>>
  memmapFileImpl(const std::filesystem::path &filename) const override;

  // Driver context
  void *driverContext() const override;

  // Metal-specific helper
  ErrorOrSuccess enqueueMetalCopy(MTL::Buffer *src, MTL::Buffer *dst,
                                  size_t size) const;

private:
  MTL::CommandQueue *commandQueue_;
  // Memory allocation
  ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                          Device::AllocMode mode) const override;
  ErrorOrSuccess memFree(void *ptr, bool isHost) const override;
  // Helper methods for buffer management
  bool isMetalBuffer(const void *ptr) const;
  MTL::Buffer *getMetalBuffer(const void *ptr) const;
  MTL::StorageMode getStorageMode(Device::AllocMode mode) const;
  MTL::ResourceOptions getResourceOptions(Device::AllocMode mode) const;
};

class MetalDeviceEvent : public DeviceEvent {
public:
  MetalDeviceEvent(const MetalDevice &device, MTL::CommandBuffer *buffer);
  ~MetalDeviceEvent() override;

  ErrorOrSuccess synchronize() const override;
  ErrorOr<bool> isReady() const override;

  static const MetalDeviceEvent &cast(const DeviceEvent &event);

private:
  MTL::CommandBuffer *buffer_;
};

class MetalDeviceTimer : public DeviceTimer {
public:
  MetalDeviceTimer(const MetalDeviceContext &context,
                   std::chrono::time_point<std::chrono::steady_clock> start);
  ~MetalDeviceTimer() override = default;

  std::chrono::time_point<std::chrono::steady_clock> cpu_start() const;

  static const MetalDeviceTimer &cast(const DeviceTimer &timer);

private:
  const std::chrono::time_point<std::chrono::steady_clock> start_;
};

class MetalDeviceFunctionArguments : public DeviceFunction::Arguments {
public:
  explicit MetalDeviceFunctionArguments(const MetalDeviceFunction &fn);
  ~MetalDeviceFunctionArguments() override = default;

  void pushArg(int index, const uint64_t &val) override;
  void pushArg(int index, const DeviceBuffer &val) override;

  const std::vector<void *> &metalArgs() const;

  static const MetalDeviceFunctionArguments &
  cast(const DeviceFunction::Arguments &args);

  const MetalDeviceFunction *getFunction() const { return metalFnPtr_; }

private:
  std::vector<void *> argValues_;
  std::vector<DeviceBufferRef> refs_;
  const MetalDeviceFunction *metalFnPtr_ = nullptr;
};

class MetalDeviceFunction : public DeviceFunction {
public:
  MetalDeviceFunction(const MetalDeviceContext &context, StringRef moduleName,
                      StringRef functionName,
                      const DeviceFunction::SignatureBuilder &signature,
                      MTL::ComputePipelineState &pipelineState);
  ~MetalDeviceFunction() override;

  const std::string &name() const override;

  ErrorOrSuccess copyToNamedConstant(const char *name, const void *data,
                                     size_t byteSize) const override;

  ErrorOr<int32_t> getAttribute(int32_t code) const override;

  DeviceFunction::Arguments *
  allocateArguments(const DeviceFunction &fn) const override;

  MTL::ComputePipelineState &getPipelineState() const { return pipelineState_; }

  static const MetalDeviceFunction &cast(const DeviceFunction &func);

private:
  std::string name_;
  MTL::ComputePipelineState &pipelineState_;
};

} // namespace M::AsyncRT

#else // !MODULAR_MACOS

namespace M::AsyncRT {
// Stub declarations for non-macOS platforms
class MetalDevice;
class MetalDeviceContext;
using MetalDeviceRef = RCRef<const MetalDevice>;
using MetalDeviceContextRef = RCRef<const MetalDeviceContext>;
} // namespace M::AsyncRT

#endif // MODULAR_MACOS
#endif // ASYNCRT_LIB_DEVICECONTEXT_METAL_METALDEVICECONTEXT_H
