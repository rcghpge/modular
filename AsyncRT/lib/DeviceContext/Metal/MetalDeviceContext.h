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

#ifdef MODULAR_MACOS
#include "MetalImpl.hpp"

namespace M::AsyncRT {

class MetalDevice;
class MetalDeviceContext;

using MetalDeviceRef = RCRef<const MetalDevice>;
using MetalDeviceContextRef = RCRef<const MetalDeviceContext>;

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

private:
  MTL::Device *device_;
  std::string name_;
  Device::MemInfo memInfo_;
  size_t allocAlignment_;
  mutable DriverVersion driverVersion_;
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

  ErrorOrSuccess enqueueFunctionExecDirect(const DeviceFunction &func,
                                           uint32_t gridX, uint32_t gridY,
                                           uint32_t gridZ, uint32_t blockX,
                                           uint32_t blockY, uint32_t blockZ,
                                           uint32_t sharedMemBytes, void *attrs,
                                           uint32_t numAttrs,
                                           void **args) const override;

  // Synchronize asynchronous operations.
  ErrorOr<DeviceStreamRef> stream() const override;
  ErrorOrSuccess synchronize() const override;

  // Event management
  ErrorOr<DeviceEventRef> enqueueEvent() const override;
  ErrorOrSuccess enqueueWaitFor(const DeviceEvent &event) const override;
  ErrorOrSuccess enqueueWaitFor(const DeviceContext &ctx) const override;

  // Timer management
  ErrorOr<DeviceTimerRef> startTimer() const override;
  ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) const override;

  // Peer access
  ErrorOrSuccess enablePeerAccess(const DeviceContext &peer) const override;

  // Memory info
  ErrorOr<Device::MemInfo> getMemoryInfo() const override;

  // Memory allocation
  ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                          Device::AllocMode mode) const override;
  ErrorOrSuccess memFree(void *ptr, bool isHost) const override;

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
