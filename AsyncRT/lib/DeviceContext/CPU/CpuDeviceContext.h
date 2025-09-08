//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the CPU device context.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_LIB_DEVICECONTEXT_CPU_CPUDEVICECONTEXT_H
#define ASYNCRT_LIB_DEVICECONTEXT_CPU_CPUDEVICECONTEXT_H

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"

#include <chrono>
#include <cstdint>
#include <stddef.h>

namespace M::AsyncRT {

class CpuDevice;
class CpuDeviceContext;
class CpuDeviceStream;
class CpuDeviceTimer;

using CpuDeviceRef = RCRef<const CpuDevice>;
using CpuDeviceContextRef = RCRef<const CpuDeviceContext>;
using CpuDeviceStreamRef = RCRef<const CpuDeviceStream>;
using CpuDeviceTimerRef = RCRef<const CpuDeviceTimer>;

class CpuDevice : public Device {
public:
  static constexpr API api = Device::cpuAPI;

  CpuDevice(Device::MemInfo memInfo = Device::MemInfo());
  ~CpuDevice() override;

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

  ErrorOr<Device::MemInfo> getMemoryInfo() const override;

  ErrorOr<void *> getCurrentDriverContext() const override;
  ErrorOrSuccess setCurrentDriverContext(void *ctx) const override;

  ErrorOr<DeviceContextRef> createContext() const override;

  ErrorOr<DeviceStreamRef>
  createExternalStream(void *externalStream) const override;

  static const CpuDevice &cast(const Device &device);

  static ErrorOr<const CpuDevice &> get(int id);
};

class CpuDeviceStream : public DeviceStream {
public:
  CpuDeviceStream(const CpuDevice &device);
  ~CpuDeviceStream() override = default;

  ErrorOrSuccess synchronize() const override;

  ErrorOr<DeviceEventRef> enqueueEvent() const override;

  ErrorOrSuccess eventRecord(const DeviceEvent &event) const override;
  ErrorOrSuccess waitEvent(const DeviceEvent &event,
                           unsigned int flags) const override;

  ErrorOrSuccess enqueueWaitForEvent(const DeviceEvent &event) const override;

  ErrorOrSuccess enqueueFunctionExecDirect(const DeviceFunction &func,
                                           uint32_t gridX, uint32_t gridY,
                                           uint32_t gridZ, uint32_t blockX,
                                           uint32_t blockY, uint32_t blockZ,
                                           uint32_t sharedMemBytes, void *attrs,
                                           uint32_t numAttrs,
                                           void **args) const override;

  void *ptr() const override;
};

class CpuDeviceTimer : public DeviceTimer {
public:
  CpuDeviceTimer(const CpuDeviceContext &context,
                 std::chrono::time_point<std::chrono::steady_clock> start);
  ~CpuDeviceTimer() override = default;

  std::chrono::time_point<std::chrono::steady_clock> cpu_start() const;

  static const CpuDeviceTimer &cast(const DeviceTimer &timer);

private:
  const std::chrono::time_point<std::chrono::steady_clock> start_;
};

class CpuDeviceContext : public DeviceContext {
public:
  CpuDeviceContext(const CpuDevice &device);
  ~CpuDeviceContext() override = default;

  // DeviceBuffer copies.
  ErrorOrSuccess enqueueCopyToDevice(const DeviceBuffer &dst,
                                     const void *src) const override;
  ErrorOrSuccess enqueueCopyFromDevice(void *dst,
                                       const DeviceBuffer &src) const override;
  ErrorOrSuccess enqueueCopyFromDeviceSized(void *dst, void *src,
                                            size_t byteSize) const override;
  ErrorOrSuccess enqueueCopyToDeviceSized(void *dst, void *src,
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

  ErrorOr<DeviceEventRef> enqueueEvent() const override;
  ErrorOr<DeviceEventRef> eventCreate(unsigned int flags) const override;
  ErrorOrSuccess enqueueWaitFor(const DeviceEvent &event) const override;
  ErrorOrSuccess enqueueWaitFor(const DeviceContext &ctx) const override;
  ErrorOrSuccess occupancyMaxActiveBlocksPerMultiprocessor(
      int *numBlocks, const DeviceFunction &func, int blockSize,
      size_t dynamicSharedMemSize) const override;

  ErrorOr<DeviceTimerRef> startTimer() const override;
  ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) const override;

  ErrorOr<Device::MemInfo> getMemoryInfo() const override;

  static const CpuDeviceContext &cast(const DeviceContext &ctx);

private:
  ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                          Device::AllocMode mode) const override;
  ErrorOrSuccess memFree(void *ptr, bool isHost) const override;

  void *driverContext() const override;

  MMapFileImplResult
  memmapFileImpl(const std::filesystem::path &filename) const override;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_CPU_CPUDEVICECONTEXT_H
