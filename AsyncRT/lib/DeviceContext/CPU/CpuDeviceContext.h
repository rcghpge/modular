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
#include <string>

namespace M::AsyncRT {

class CpuDevice;
class CpuDeviceBuffer;
class CpuDeviceContext;
class CpuDeviceStream;
class CpuDeviceTimer;

using CpuDeviceRef = RCRef<const CpuDevice>;
using CpuDeviceBufferRef = RCRef<const CpuDeviceBuffer>;
using CpuDeviceContextRef = RCRef<const CpuDeviceContext>;
using CpuDeviceStreamRef = RCRef<const CpuDeviceStream>;
using CpuDeviceTimerRef = RCRef<const CpuDeviceTimer>;

class CpuDevice : public Device {
public:
  static constexpr API api = Device::cpuAPI;

  CpuDevice();
  ~CpuDevice() override = default;

  ErrorOr<int> getDriverVersion() const override;
  ErrorOrSuccess isCompatibleWithMAX() const override;
  ErrorOr<int> computeCapability() const override;

  ErrorOr<MemInfo> getMemoryInfo() const override;

  ErrorOr<bool> canAccess(const Device &peer) const override;
  ErrorOr<int> getAttribute(Attribute attr) const override;

  int deviceCount() const override;

  ErrorOr<DeviceContextRef>
  createContext(size_t buffer_cache_size = 0) const override;

  static ErrorOr<const CpuDevice &> get(int id);
};

class CpuDeviceBuffer : public DeviceBuffer {
public:
  CpuDeviceBuffer(const CpuDeviceContext &context, size_t byteSize, void *ptr);
  CpuDeviceBuffer(const CpuDeviceContext &context,
                  const CpuDeviceBuffer &parent, size_t byteSize, void *ptr);
  // TODO: Remove the V1 compatibility.
  CpuDeviceBuffer(const CpuDeviceContext &context, size_t byteSize, void *ptr,
                  bool owning);
  ~CpuDeviceBuffer() override;

  void *cpu_ptr() const;
  void *ptr() const override;
  // TODO: Remove the V1 compatibility.
  void *releasePtr() const override;

  ErrorOr<DeviceBufferRef> createSubBuffer(size_t byteOffset,
                                           size_t byteSize) const override;

  static const CpuDeviceBuffer &cast(const DeviceBuffer &buf);

private:
  // TODO: Remove the V1 compatibility.
  mutable void *ptr_;
  mutable bool owning_ = true;
};

class CpuDeviceStream : public DeviceStream {
public:
  CpuDeviceStream(const CpuDeviceContext &context);
  ~CpuDeviceStream() override = default;

  virtual ErrorOrSuccess synchronize() const override;
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

  // Host memory allocation.
  ErrorOr<void *> malloc_host(size_t size) const override;
  ErrorOrSuccess free_host(void *ptr) const override;

  // DeviceBuffer allocation.
  ErrorOr<DeviceBufferRef> createBuffer(size_t byteSize,
                                        SyncMode mode) const override;
  // TODO: Remove the V1 compatibility.
  DeviceBufferRef createBufferOwning(void *ptr, size_t byteSize,
                                     bool owning) const override;

  // DeviceBuffer copies.
  ErrorOrSuccess copyToDevice(const DeviceBuffer &dst, const void *src,
                              SyncMode mode) const override;
  ErrorOrSuccess copyFromDevice(void *dst, const DeviceBuffer &src,
                                SyncMode mode) const override;
  ErrorOrSuccess copyDeviceToDevice(const DeviceBuffer &dst,
                                    const DeviceBuffer &src,
                                    SyncMode mode) const override;

  ErrorOrSuccess setMemory(const DeviceBuffer &dst, uint32_t val,
                           size_t val_size, SyncMode mode) const override;

  // DeviceFunction loading and execution.
  ErrorOr<DeviceFunctionRef> loadFunction(
      const char *module_name, const char *function_name,
      const DeviceFunction::SignatureBuilder &signature, const void *data,
      int32_t max_registers, int32_t threads_per_block, int32_t cache_mode,
      int32_t cache_config, int32_t max_dynamic_shared_bytes,
      const char *debug_level, int32_t optimization_level) const override;
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
                                           uint32_t num_attrs,
                                           void **args) const override;

  ErrorOr<DeviceStreamRef> stream() const override;

  // Synchronize asynchronous operations.
  ErrorOrSuccess synchronize() const override;

  ErrorOr<DeviceTimerRef> startTimer() const override;
  ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) const override;

  ErrorOrSuccess enablePeerAccess(const DeviceContext &peer) const override;

  static const CpuDeviceContext &cast(const DeviceContext &ctx);
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_CPU_CPUDEVICECONTEXT_H
