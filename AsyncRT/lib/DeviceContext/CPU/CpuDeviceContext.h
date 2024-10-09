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
class CpuDeviceTimer;

using CpuDeviceRef = RCRef<const CpuDevice>;
using CpuDeviceBufferRef = RCRef<const CpuDeviceBuffer>;
using CpuDeviceContextRef = RCRef<const CpuDeviceContext>;
using CpuDeviceTimerRef = RCRef<const CpuDeviceTimer>;

class CpuDevice : public Device {
public:
  CpuDevice() = default;
  ~CpuDevice() override = default;

  int id() const override;

  std::string name() const override;

  static ErrorOr<int> numberOfDevices();
};

class CpuDeviceBuffer : public DeviceBuffer {
public:
  CpuDeviceBuffer(const CpuDeviceContext &context, size_t len, size_t elem_size,
                  void *ptr);
  ~CpuDeviceBuffer() override;

  void *cpu_ptr() const;
  void *ptr() const override;

  static const CpuDeviceBuffer &cast(const DeviceBuffer &buf);

private:
  void *ptr_;
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
  ErrorOr<DeviceBufferRef> createBuffer(size_t len, size_t elem_size,
                                        SyncMode mode) const override;

  // DeviceBuffer copies.
  ErrorOrSuccess copyToDevice(const DeviceBuffer &dst, const void *src,
                              SyncMode mode) const override;
  ErrorOrSuccess copyFromDevice(void *dst, const DeviceBuffer &src,
                                SyncMode mode) const override;
  ErrorOrSuccess copyDeviceToDevice(const DeviceBuffer &dst,
                                    const DeviceBuffer &src,
                                    SyncMode mode) const override;

  ErrorOrSuccess setMemory(const DeviceBuffer &dst, uint32_t val,
                           size_t elem_size, SyncMode mode) const override;

  // DeviceFunction compilation and execution.
  ErrorOr<DeviceFunctionRef> compileFunction(
      const char *function_name,
      const DeviceFunction::SignatureBuilder &signature, const void *data,
      int32_t max_registers, int32_t threads_per_block, int32_t cache_mode,
      int32_t cache_config, int32_t max_dynamic_shared_bytes) const override;
  ErrorOrSuccess
  enqueueFunctionExec(const DeviceFunction &func, uint32_t gridX,
                      uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                      uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
                      const DeviceFunction::Arguments &args) const override;
  ErrorOrSuccess enqueueFunctionExecDirect(const DeviceFunction &func,
                                           uint32_t gridX, uint32_t gridY,
                                           uint32_t gridZ, uint32_t blockX,
                                           uint32_t blockY, uint32_t blockZ,
                                           uint32_t sharedMemBytes,
                                           void **args) const override;

  // Synchronize asynchronous operations.
  ErrorOrSuccess synchronize() const override;

  ErrorOr<DeviceTimerRef> startTimer() const override;
  ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) const override;

  ErrorOr<DeviceMemInfo> getMemoryInfo() const override;

  static const CpuDeviceContext &cast(const DeviceContext &ctx);

  static ErrorOr<CpuDeviceContextRef> create(int id);
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_CPU_CPUDEVICECONTEXT_H
