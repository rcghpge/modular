//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the HIP device context.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_LIB_DEVICECONTEXT_HIP_HIPDEVICECONTEXT_H
#define ASYNCRT_LIB_DEVICECONTEXT_HIP_HIPDEVICECONTEXT_H

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "AsyncRT/lib/DeviceContext/BufferCache.h"
#include "HIPDriver.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "llvm/ADT/StringMap.h"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace M::AsyncRT {

class HIPDevice;
class HIPDeviceBuffer;
class HIPDeviceContext;
class HIPDeviceFunction;
class HIPDeviceFunctionArguments;
class HIPDeviceStream;
class HIPDeviceTimer;

using HIPDeviceRef = RCRef<const HIPDevice>;
using HIPDeviceBufferRef = RCRef<const HIPDeviceBuffer>;
using HIPDeviceContextRef = RCRef<const HIPDeviceContext>;
using HIPDeviceFunctionRef = RCRef<const HIPDeviceFunction>;
using HIPDeviceFunctionArgumentsRef = RCRef<const HIPDeviceFunctionArguments>;
using HIPDeviceStreamRef = RCRef<const HIPDeviceStream>;
using HIPDeviceTimerRef = RCRef<const HIPDeviceTimer>;

class HIPDevice : public Device {
public:
  static constexpr API api = "hip";

  using DeviceList = std::vector<HIPDeviceRef>;
  HIPDevice(const HIPDriver &driver, int id, hipDevice_t device,
            std::string name, size_t alloc_alignment);
  ~HIPDevice() override = default;

  const HIPDriver &hip_driver() const;
  hipDevice_t hip_device() const;

  ErrorOr<int> getDriverVersion() const override;
  ErrorOrSuccess isCompatibleWithMAX() const override;
  ErrorOr<int> computeCapability() const override;

  ErrorOr<MemInfo> getMemoryInfo() const override;

  ErrorOr<bool> canAccess(const Device &peer) const override;
  ErrorOr<int> getAttribute(Attribute attr) const override;

  int deviceCount() const override;

  ErrorOr<DeviceContextRef>
  createContext(size_t buffer_cache_size = 0) const override;

  static const HIPDevice &cast(const Device &device);

  static ErrorOr<const HIPDevice &> get(int id);

private:
  // We only support HIP architectures sm_80 and above.
  static constexpr int MIN_COMPUTE_CAPABILITY = 80;
  // We only support HIP versions 12.0 and above.
  static constexpr int MIN_DRIVER_VERSION = 12.0;

  static ErrorOr<const DeviceList &> deviceList();

  const HIPDriver &driver_;
  hipDevice_t device_;
};

class HIPDeviceBuffer : public DeviceBuffer {
public:
  HIPDeviceBuffer(const HIPDeviceContext &context, size_t len, size_t elem_size,
                  hipDeviceptr_t ptr);
  HIPDeviceBuffer(const HIPDeviceContext &context,
                  const HIPDeviceBuffer &parent, size_t len, size_t elem_size,
                  hipDeviceptr_t ptr);
  // TODO: Remove the V1 compatibility.
  HIPDeviceBuffer(const HIPDeviceContext &context, size_t len, size_t elem_size,
                  hipDeviceptr_t ptr, bool owning);
  ~HIPDeviceBuffer() override;

  hipDeviceptr_t hip_ptr() const;
  void *ptr() const override;
  // TODO: Remove the V1 compatibility.
  void *releasePtr() const override;

  ErrorOr<DeviceBufferRef> createSubBuffer(size_t offset, size_t len,
                                           size_t elem_size) const override;

  static const HIPDeviceBuffer &cast(const DeviceBuffer &buf);

private:
  // TODO: Remove the V1 compatibility.
  mutable hipDeviceptr_t dptr_;
  mutable bool owning_ = true;
};

class HIPDeviceStream : public DeviceStream {
public:
  HIPDeviceStream(const HIPDeviceContext &context, hipStream_t stream);
  ~HIPDeviceStream() override = default;

  virtual ErrorOrSuccess synchronize() const override;

  hipStream_t hip_stream() const;

private:
  hipStream_t stream_;
};

class HIPDeviceFunctionArguments : public DeviceFunction::Arguments {
public:
  HIPDeviceFunctionArguments(const HIPDeviceFunction &fn);
  ~HIPDeviceFunctionArguments() override = default;

  void pushArg(int index, const uint64_t &val) override;
  void pushArg(int index, const DeviceBuffer &val) override;

  std::vector<void *> hip_args() const;

  static const HIPDeviceFunctionArguments &
  cast(const DeviceFunction::Arguments &args);

private:
  std::vector<void *> arg_values_;
  std::vector<DeviceBufferRef> refs_;
};

class HIPDeviceFunction : public DeviceFunction {
public:
  HIPDeviceFunction(const HIPDeviceContext &context, std::string name,
                    const DeviceFunction::SignatureBuilder &signature,
                    hipModule_t module, hipFunction_t function);
  ~HIPDeviceFunction() override;

  const std::string &name() const override;

  ErrorOrSuccess copyToNamedConstant(const char *name, const void *data,
                                     size_t byte_size) const override;

  ErrorOr<int32_t> TEST_ONLY_getAttribute(int32_t code) const override;

  DeviceFunction::Arguments *
  allocateArguments(const DeviceFunction &fn) const override;

  ErrorOrSuccess hip_callKernel(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                                uint32_t blockX, uint32_t blockY,
                                uint32_t blockZ, uint32_t sharedMemBytes,
                                const HIPDeviceFunctionArguments &args,
                                hipStream_t stream) const;

  ErrorOrSuccess hip_callKernelDirect(uint32_t gridX, uint32_t gridY,
                                      uint32_t gridZ, uint32_t blockX,
                                      uint32_t blockY, uint32_t blockZ,
                                      uint32_t sharedMemBytes, void **args,
                                      hipStream_t stream) const;

  static const HIPDeviceFunction &cast(const DeviceFunction &func);

private:
  hipModule_t module_;
  hipFunction_t function_;
  std::string name_;
};

class HIPDeviceTimer : public DeviceTimer {
public:
  HIPDeviceTimer(const HIPDeviceContext &context, hipEvent_t start,
                 hipEvent_t stop);
  ~HIPDeviceTimer() override;

  hipEvent_t hip_start_event() const;
  hipEvent_t hip_stop_event() const;

  static const HIPDeviceTimer &cast(const DeviceTimer &timer);

private:
  hipEvent_t start_;
  hipEvent_t stop_;
};

class HIPDeviceContext : public DeviceContext {
public:
  HIPDeviceContext(const HIPDevice &device, hipCtx_t context,
                   hipStream_t stream, void *buffer_cache_base,
                   size_t buffer_cache_size);
  ~HIPDeviceContext() override;

  // Host memory allocation.
  ErrorOr<void *> malloc_host(size_t size) const override;
  ErrorOrSuccess free_host(void *ptr) const override;

  // DeviceBuffer allocation.
  ErrorOr<DeviceBufferRef> createBuffer(size_t len, size_t elem_size,
                                        SyncMode mode) const override;
  DeviceBufferRef createBufferOwning(void *ptr, size_t len, size_t elem_size,
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
                           size_t elem_size, SyncMode mode) const override;

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

  const HIPDriver &hip_driver() const;
  hipStream_t hip_stream() const;

  BufferCache &buffer_cache() const;

  HIPDriver::Result setCurrent() const;

  static const HIPDeviceContext &cast(const DeviceContext &ctx);

private:
  const HIPDriver &driver_;
  hipCtx_t context_;
  hipStream_t stream_;

  mutable std::mutex mu_;
  mutable llvm::StringMap<HIPDeviceFunctionRef> func_map_;

  mutable BufferCache buf_cache_;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_HIP_HIPDEVICECONTEXT_H
