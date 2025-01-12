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
  static constexpr API api = Device::hipAPI;

  using DeviceList = std::vector<HIPDeviceRef>;
  HIPDevice(const HIPDriver &driver, int id, hipDevice_t device,
            std::string name, size_t allocAlignment);
  ~HIPDevice() override = default;

  const HIPDriver &hipDriver() const;
  hipDevice_t hipDevice() const;

  ErrorOr<int> getDriverVersion() const override;
  ErrorOrSuccess isCompatibleWithMAX() const override;
  ErrorOr<int> computeCapability() const override;

  ErrorOr<bool> canAccess(const Device &peer) const override;
  ErrorOr<int> getAttribute(Attribute attr) const override;

  int deviceCount() const override;

  ErrorOr<DeviceContextRef>
  createContext(size_t bufferCacheSize) const override;

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
  HIPDeviceBuffer(const HIPDeviceContext &context, size_t byteSize,
                  hipDeviceptr_t ptr);
  HIPDeviceBuffer(const HIPDeviceContext &context,
                  const HIPDeviceBuffer &parent, size_t byteSize,
                  hipDeviceptr_t ptr);
  // TODO: Remove the V1 compatibility.
  HIPDeviceBuffer(const HIPDeviceContext &context, size_t byteSize,
                  hipDeviceptr_t ptr, bool owning);
  ~HIPDeviceBuffer() override;

  hipDeviceptr_t hipPtr() const;
  void *ptr() const override;
  // TODO: Remove the V1 compatibility.
  void *releasePtr() const override;

  ErrorOr<DeviceBufferRef> createSubBuffer(size_t byteOffset,
                                           size_t byteSize) const override;

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

  hipStream_t hipStream() const;

private:
  hipStream_t stream_;
};

class HIPDeviceFunctionArguments : public DeviceFunction::Arguments {
public:
  HIPDeviceFunctionArguments(const HIPDeviceFunction &fn);
  ~HIPDeviceFunctionArguments() override = default;

  void pushArg(int index, const uint64_t &val) override;
  void pushArg(int index, const DeviceBuffer &val) override;

  std::vector<void *> hipArgs() const;

  static const HIPDeviceFunctionArguments &
  cast(const DeviceFunction::Arguments &args);

private:
  std::vector<void *> argValues_;
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
                                     size_t byteSize) const override;

  ErrorOr<int32_t> getAttribute(int32_t code) const override;

  DeviceFunction::Arguments *
  allocateArguments(const DeviceFunction &fn) const override;

  ErrorOrSuccess hipCallKernel(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                               uint32_t blockX, uint32_t blockY,
                               uint32_t blockZ, uint32_t sharedMemBytes,
                               const HIPDeviceFunctionArguments &args,
                               hipStream_t stream) const;

  ErrorOrSuccess hipCallKernelDirect(uint32_t gridX, uint32_t gridY,
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

  hipEvent_t hipStartEvent() const;
  hipEvent_t hipStopEvent() const;

  static const HIPDeviceTimer &cast(const DeviceTimer &timer);

private:
  hipEvent_t start_;
  hipEvent_t stop_;
};

class HIPDeviceContext : public DeviceContext {
public:
  HIPDeviceContext(const HIPDevice &device, hipCtx_t context,
                   hipStream_t stream, void *bufferCacheBase,
                   size_t bufferCacheSize);
  ~HIPDeviceContext() override;

  // Host memory allocation.
  ErrorOr<void *> mallocHost(size_t size) const override;
  ErrorOrSuccess freeHost(void *ptr) const override;

  // DeviceBuffer allocation.
  ErrorOr<DeviceBufferRef> allocBuffer(size_t byteSize,
                                       SyncMode mode) const override;
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
                           size_t valSize, SyncMode mode) const override;

  // DeviceFunction loading and execution.
  ErrorOr<DeviceFunctionRef>
  loadFunction(const char *moduleName, const char *functionName,
               const DeviceFunction::SignatureBuilder &signature,
               const void *data, int32_t maxDynamicSharedBytes,
               const char *debugLevel,
               int32_t optimizationLevel) const override;
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

  ErrorOr<DeviceStreamRef> stream() const override;

  // Synchronize asynchronous operations.
  ErrorOrSuccess synchronize() const override;

  ErrorOr<DeviceTimerRef> startTimer() const override;
  ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) const override;

  ErrorOrSuccess enablePeerAccess(const DeviceContext &peer) const override;

  ErrorOr<MemInfo> getMemoryInfo() const override;

  const HIPDriver &hipDriver() const;
  hipStream_t hipStream() const;

  BufferCache &bufferCache() const;

  HIPDriver::Result setCurrent() const;

  static const HIPDeviceContext &cast(const DeviceContext &ctx);

private:
  const HIPDriver &driver_;
  hipCtx_t context_;
  hipStream_t stream_;

  mutable std::mutex mu_;
  mutable llvm::StringMap<HIPDeviceFunctionRef> funcMap_;

  mutable BufferCache bufCache_;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_HIP_HIPDEVICECONTEXT_H
