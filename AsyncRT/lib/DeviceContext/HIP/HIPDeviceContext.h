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
#include "HIPDriver.h"
#include "Support/ErrorOr.h"
#include "Support/PlatformUtils.h"
#include "Support/RCRef.h"
#include "llvm/ADT/StringMap.h"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace M::AsyncRT {

class HIPDevice;
class HIPDeviceContext;
class HIPDeviceEvent;
class HIPDeviceFunction;
class HIPDeviceFunctionArguments;
class HIPDeviceStream;
class HIPDeviceTimer;
class HIPModuleFunction;

using HIPDeviceRef = RCRef<const HIPDevice>;
using HIPDeviceContextRef = RCRef<const HIPDeviceContext>;
using HIPDeviceEventRef = RCRef<const HIPDeviceEvent>;
using HIPDeviceFunctionRef = RCRef<const HIPDeviceFunction>;
using HIPDeviceFunctionArgumentsRef = RCRef<const HIPDeviceFunctionArguments>;
using HIPDeviceStreamRef = RCRef<const HIPDeviceStream>;
using HIPDeviceTimerRef = RCRef<const HIPDeviceTimer>;
using HIPModuleFunctionRef = RCRef<const HIPModuleFunction>;

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
  ErrorOrSuccess isCompatible() const override;
  ErrorOr<int> computeCapability() const override;

  ErrorOr<bool> canAccess(const Device &peer) const override;
  ErrorOr<int> getAttribute(Attribute attr) const override;

  int deviceCount() const override;

  ErrorOr<DeviceContextRef>
  createContext(size_t bufferCacheSize) const override;

  ErrorOr<DeviceStreamRef>
  createExternalStream(void *externalStream) const override;

  HIPDriver::Result setCurrent() const;

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

class HIPDeviceEvent : public DeviceEvent {
public:
  HIPDeviceEvent(const HIPDevice &device, hipEvent_t event);
  ~HIPDeviceEvent() override;

  ErrorOrSuccess synchronize() const override;

  hipEvent_t hipEvent() const;

  static const HIPDeviceEvent &cast(const DeviceEvent &event);

private:
  hipEvent_t event_;
};

class HIPDeviceStream : public DeviceStream {
public:
  HIPDeviceStream(const HIPDevice &device, hipStream_t stream);
  ~HIPDeviceStream() override = default;

  ErrorOrSuccess synchronize() const override;

  ErrorOr<DeviceEventRef> enqueueEvent() const override;

  ErrorOrSuccess enqueueWaitForEvent(const DeviceEvent &event) const override;

  void *ptr() const override;

  hipStream_t hipStream() const;

  static const HIPDeviceStream *verified_cast(const DeviceStream *stream);

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

class HIPModuleFunction : public ReferenceCounted<HIPModuleFunction> {
public:
  HIPModuleFunction(const std::string name, const HIPDriver &driver,
                    hipModule_t module, hipFunction_t function);
  ~HIPModuleFunction();

  const std::string &name() const;
  const HIPDriver &hipDriver() const;
  hipModule_t hipModule() const;
  hipFunction_t hipFunction() const;

private:
  const HIPDriver &driver_;
  hipModule_t module_;
  hipFunction_t function_;
  std::string name_;
};

class HIPDeviceFunction : public DeviceFunction {
public:
  HIPDeviceFunction(const HIPDeviceContext &context,
                    const DeviceFunction::SignatureBuilder &signature,
                    HIPModuleFunctionRef modFunc);
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
  // The moduleFunction_ holds the HIP specific implementation. Methods on
  // HIPDeviceFunction forward to moduleFunction_ as needed.
  // This layout is needed to break the reference cycle between the funcMap_
  // cache in the HIPDeviceContext and the cached functions.
  HIPModuleFunctionRef moduleFunction_;
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
                   hipStream_t stream, size_t cacheChunkSize,
                   size_t maxCacheSize);
  ~HIPDeviceContext() override;

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
  loadFunction(StringRef moduleName, StringRef functionName,
               const DeviceFunction::SignatureBuilder &signature,
               const void *data, int32_t maxDynamicSharedBytes,
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

  ErrorOr<DeviceEventRef> enqueueEvent() const override;
  ErrorOrSuccess enqueueWaitFor(const DeviceEvent &event) const override;
  ErrorOrSuccess enqueueWaitFor(const DeviceContext &ctx) const override;

  ErrorOr<DeviceTimerRef> startTimer() const override;
  ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) const override;

  ErrorOrSuccess enablePeerAccess(const DeviceContext &peer) const override;

  ErrorOr<MemInfo> getMemoryInfo() const override;

  const HIPDriver &hipDriver() const;
  hipCtx_t hipContext() const;
  hipStream_t hipStream() const;

  HIPDriver::Result setCurrent() const;

  static const HIPDeviceContext &cast(const DeviceContext &ctx);
  static const HIPDeviceContext *verified_cast(const DeviceContext *ctx);

private:
  friend class HIPDeviceBuffer;

  const HIPDriver &driver_;
  hipCtx_t context_;
  hipStream_t stream_;

  mutable std::mutex mu_;
  mutable llvm::StringMap<HIPModuleFunctionRef> funcMap_;

  ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                          AllocMode mode) const override;
  ErrorOrSuccess memFree(void *ptr, bool isHost) const override;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_HIP_HIPDEVICECONTEXT_H
