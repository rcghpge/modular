//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the HIP device context.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_HIP_HIPDEVICECONTEXT_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_HIP_HIPDEVICECONTEXT_H

#include "HIPDriver.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"
#include "Support/PlatformUtils.h"
#include "Support/RCRef.h"
#include "llvm/ADT/StringMap.h"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace M::Driver {

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

  using DeviceList = std::vector<HIPDevice *>;
  HIPDevice(const HIPDriver &driver, int id, hipDevice_t device,
            std::string name, std::string arch, MemInfo memInfo,
            size_t allocAlignment, int64_t deviceMemoryLimit = -1);
  ~HIPDevice() override;

  ErrorOr<const char *> getArchName() const override;
  static ErrorOr<std::string> getArchName(hipDevice_t device = 0);

  const HIPDriver &hipDriver() const;
  hipDevice_t hipDevice() const;

  ErrorOr<int> getApiVersion() const override;
  ErrorOr<DriverVersion &> getDriverVersion() const override;
  ErrorOrSuccess isCompatible() const override;
  ErrorOr<int> computeCapability() const override;

  ErrorOr<bool> canAccess(const Device &peer) const override;

  // Enable peer access between this device and the peer device.
  ErrorOrSuccess enablePeerAccess(const Device &peer) const override;

  ErrorOr<int> getAttribute(Attribute attr) const override;

  static ErrorOr<int> numberOfGPUs();
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

  const CompilationDevice *compilationDevice() const override {
    return nullptr; // HIP device does not have a compilation device yet
  }

  static const HIPDevice &cast(const Device &device);
  static const HIPDevice *verified_cast(const Device *device);

  static ErrorOr<const HIPDevice &> get(int id, int64_t deviceMemoryLimit = -1);

private:
  // We only support HIP architectures sm_80 and above.
  static constexpr int MIN_COMPUTE_CAPABILITY = 80;

  // We only support HIP versions 6.3.30000 and above.
  static constexpr int MIN_HIP_DRIVER_VERSION_MAJOR = 6;
  static constexpr int MIN_HIP_DRIVER_VERSION_MINOR = 3;
  static constexpr int MIN_HIP_DRIVER_VERSION_PATCH = 30000;

  const std::string arch_;
  const HIPDriver &driver_;
  hipDevice_t device_;
};

class HIPDeviceEvent : public DeviceEvent {
public:
  HIPDeviceEvent(const HIPDevice &device, hipEvent_t event);
  ~HIPDeviceEvent() override;

  ErrorOrSuccess synchronize() const override;

  ErrorOr<bool> isReady() const override;

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

  ErrorOrSuccess eventRecord(const DeviceEvent &event) const override;
  ErrorOrSuccess waitEvent(const DeviceEvent &event,
                           unsigned int flags) const override;

  ErrorOrSuccess enqueueWaitForEvent(const DeviceEvent &event) const override;

  virtual ErrorOrSuccess
  enqueueFunctionExecDirect(const DeviceFunction &func, uint32_t gridX,
                            uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                            uint32_t blockY, uint32_t blockZ,
                            uint32_t sharedMemBytes, void *attrs,
                            uint32_t numAttrs, void **args) const override;

  void *ptr() const override;

  hipStream_t hipStream() const;

  static const HIPDeviceStream *verified_cast(const DeviceStream *stream);

private:
  hipStream_t stream_;
};

class HIPModuleFunction : public DeviceFunctionCacheEntry {
public:
  HIPModuleFunction(const std::string name, const HIPDriver &driver,
                    hipModule_t module, hipFunction_t function);
  ~HIPModuleFunction();

  const HIPDriver &hipDriver() const;
  hipModule_t hipModule() const;
  hipFunction_t hipFunction() const;

  static const HIPModuleFunction *cast(const DeviceFunctionCacheEntry *entry);

private:
  const HIPDriver &driver_;
  hipModule_t module_;
  hipFunction_t function_;
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

  ErrorOrSuccess hipCallKernelDirect(uint32_t gridX, uint32_t gridY,
                                     uint32_t gridZ, uint32_t blockX,
                                     uint32_t blockY, uint32_t blockZ,
                                     uint32_t sharedMemBytes, void **args,
                                     hipStream_t stream) const;

  static const HIPDeviceFunction &cast(const DeviceFunction &func);

  ErrorOrSuccess
  occupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, int blockSize,
                                            size_t dynamicSharedMemSize) const;

private:
  // The moduleFunction_ holds the HIP specific implementation. Methods on
  // HIPDeviceFunction forward to moduleFunction_ as needed.
  // This layout is needed to break the reference cycle between the function
  // cache in the Device and the cached functions.
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
  HIPDeviceContext(const HIPDevice &device, hipStream_t stream);
  ~HIPDeviceContext() override;

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

  ErrorOr<DeviceTimerRef> startTimer() const override;
  ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) const override;

  ErrorOrSuccess enablePeerAccess(const DeviceContext &peer) const override;

  ErrorOr<Device::MemInfo> getDeviceMemoryInfo() const override;

  ErrorOrSuccess occupancyMaxActiveBlocksPerMultiprocessor(
      int *numBlocks, const DeviceFunction &func, int blockSize,
      size_t dynamicSharedMemSize) const override;

  const HIPDriver &hipDriver() const;
  hipDevice_t hipDevice() const;
  hipStream_t hipStream() const;

  static const HIPDeviceContext &cast(const DeviceContext &ctx);
  static const HIPDeviceContext *verified_cast(const DeviceContext *ctx);

private:
  friend class HIPDeviceBuffer;

  const HIPDriver &driver_;
  hipDevice_t device_;
  hipStream_t stream_;

  ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                          Device::AllocMode mode) const override;
  ErrorOrSuccess memFree(void *ptr, bool isHost) const override;

  void *driverContext() const override;

  MMapFileImplResult
  memmapFileImpl(const std::filesystem::path &filename) const override;
};

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_HIP_HIPDEVICECONTEXT_H
