//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the CUDA device context.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_LIB_DEVICECONTEXT_CUDA_CUDADEVICECONTEXT_H
#define ASYNCRT_LIB_DEVICECONTEXT_CUDA_CUDADEVICECONTEXT_H

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "CUDADriver.h"
#include "NVML.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "llvm/ADT/StringMap.h"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace M::AsyncRT {

class CUDADevice;
class CUDADeviceContext;
class CUDADeviceEvent;
class CUDAModuleFunction;
class CUDADeviceFunction;
class CUDADeviceFunctionArguments;
class CUDADeviceMulticastBuffer;
class CUDADeviceStream;
class CUDADeviceTimer;

using CUDADeviceRef = RCRef<const CUDADevice>;
using CUDADeviceContextRef = RCRef<const CUDADeviceContext>;
using CUDADeviceEventRef = RCRef<const CUDADeviceEvent>;
using CUDAModuleFunctionRef = RCRef<const CUDAModuleFunction>;
using CUDADeviceFunctionRef = RCRef<const CUDADeviceFunction>;
using CUDADeviceFunctionArgumentsRef = RCRef<const CUDADeviceFunctionArguments>;
using CUDADeviceMulticastBufferRef = RCRef<const CUDADeviceMulticastBuffer>;
using CUDADeviceStreamRef = RCRef<const CUDADeviceStream>;
using CUDADeviceTimerRef = RCRef<const CUDADeviceTimer>;

class CUDADevice : public Device {
public:
  static constexpr API api = Device::cudaAPI;

  using DeviceList = std::vector<CUDADevice *>;

  CUDADevice(const CUDADriver &driver, int id, CUdevice device,
             CUcontext context, std::string name, Device::MemInfo memInfo,
             size_t allocAlignment);
  ~CUDADevice() override;

  const CUDADriver &cudaDriver() const;
  CUdevice cudaDevice() const;
  ErrorOr<CUcontext> cudaContext() const;

  ErrorOr<int> getApiVersion() const override;
  ErrorOr<DriverVersion &> getDriverVersion() const override;
  ErrorOrSuccess isCompatible() const override;
  ErrorOr<int> computeCapability() const override;

  ErrorOr<bool> canAccess(const Device &peer) const override;
  ErrorOr<int> getAttribute(Attribute attr) const override;

  ErrorOr<bool> supportsMulticast() const override;

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

  static const CUDADevice &cast(const Device &device);

  static ErrorOr<int> numberOfGPUs();
  static ErrorOr<const CUDADevice &> get(int id);

  ErrorOr<CUevent> eventCachePop() const;
  void eventCachePush(CUevent event) const;

private:
  void eventCacheClear() const;

  // We only support CUDA architectures sm_75 and above.
  static constexpr int MIN_COMPUTE_CAPABILITY = 75;

  // We only support CUDA 12.5 and NVIDIA driver 555 and above.
  // We should support CUDA 12.0 and later, but some users reported problems
  // with NVIDIA drivers earlier than 555. We don't have a good way to check the
  // NVIDIA driver version, but we can check the CUDA version.
  //
  // It is a bit difficult to find the mapping of NVIDIA driver version to CUDA
  // version. This table is the best I can find:
  // https://docs.nvidia.com/deploy/cuda-compatibility/index.html#use-the-right-compat-package
  //
  // So following this table, we assume that CUDA 12.5 corresponds to NVIDIA
  // driver 555.
  //
  // On AWS and GCP, the CUDA driver version is 12.4 and the NVIDIA driver
  // version is 550 by default. Since we want to easily deploy our cluster
  // product on those cloud providers, we use CUDA 12.4 and NVIDIA driver 550
  // as the minimum version.
  static constexpr int MIN_CUDA_DRIVER_VERSION = 12040;
  static constexpr int MIN_NVIDIA_DRIVER_VERSION = 550;

  const CUDADriver &driver_;
  const CUdevice device_;
  // Device context to be shared across all `CUDADeviceContext` objects. We
  // initialize this with the primary device context for the device to be able
  // to share resources with libraries using the CUDA Runtime API.
  CUcontext context_;

  // Mutex to protect mutable data.
  mutable std::mutex mu_;
  // Non-timing enabled events which can be reused for record/wait pairs.
  mutable std::vector<CUevent> eventCache_;
};

class CUDADeviceMulticastBuffer : public DeviceMulticastBuffer {
public:
  struct DeviceToBuffer {
    const CUdevice device_;
    const CUmemGenericAllocationHandle handle_;
    const CUdeviceptr ptr_;
  };

  CUDADeviceMulticastBuffer(
      size_t byteSize, size_t paddedByteSize, const CUDADriver &driver,
      CUmemGenericAllocationHandle multicastHandle, CUdeviceptr multicastPtr,
      std::vector<CUDADeviceMulticastBuffer::DeviceToBuffer> unicastBuffers);

  ~CUDADeviceMulticastBuffer() override;

  ErrorOr<DeviceBufferRef>
  multicastBuffer(const DeviceContext &ctx) const override;
  ErrorOr<DeviceBufferRef>
  unicastBufferFor(const DeviceContext &ctx) const override;

  size_t numberOfDevices() const override;

  static ErrorOr<DeviceMulticastBufferRef>
  allocate(const std::vector<const Device *> &devices, size_t byteSize);

private:
  const CUDADriver &driver_;
  CUmemGenericAllocationHandle multicastHandle_;
  CUdeviceptr multicastPtr_;
  size_t paddedByteSize_;
  std::vector<DeviceToBuffer> unicastBuffers_;
};

class CUDADeviceEvent : public DeviceEvent {
public:
  CUDADeviceEvent(const CUDADevice &device, CUevent event);
  ~CUDADeviceEvent() override;

  ErrorOrSuccess synchronize() const override;

  ErrorOr<bool> isReady() const override;

  CUevent cudaEvent() const;

  static const CUDADeviceEvent &cast(const DeviceEvent &event);

private:
  CUevent event_;
};

class CUDADeviceStream : public DeviceStream {
public:
  CUDADeviceStream(const CUDADevice &device, CUstream stream);
  ~CUDADeviceStream() override = default;

  ErrorOrSuccess synchronize() const override;

  ErrorOr<DeviceEventRef> enqueueEvent() const override;

  ErrorOrSuccess enqueueWaitForEvent(const DeviceEvent &event) const override;

  void *ptr() const override;

  CUstream cudaStream() const;

  static const CUDADeviceStream *verified_cast(const DeviceStream *stream);

private:
  CUstream stream_;
};

class CUDADeviceFunctionArguments : public DeviceFunction::Arguments {
public:
  CUDADeviceFunctionArguments(const CUDADeviceFunction &fn);
  ~CUDADeviceFunctionArguments() override = default;

  void pushArg(int index, const uint64_t &val) override;
  void pushArg(int index, const DeviceBuffer &val) override;

  std::vector<void *> cudaArgs() const;

  static const CUDADeviceFunctionArguments &
  cast(const DeviceFunction::Arguments &args);

private:
  std::vector<void *> argValues_;
  std::vector<DeviceBufferRef> refs_;
};

class CUDAModuleFunction : public DeviceFunctionCacheEntry {
public:
  CUDAModuleFunction(const std::string name, const CUDADriver &driver,
                     CUmodule module, CUfunction function);
  ~CUDAModuleFunction();

  const CUDADriver &cudaDriver() const;
  CUmodule cudaModule() const;
  CUfunction cudaFunction() const;

  static const CUDAModuleFunction *cast(const DeviceFunctionCacheEntry *entry);

private:
  const CUDADriver &driver_;
  const CUmodule module_;
  const CUfunction function_;
};

class CUDADeviceFunction : public DeviceFunction {
public:
  CUDADeviceFunction(const CUDADeviceContext &context,
                     const DeviceFunction::SignatureBuilder &signature,
                     CUDAModuleFunctionRef modFunc);
  ~CUDADeviceFunction() override;

  const std::string &name() const override;

  ErrorOrSuccess copyToNamedConstant(const char *name, const void *data,
                                     size_t byteSize) const override;

  ErrorOr<int32_t> getAttribute(int32_t code) const override;

  DeviceFunction::Arguments *
  allocateArguments(const DeviceFunction &fn) const override;

  ErrorOrSuccess cuda_callKernel(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                                 uint32_t blockX, uint32_t blockY,
                                 uint32_t blockZ, uint32_t sharedMemBytes,
                                 const CUDADeviceFunctionArguments &args,
                                 CUstream stream) const;
  ErrorOrSuccess
  cudaCallKernelDirect(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                       uint32_t blockX, uint32_t blockY, uint32_t blockZ,
                       uint32_t sharedMemBytes, CUlaunchAttribute *attrs,
                       uint32_t numAttrs, void **args, CUstream stream) const;

  CUmodule cudaModule() const;

  static const CUDADeviceFunction &cast(const DeviceFunction &func);

private:
  // The moduleFunction_ holds the CUDA specific implementation. Methods on
  // CUDADeviceFunction forward to moduleFunction_ as needed.
  // This layout is needed to break the reference cycle between the function
  // cache in the Device and the cached functions.
  CUDAModuleFunctionRef moduleFunction_;
};

class CUDADeviceTimer : public DeviceTimer {
public:
  CUDADeviceTimer(const CUDADeviceContext &context, CUevent start,
                  CUevent stop);
  ~CUDADeviceTimer() override;

  CUevent cudaStartEvent() const;
  CUevent cudaStopEvent() const;

  static const CUDADeviceTimer &cast(const DeviceTimer &timer);

private:
  CUevent start_;
  CUevent stop_;
};

class CUDADeviceContext : public DeviceContext {
public:
  CUDADeviceContext(const CUDADevice &device, CUcontext context,
                    CUstream stream);
  ~CUDADeviceContext() override;

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

  // DeviceFunction compiling.
  ErrorOr<M::BufferRef>
  compileFunction(StringRef ir, StringRef debugLevel, int32_t optimizationLevel,
                  StringRef targetAccelerator) const override;

  // DeviceFunction compiling.
  ErrorOr<M::BufferRef>
  compileFunction_v2(StringRef ir, StringRef debugLevel,
                     int32_t optimizationLevel) const override;

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

  ErrorOr<DeviceEventRef> enqueueEvent() const override;
  ErrorOrSuccess enqueueWaitFor(const DeviceEvent &event) const override;
  ErrorOrSuccess enqueueWaitFor(const DeviceContext &ctx) const override;

  ErrorOr<DeviceTimerRef> startTimer() const override;
  ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) const override;

  /// Makes all allocations on `peer` accessible from this context.
  ErrorOrSuccess enablePeerAccess(const DeviceContext &peer) const override;

  ErrorOr<Device::MemInfo> getMemoryInfo() const override;

  const CUDADriver &cudaDriver() const;
  CUcontext cudaContext() const;
  CUstream cudaStream() const;

  ErrorOrSuccess cudaTensorMapEncodeTiled(
      void *tensorMap, CUtensorMapDataType tensorDataType, uint32_t tensorRank,
      const DeviceBuffer &globalAddress, const uint64_t *globalDim,
      const uint64_t *globalStrides, const uint32_t *boxDim,
      const uint32_t *elementStrides, CUtensorMapInterleave interleave,
      CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
      CUtensorMapFloatOOBfill oobFill) const;

  static const CUDADeviceContext &cast(const DeviceContext &ctx);
  static const CUDADeviceContext *verified_cast(const DeviceContext *ctx);

private:
  const CUDADriver &driver_;
  const CUcontext context_;
  const CUstream stream_;

  bool logLoadFunction_ = false;
  bool logVerbose_ = false;

  ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                          Device::AllocMode mode) const override;
  ErrorOrSuccess memFree(void *ptr, bool isHost) const override;

  void *driverContext() const override;

  MMapFileImplResult
  memmapFileImpl(const std::filesystem::path &filename) const override;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_CUDA_CUDADEVICECONTEXT_H
