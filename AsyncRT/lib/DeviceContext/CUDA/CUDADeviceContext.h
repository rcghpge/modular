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
#include "AsyncRT/lib/DeviceContext/BufferCache.h"
#include "CUDADriver.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "Support/include/Support/ReferenceCounted.h"
#include "llvm/ADT/StringMap.h"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace M::AsyncRT {

class CUDADevice;
class CUDADeviceBuffer;
class CUDADeviceContext;
class CUDAModuleFunction;
class CUDADeviceFunction;
class CUDADeviceFunctionArguments;
class CUDADeviceStream;
class CUDADeviceTimer;

using CUDADeviceRef = RCRef<const CUDADevice>;
using CUDADeviceBufferRef = RCRef<const CUDADeviceBuffer>;
using CUDADeviceContextRef = RCRef<const CUDADeviceContext>;
using CUDAModuleFunctionRef = RCRef<const CUDAModuleFunction>;
using CUDADeviceFunctionRef = RCRef<const CUDADeviceFunction>;
using CUDADeviceFunctionArgumentsRef = RCRef<const CUDADeviceFunctionArguments>;
using CUDADeviceStreamRef = RCRef<const CUDADeviceStream>;
using CUDADeviceTimerRef = RCRef<const CUDADeviceTimer>;

class CUDADevice : public Device {
public:
  static constexpr API api = Device::cudaAPI;

  using DeviceList = std::vector<CUDADeviceRef>;

  CUDADevice(const CUDADriver &driver, int id, CUdevice device,
             std::string name, size_t alloc_alignment);
  ~CUDADevice() override = default;

  const CUDADriver &cuda_driver() const;
  CUdevice cuda_device() const;

  ErrorOr<int> getDriverVersion() const override;
  ErrorOrSuccess isCompatibleWithMAX() const override;
  ErrorOr<int> computeCapability() const override;

  ErrorOr<MemInfo> getMemoryInfo() const override;

  ErrorOr<bool> canAccess(const Device &peer) const override;
  ErrorOr<int> getAttribute(Attribute attr) const override;

  int deviceCount() const override;

  ErrorOr<DeviceContextRef>
  createContext(size_t buffer_cache_size = 0) const override;

  static const CUDADevice &cast(const Device &device);

  static ErrorOr<const CUDADevice &> get(int id);

private:
  // We only support CUDA architectures sm_80 and above.
  static constexpr int MIN_COMPUTE_CAPABILITY = 80;
  // We only support CUDA versions 12.0 and above.
  static constexpr int MIN_DRIVER_VERSION = 12.0;

  static ErrorOr<const DeviceList &> deviceList();

  const CUDADriver &driver_;
  CUdevice device_;
};

class CUDADeviceBuffer : public DeviceBuffer {
public:
  CUDADeviceBuffer(const CUDADeviceContext &context, size_t byteSize,
                   CUdeviceptr ptr);
  CUDADeviceBuffer(const CUDADeviceContext &context,
                   const CUDADeviceBuffer &parent, size_t byteSize,
                   CUdeviceptr ptr);
  // TODO: Remove the V1 compatibility.
  CUDADeviceBuffer(const CUDADeviceContext &context, size_t byteSize,
                   CUdeviceptr ptr, bool owning);
  ~CUDADeviceBuffer() override;

  CUdeviceptr cuda_ptr() const;
  void *ptr() const override;
  // TODO: Remove the V1 compatibility.
  void *releasePtr() const override;

  ErrorOr<DeviceBufferRef> createSubBuffer(size_t byteOffset,
                                           size_t byteSize) const override;

  static const CUDADeviceBuffer &cast(const DeviceBuffer &buf);

private:
  // TODO: Remove the V1 compatibility.
  mutable CUdeviceptr dptr_;
  mutable bool owning_ = true;
};

class CUDADeviceStream : public DeviceStream {
public:
  CUDADeviceStream(const CUDADeviceContext &context, CUstream stream);
  ~CUDADeviceStream() override = default;

  virtual ErrorOrSuccess synchronize() const override;

  CUstream cuda_stream() const;

private:
  CUstream stream_;
};

class CUDADeviceFunctionArguments : public DeviceFunction::Arguments {
public:
  CUDADeviceFunctionArguments(const CUDADeviceFunction &fn);
  ~CUDADeviceFunctionArguments() override = default;

  void pushArg(int index, const uint64_t &val) override;
  void pushArg(int index, const DeviceBuffer &val) override;

  std::vector<void *> cuda_args() const;

  static const CUDADeviceFunctionArguments &
  cast(const DeviceFunction::Arguments &args);

private:
  std::vector<void *> arg_values_;
  std::vector<DeviceBufferRef> refs_;
};

class CUDAModuleFunction : public ReferenceCounted<CUDAModuleFunction> {
public:
  CUDAModuleFunction(const std::string name, const CUDADriver &driver,
                     CUmodule module, CUfunction function);
  ~CUDAModuleFunction();

  const std::string &name() const;
  const CUDADriver &cuda_driver() const;
  CUmodule cuda_module() const;
  CUfunction cuda_function() const;

private:
  const CUDADriver &driver_;
  CUmodule module_;
  CUfunction function_;
  std::string name_;
};

class CUDADeviceFunction : public DeviceFunction {
public:
  CUDADeviceFunction(const CUDADeviceContext &context,
                     const DeviceFunction::SignatureBuilder &signature,
                     CUDAModuleFunctionRef mod_func);
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
  cuda_callKernelDirect(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                        uint32_t blockX, uint32_t blockY, uint32_t blockZ,
                        uint32_t sharedMemBytes, CUlaunchAttribute *attrs,
                        uint32_t num_attrs, void **args, CUstream stream) const;

  static const CUDADeviceFunction &cast(const DeviceFunction &func);

private:
  // The module_function_ holds the CUDA specific implementation. Methods on
  // CUDADeviceFunction forward to module_function_ as needed.
  CUDAModuleFunctionRef module_function_;
};

class CUDADeviceTimer : public DeviceTimer {
public:
  CUDADeviceTimer(const CUDADeviceContext &context, CUevent start,
                  CUevent stop);
  ~CUDADeviceTimer() override;

  CUevent cuda_start_event() const;
  CUevent cuda_stop_event() const;

  static const CUDADeviceTimer &cast(const DeviceTimer &timer);

private:
  CUevent start_;
  CUevent stop_;
};

class CUDADeviceContext : public DeviceContext {
public:
  CUDADeviceContext(const CUDADevice &device, CUcontext context,
                    CUstream stream, void *buffer_cache_base,
                    size_t buffer_cache_size);
  ~CUDADeviceContext() override;

  // Host memory allocation.
  ErrorOr<void *> malloc_host(size_t size) const override;
  ErrorOrSuccess free_host(void *ptr) const override;

  // DeviceBuffer allocation.
  ErrorOr<DeviceBufferRef> createBuffer(size_t byteSize,
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

  const CUDADriver &cuda_driver() const;
  CUcontext cuda_context() const;
  CUstream cuda_stream() const;

  ErrorOrSuccess cuda_TensorMapEncodeTiled(
      void *tensorMap, CUtensorMapDataType tensorDataType, uint32_t tensorRank,
      const DeviceBuffer &globalAddress, const uint64_t *globalDim,
      const uint64_t *globalStrides, const uint32_t *boxDim,
      const uint32_t *elementStrides, CUtensorMapInterleave interleave,
      CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
      CUtensorMapFloatOOBfill oobFill) const;

  BufferCache &buffer_cache() const;

  CUDADriver::Result setCurrent() const;

  static const CUDADeviceContext &cast(const DeviceContext &ctx);
  static const CUDADeviceContext *verified_cast(const DeviceContext *ctx);

private:
  const CUDADriver &driver_;
  CUcontext context_;
  CUstream stream_;

  // The function cache holds the lower-level CUDAModuleFunction references
  // which do not keep contain a device context reference to avoid reference
  // counting cycles.
  mutable std::mutex mu_;
  mutable llvm::StringMap<CUDAModuleFunctionRef> func_map_;

  mutable BufferCache buf_cache_;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_CUDA_CUDADEVICECONTEXT_H
