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
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace M::AsyncRT {

class CUDADevice;
class CUDADeviceBuffer;
class CUDADeviceContext;
class CUDADeviceFunction;
class CUDADeviceFunctionArguments;

using CUDADeviceRef = RCRef<const CUDADevice>;
using CUDADeviceBufferRef = RCRef<const CUDADeviceBuffer>;
using CUDADeviceContextRef = RCRef<const CUDADeviceContext>;
using CUDADeviceFunctionRef = RCRef<const CUDADeviceFunction>;
using CUDADeviceFunctionArgumentsRef = RCRef<const CUDADeviceFunctionArguments>;

class CUDADevice : public Device {
public:
  CUDADevice(const CUDADriver &driver, CUdevice id);
  ~CUDADevice() override = default;

  const CUDADriver &driver() const;

  int id() const override;

  std::string name() const override;

private:
  const CUDADriver &driver_;
  CUdevice id_;
};

class CUDADeviceBuffer : public DeviceBuffer {
public:
  CUDADeviceBuffer(const CUDADeviceContext &context, size_t len,
                   size_t elem_size, CUdeviceptr ptr);
  ~CUDADeviceBuffer() override;

  CUdeviceptr cuda_ptr() const;

  static const CUDADeviceBuffer &cast(const DeviceBuffer &buf);

private:
  CUdeviceptr dptr_;
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

class CUDADeviceFunction : public DeviceFunction {
public:
  CUDADeviceFunction(const CUDADeviceContext &context, std::string name,
                     const DeviceFunction::SignatureBuilder &signature,
                     CUmodule module, CUfunction function);
  ~CUDADeviceFunction() override;

  const std::string &name() const override;

  DeviceFunction::Arguments *
  allocateArguments(const DeviceFunction &fn) const override;

  ErrorOrSuccess cuda_callKernel(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                                 uint32_t blockX, uint32_t blockY,
                                 uint32_t blockZ,
                                 const CUDADeviceFunctionArguments &args,
                                 CUstream stream) const;

  static const CUDADeviceFunction &cast(const DeviceFunction &func);

private:
  CUmodule module_;
  CUfunction function_;
  std::string name_;
};

class CUDADeviceContext : public DeviceContext {
public:
  CUDADeviceContext(const CUDADevice &device, CUcontext context,
                    CUstream stream);
  ~CUDADeviceContext() override;

  // Host memory allocation.
  ErrorOr<void *> malloc_host(size_t size) const override;
  ErrorOrSuccess free_host(void *ptr) const override;

  // DeviceBuffer allocation.
  ErrorOr<DeviceBufferRef> createBuffer(size_t len,
                                        size_t elem_size) const override;

  // DeviceBuffer copies.
  ErrorOrSuccess enqueueCopyToDevice(const DeviceBuffer &dst,
                                     const void *src) const override;
  ErrorOrSuccess enqueueCopyFromDevice(void *dst,
                                       const DeviceBuffer &src) const override;
  ErrorOrSuccess
  enqueueCopyDeviceToDevice(const DeviceBuffer &dst,
                            const DeviceBuffer &src) const override;

  // DeviceFunction compilation and execution.
  ErrorOr<DeviceFunctionRef>
  compileFunction(const char *function_name,
                  const DeviceFunction::SignatureBuilder &signature,
                  const void *data) const override;
  ErrorOrSuccess
  enqueueFunctionExec(const DeviceFunction &func, uint32_t gridX,
                      uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                      uint32_t blockY, uint32_t blockZ,
                      const DeviceFunction::Arguments &args) const override;

  // Synchronize asynchronous operations.
  ErrorOrSuccess synchronize() const override;

  const CUDADriver &cuda_driver() const;
  CUstream cuda_stream() const;

  static const CUDADeviceContext &cast(const DeviceContext &ctx);

  static ErrorOr<CUDADeviceContextRef> create(int id);

private:
  const CUDADriver &driver_;
  CUcontext context_;
  CUstream stream_;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_CUDA_CUDADEVICECONTEXT_H
