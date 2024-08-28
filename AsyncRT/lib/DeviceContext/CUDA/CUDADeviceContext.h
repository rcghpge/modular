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
#include "AsyncRT/lib/DeviceContext/CUDA/CUDADriver.h"
#include "Support/RCRef.h"

#include <string>

namespace M::AsyncRT {

class CUDADevice;
class CUDADeviceBuffer;
class CUDADeviceContext;

using CUDADeviceRef = RCRef<const CUDADevice>;
using CUDADeviceBufferRef = RCRef<const CUDADeviceBuffer>;
using CUDADeviceContextRef = RCRef<const CUDADeviceContext>;

class CUDADevice : public Device {
public:
  CUDADevice();
  ~CUDADevice() override;

  CUDADriver *driver() const;

  CUdevice id() const;

  std::string name() const override;

private:
  CUDADriver *driver_;
  CUdevice id_;
};

class CUDADeviceBuffer : public DeviceBuffer {
public:
  CUDADeviceBuffer(CUDADeviceContextRef context, size_t len, size_t elem_size,
                   CUdeviceptr ptr);
  ~CUDADeviceBuffer() override;

  CUdeviceptr cuda_ptr() const;

  static const CUDADeviceBuffer *cast(const DeviceBufferRef &ref);

private:
  CUdeviceptr dptr_;
};

class CUDADeviceContext : public DeviceContext {
public:
  CUDADeviceContext(CUDADeviceRef device, CUcontext context, CUstream stream);
  ~CUDADeviceContext() override;

  // Host memory allocation.
  void *malloc_host(size_t size) const override;
  void free_host(void *ptr) const override;

  // DeviceBuffer allocation.
  DeviceBufferRef createBuffer(size_t len, size_t elem_size) const override;

  // DeviceBuffer copies.
  void enqueueCopyToDevice(const DeviceBufferRef &dst,
                           const void *src) const override;
  void enqueueCopyFromDevice(void *dst,
                             const DeviceBufferRef &src) const override;
  void enqueueCopyDeviceToDevice(const DeviceBufferRef &dst,
                                 const DeviceBufferRef &src) const override;

  // Synchronize asynchronous operations.
  void synchronize() const override;

  const CUDADevice *cuda_device() const;
  CUstream cuda_stream() const;

  static const CUDADeviceContext *cast(const DeviceContextRef &ref);

  static CUDADeviceContextRef create();

private:
  CUDADriver *driver_;
  CUcontext context_;
  CUstream stream_;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_CUDA_CUDADEVICECONTEXT_H
