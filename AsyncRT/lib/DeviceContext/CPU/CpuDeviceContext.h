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
#include "Support/RCRef.h"

#include <stddef.h>
#include <string>

namespace M::AsyncRT {

class CpuDevice;
class CpuDeviceBuffer;
class CpuDeviceContext;

using CpuDeviceRef = RCRef<const CpuDevice>;
using CpuDeviceBufferRef = RCRef<const CpuDeviceBuffer>;
using CpuDeviceContextRef = RCRef<const CpuDeviceContext>;

class CpuDevice : public Device {
public:
  CpuDevice();
  ~CpuDevice() override;

  std::string name() const override;
};

class CpuDeviceBuffer : public DeviceBuffer {
public:
  CpuDeviceBuffer(CpuDeviceContextRef context, size_t len, size_t elem_size,
                  void *ptr);
  ~CpuDeviceBuffer() override;

  void *cpu_ptr() const;

private:
  void *ptr_;
};

class CpuDeviceContext : public DeviceContext {
public:
  CpuDeviceContext(CpuDeviceRef device);
  ~CpuDeviceContext() override;

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

  static CpuDeviceContextRef create();
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_CPU_CPUDEVICECONTEXT_H
