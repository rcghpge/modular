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
  CpuDeviceBuffer(const CpuDeviceContext &context, size_t len, size_t elem_size,
                  void *ptr);
  ~CpuDeviceBuffer() override;

  void *cpu_ptr() const;

  static const CpuDeviceBuffer &cast(const DeviceBuffer &buf);

private:
  void *ptr_;
};

class CpuDeviceContext : public DeviceContext {
public:
  CpuDeviceContext(const CpuDevice &device);
  ~CpuDeviceContext() override;

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

  // Synchronize asynchronous operations.
  ErrorOrSuccess synchronize() const override;

  static const CpuDeviceContext &cast(const DeviceContext &ctx);

  static ErrorOr<CpuDeviceContextRef> create();
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_CPU_CPUDEVICECONTEXT_H
