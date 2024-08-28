//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the M::AsyncRT::DeviceContext interface.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_DEVICECONTEXT_DEVICECONTEXT_H
#define ASYNCRT_DEVICECONTEXT_DEVICECONTEXT_H

#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"

#include <stddef.h>
#include <string>

namespace M::AsyncRT {

class Device;
class DeviceBuffer;
class DeviceContext;

using DeviceRef = RCRef<const Device>;
using DeviceBufferRef = RCRef<const DeviceBuffer>;
using DeviceContextRef = RCRef<const DeviceContext>;

class Device : public ReferenceCounted<Device> {
public:
  virtual ~Device();

  virtual std::string name() const = 0;
};

class DeviceBuffer : public ReferenceCounted<DeviceBuffer> {
public:
  virtual ~DeviceBuffer();

  const DeviceContextRef &context() const;

  size_t len() const;
  size_t elem_size() const;

protected:
  DeviceBuffer(DeviceContextRef context, size_t len, size_t elem_size);

private:
  DeviceContextRef context_;
  size_t len_;
  size_t elem_size_;
};

class DeviceContext : public ReferenceCounted<DeviceContext> {
public:
  virtual ~DeviceContext();

  const Device *device() const;

  // Host memory allocation.
  virtual void *malloc_host(size_t size) const = 0;
  virtual void free_host(void *ptr) const = 0;

  // DeviceBuffer allocation.
  virtual DeviceBufferRef createBuffer(size_t len, size_t elem_size) const = 0;

  // DeviceBuffer copies.
  virtual void enqueueCopyToDevice(const DeviceBufferRef &dst,
                                   const void *src) const = 0;
  virtual void enqueueCopyFromDevice(void *dst,
                                     const DeviceBufferRef &src) const = 0;
  virtual void enqueueCopyDeviceToDevice(const DeviceBufferRef &dst,
                                         const DeviceBufferRef &src) const = 0;

  // Synchronize asynchronous operations.
  virtual void synchronize() const = 0;

  static DeviceContextRef create(const char *kind);

protected:
  DeviceContext(DeviceRef device);

private:
  DeviceRef device_;

  // Explicitly mark copy and assignment as unsupported.
  DeviceContext(const DeviceContext &other) = delete;
  DeviceContext(DeviceContext &&other) = delete;
  DeviceContext &operator=(const DeviceContext &) = delete;
  DeviceContext &operator=(DeviceContext &&other) = delete;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_DEVICECONTEXT_DEVICECONTEXT_H
