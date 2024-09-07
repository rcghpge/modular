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

#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"

#include <cstdint>
#include <stddef.h>
#include <string>

namespace M::AsyncRT {

class Device;
class DeviceBuffer;
class DeviceContext;
class DeviceFunction;
class DeviceFunctionArguments;

using DeviceRef = RCRef<const Device>;
using DeviceBufferRef = RCRef<const DeviceBuffer>;
using DeviceContextRef = RCRef<const DeviceContext>;
using DeviceFunctionRef = RCRef<const DeviceFunction>;
using DeviceFunctionArgumentsRef = RCRef<const DeviceFunctionArguments>;

class Device : public ReferenceCounted<Device> {
public:
  virtual ~Device() = default;

  virtual int id() const = 0;

  virtual std::string name() const = 0;

protected:
  Device() = default;
};

class DeviceBuffer : public ReferenceCounted<DeviceBuffer> {
public:
  virtual ~DeviceBuffer() = default;

  const DeviceContext &context() const;

  size_t len() const;
  size_t elem_size() const;

protected:
  DeviceBuffer(const DeviceContext &context, size_t len, size_t elem_size);

private:
  DeviceContextRef context_;
  size_t len_;
  size_t elem_size_;
};

class DeviceFunctionArguments
    : public ReferenceCounted<DeviceFunctionArguments> {
public:
  virtual ~DeviceFunctionArguments() = default;

protected:
  DeviceFunctionArguments() = default;

  // The implementations of these methods are being called based on the
  // arguments being passed to the DeviceFunction::buildArguments() method.
  // TODO: Add argument types incrementally as needed.
  virtual void pushArg(int index, const uint64_t &val) = 0;
  virtual void pushArg(int index, const DeviceBuffer &val) = 0;

private:
  friend class DeviceFunction;

  template <typename... Ts>
  void build(const Ts &...ts);

  template <int index, typename T0, typename... T1s>
  void pushArgs(const T0 &&t0, const T1s &&...t1s);

  template <int index, typename T0>
  void pushArgs(const T0 &&t0);
};

class DeviceFunction : public ReferenceCounted<DeviceFunction> {
public:
  virtual ~DeviceFunction() = default;

  const DeviceContext &context() const;

  virtual const std::string &name() const = 0;

  template <typename... Ts>
  DeviceFunctionArgumentsRef buildArguments(const Ts &...ts) const;

protected:
  DeviceFunction(const DeviceContext &context);

  virtual DeviceFunctionArguments *allocateArguments() const = 0;

private:
  DeviceContextRef context_;
};

class DeviceContext : public ReferenceCounted<DeviceContext> {
public:
  virtual ~DeviceContext() = default;

  const Device &device() const;

  // Host memory allocation.
  virtual ErrorOr<void *> malloc_host(size_t size) const = 0;
  virtual ErrorOrSuccess free_host(void *ptr) const = 0;

  // DeviceBuffer allocation.
  virtual ErrorOr<DeviceBufferRef> createBuffer(size_t len,
                                                size_t elem_size) const = 0;

  // DeviceBuffer copies.
  virtual ErrorOrSuccess enqueueCopyToDevice(const DeviceBuffer &dst,
                                             const void *src) const = 0;
  virtual ErrorOrSuccess
  enqueueCopyFromDevice(void *dst, const DeviceBuffer &src) const = 0;
  virtual ErrorOrSuccess
  enqueueCopyDeviceToDevice(const DeviceBuffer &dst,
                            const DeviceBuffer &src) const = 0;

  // DeviceFunction compilation and execution.
  virtual ErrorOr<DeviceFunctionRef>
  compileFunction(const char *function_name, const void *data) const = 0;
  virtual ErrorOrSuccess
  enqueueFunctionExec(const DeviceFunction &func, uint32_t gridX,
                      uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                      uint32_t blockY, uint32_t blockZ,
                      const DeviceFunctionArguments &args) const = 0;

  // Synchronize asynchronous operations.
  virtual ErrorOrSuccess synchronize() const = 0;

  static ErrorOr<DeviceContextRef> create(const char *kind, int id = 0);

protected:
  DeviceContext(const Device &device);

private:
  DeviceRef device_;

  // Explicitly mark copy and assignment as unsupported.
  DeviceContext(const DeviceContext &other) = delete;
  DeviceContext(DeviceContext &&other) = delete;
  DeviceContext &operator=(const DeviceContext &) = delete;
  DeviceContext &operator=(DeviceContext &&other) = delete;
};

//
// Internal implementation details follow.
//
// Private template definitions used by external code.
//
template <typename... Ts>
DeviceFunctionArgumentsRef
DeviceFunction::buildArguments(const Ts &...ts) const {
  DeviceFunctionArguments *args = allocateArguments();
  args->build(std::forward<const Ts>(ts)...);
  return DeviceFunctionArgumentsRef::take(args);
}

template <typename... Ts>
void DeviceFunctionArguments::build(const Ts &...ts) {
  pushArgs<0>(std::forward<const Ts>(ts)...);
}

template <int index, typename T0, typename... T1s>
void DeviceFunctionArguments::pushArgs(const T0 &&t0, const T1s &&...t1s) {
  pushArg(index, t0);
  pushArgs<index + 1, const T1s...>(std::forward<const T1s>(t1s)...);
}

template <int index, typename T0>
void DeviceFunctionArguments::pushArgs(const T0 &&t0) {
  pushArg(index, t0);
}

} // namespace M::AsyncRT

#endif // ASYNCRT_DEVICECONTEXT_DEVICECONTEXT_H
