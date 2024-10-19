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

#include "Support/AlignedAlloc.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"

#include <chrono>
#include <cstdint>
#include <stddef.h>
#include <string>
#include <vector>

namespace M::AsyncRT {

class Device;
class DeviceBuffer;
class DeviceContext;
class DeviceFunction;
class DeviceTimer;

using DeviceRef = RCRef<const Device>;
using DeviceBufferRef = RCRef<const DeviceBuffer>;
using DeviceContextRef = RCRef<const DeviceContext>;
using DeviceFunctionRef = RCRef<const DeviceFunction>;
using DeviceTimerRef = RCRef<const DeviceTimer>;

// RUNP-397: This should come as a form of a function dependent on the device
static constexpr size_t kPreferredCUDAMemoryAlignment = 256;
static constexpr size_t kPreferredCPUMemoryAlignment =
    M::kPreferredMemoryAlignment;

[[maybe_unused]] static size_t getPreferredAlignment(std::string label) {
  if (label == "cpu")
    return kPreferredCPUMemoryAlignment;
  else
    return kPreferredCUDAMemoryAlignment;
}

class Device : public ReferenceCounted<Device> {
public:
  virtual ~Device() = default;

  virtual int id() const = 0;

  virtual std::string name() const = 0;

  static ErrorOr<int> numberOfDevices(const char *kind);

protected:
  Device() = default;
};

class DeviceBuffer : public ReferenceCounted<DeviceBuffer> {
public:
  virtual ~DeviceBuffer() = default;

  const DeviceContext &context() const;
  const DeviceBuffer *parent() const;

  size_t len() const;
  size_t elem_size() const;

  // Returns the raw device pointer, which is not safe to deref on the host
  virtual void *ptr() const = 0;
  // TODO: Remove the V1 compatibility.
  virtual void releasePtr() const = 0;

  virtual ErrorOr<DeviceBufferRef> createSubBuffer(size_t offset, size_t len,
                                                   size_t elem_size) const = 0;

protected:
  DeviceBuffer(const DeviceContext &context, size_t len, size_t elem_size);
  DeviceBuffer(const DeviceContext &context, const DeviceBuffer &parent,
               size_t len, size_t elem_size);

private:
  DeviceContextRef context_;
  DeviceBufferRef parent_;
  size_t len_;
  size_t elem_size_;
};

class DeviceFunction : public ReferenceCounted<DeviceFunction> {
public:
  enum ArgumentType : uint8_t;
  class SignatureBuilder;
  class Arguments;

  using ArgumentsRef = RCRef<const Arguments>;

  virtual ~DeviceFunction() = default;

  const DeviceContext &context() const;

  virtual const std::string &name() const = 0;

  virtual ErrorOrSuccess copyToNamedConstant(const char *name, const void *data,
                                             size_t byte_size) const = 0;

  virtual ErrorOr<int32_t> TEST_ONLY_getAttribute(int32_t code) const = 0;

  template <typename... Ts>
  ArgumentsRef buildArguments(const Ts &...ts) const;

protected:
  DeviceFunction(const DeviceContext &context,
                 const DeviceFunction::SignatureBuilder &signature);

  virtual Arguments *allocateArguments(const DeviceFunction &fn) const = 0;

private:
  DeviceContextRef context_;
  std::vector<ArgumentType> signature_;

  ErrorOrSuccess checkSignatureAt(int index, ArgumentType arg_type) const;
  static const char *typeName(ArgumentType arg_type);
};

class DeviceFunction::SignatureBuilder {
public:
  SignatureBuilder() = default;
  ~SignatureBuilder() = default;

  SignatureBuilder &addUInt64();
  SignatureBuilder &addBuffer();

private:
  friend class DeviceFunction;

  std::vector<ArgumentType> types_;
};

class DeviceFunction::Arguments
    : public ReferenceCounted<DeviceFunction::Arguments> {
public:
  virtual ~Arguments() = default;

  bool hasError() const;
  const ErrorOrSuccess &getError() const;

protected:
  Arguments(const DeviceFunction &func);

  // The implementations of these methods are being called based on the
  // arguments being passed to the DeviceFunction::buildArguments() method.
  // TODO: Add argument types incrementally as needed.
  virtual void pushArg(int index, const uint64_t &val) = 0;
  virtual void pushArg(int index, const DeviceBuffer &val) = 0;

private:
  friend class DeviceFunction;

  DeviceFunctionRef fn_;
  ErrorOrSuccess err_;

  template <typename... Ts>
  void build(const Ts &...ts);

  template <int index, typename T0, typename... T1s>
  void pushArgs(const T0 &&t0, const T1s &&...t1s);

  template <int index, typename T0>
  void pushArgs(const T0 &&t0);

  // TODO: Add argument types incrementally as needed.
  void checkArg(int index, const uint64_t &val);
  void checkArg(int index, const DeviceBuffer &val);
};

class DeviceTimer : public ReferenceCounted<DeviceTimer> {
public:
  virtual ~DeviceTimer() = default;

  const DeviceContext &context() const;

protected:
  DeviceTimer(const DeviceContext &context);

private:
  DeviceContextRef context_;
};

struct DeviceMemInfo {
  DeviceMemInfo() : total(0), free(0) {}
  size_t total;
  size_t free;
};

class DeviceContext : public ReferenceCounted<DeviceContext> {
public:
  enum class SyncMode {
    SYNC = 1,
    ASYNC = 2,
  };

  static constexpr SyncMode SYNC = SyncMode::SYNC;
  static constexpr SyncMode ASYNC = SyncMode::ASYNC;

  virtual ~DeviceContext() = default;

  const Device &device() const;

  // Host memory allocation.
  virtual ErrorOr<void *> malloc_host(size_t size) const = 0;
  virtual ErrorOrSuccess free_host(void *ptr) const = 0;

  // DeviceBuffer allocation.
  ErrorOr<DeviceBufferRef> enqueueCreateBuffer(size_t len,
                                               size_t elem_size) const;
  ErrorOr<DeviceBufferRef> createBuffer(size_t len, size_t elem_size) const;
  virtual ErrorOr<DeviceBufferRef> createBuffer(size_t len, size_t elem_size,
                                                SyncMode mode) const = 0;
  // TODO: Remove the V1 compatibility.
  virtual DeviceBufferRef createBufferOwning(void *ptr, size_t len,
                                             size_t elem_size,
                                             bool owning) const = 0;

  // DeviceBuffer copies.
  ErrorOrSuccess enqueueCopyToDevice(const DeviceBuffer &dst,
                                     const void *src) const;
  ErrorOrSuccess copyToDevice(const DeviceBuffer &dst, const void *src) const;
  virtual ErrorOrSuccess copyToDevice(const DeviceBuffer &dst, const void *src,
                                      SyncMode mode) const = 0;

  ErrorOrSuccess enqueueCopyFromDevice(void *dst,
                                       const DeviceBuffer &src) const;
  ErrorOrSuccess copyFromDevice(void *dst, const DeviceBuffer &src) const;
  virtual ErrorOrSuccess copyFromDevice(void *dst, const DeviceBuffer &src,
                                        SyncMode mode) const = 0;

  ErrorOrSuccess enqueueCopyDeviceToDevice(const DeviceBuffer &dst,
                                           const DeviceBuffer &src) const;
  ErrorOrSuccess copyDeviceToDevice(const DeviceBuffer &dst,
                                    const DeviceBuffer &src) const;
  virtual ErrorOrSuccess copyDeviceToDevice(const DeviceBuffer &dst,
                                            const DeviceBuffer &src,
                                            SyncMode mode) const = 0;

  virtual ErrorOrSuccess setMemory(const DeviceBuffer &dst, uint32_t val,
                                   size_t elem_size, SyncMode mode) const = 0;

  // DeviceFunction compilation and execution.
  virtual ErrorOr<DeviceFunctionRef>
  compileFunction(const char *function_name,
                  const DeviceFunction::SignatureBuilder &signature,
                  const void *data, int32_t max_registers = -1,
                  int32_t threads_per_block = -1, int32_t cache_mode = -1,
                  int32_t cache_config = -1,
                  int32_t max_dynamic_shared_bytes = -1) const = 0;
  virtual ErrorOrSuccess
  enqueueFunctionExec(const DeviceFunction &func, uint32_t gridX,
                      uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                      uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
                      const DeviceFunction::Arguments &args) const = 0;
  virtual ErrorOrSuccess
  enqueueFunctionExecDirect(const DeviceFunction &func, uint32_t gridX,
                            uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                            uint32_t blockY, uint32_t blockZ,
                            uint32_t sharedMemBytes, void *attrs,
                            uint32_t num_attrs, void **args) const = 0;

  // Synchronize asynchronous operations.
  virtual ErrorOrSuccess synchronize() const = 0;

  virtual ErrorOr<DeviceTimerRef> startTimer() const = 0;
  virtual ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) const = 0;

  // Queries
  virtual ErrorOr<DeviceMemInfo> getMemoryInfo() const = 0;
  virtual ErrorOrSuccess isCompatibleWithMAX() const;
  virtual ErrorOr<int> computeCapability() const;
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
// Inline helpers.
inline ErrorOr<DeviceBufferRef>
DeviceContext::enqueueCreateBuffer(size_t len, size_t elem_size) const {
  return createBuffer(len, elem_size, ASYNC);
}
inline ErrorOr<DeviceBufferRef>
DeviceContext::createBuffer(size_t len, size_t elem_size) const {
  return createBuffer(len, elem_size, SYNC);
}
inline ErrorOrSuccess
DeviceContext::enqueueCopyToDevice(const DeviceBuffer &dst,
                                   const void *src) const {
  return copyToDevice(dst, src, ASYNC);
}
inline ErrorOrSuccess DeviceContext::copyToDevice(const DeviceBuffer &dst,
                                                  const void *src) const {
  return copyToDevice(dst, src, SYNC);
}
inline ErrorOrSuccess
DeviceContext::enqueueCopyFromDevice(void *dst, const DeviceBuffer &src) const {
  return copyFromDevice(dst, src, ASYNC);
}
inline ErrorOrSuccess
DeviceContext::copyFromDevice(void *dst, const DeviceBuffer &src) const {
  return copyFromDevice(dst, src, SYNC);
}
inline ErrorOrSuccess
DeviceContext::enqueueCopyDeviceToDevice(const DeviceBuffer &dst,
                                         const DeviceBuffer &src) const {
  return copyDeviceToDevice(dst, src, ASYNC);
}
inline ErrorOrSuccess
DeviceContext::copyDeviceToDevice(const DeviceBuffer &dst,
                                  const DeviceBuffer &src) const {
  return copyDeviceToDevice(dst, src, SYNC);
}

//
// Private template definitions used by external code.
//
template <typename... Ts>
DeviceFunction::ArgumentsRef
DeviceFunction::buildArguments(const Ts &...ts) const {
  Arguments *args = allocateArguments(*this);
  args->build(std::forward<const Ts>(ts)...);
  return ArgumentsRef::take(args);
}

template <typename... Ts>
void DeviceFunction::Arguments::build(const Ts &...ts) {
  pushArgs<0>(std::forward<const Ts>(ts)...);
}

template <int index, typename T0, typename... T1s>
void DeviceFunction::Arguments::pushArgs(const T0 &&t0, const T1s &&...t1s) {
  checkArg(index, t0);
  pushArgs<index + 1, const T1s...>(std::forward<const T1s>(t1s)...);
}

template <int index, typename T0>
void DeviceFunction::Arguments::pushArgs(const T0 &&t0) {
  checkArg(index, t0);
}

} // namespace M::AsyncRT

#endif // ASYNCRT_DEVICECONTEXT_DEVICECONTEXT_H
