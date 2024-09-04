//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CPU device context.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/DeviceContext/DeviceContext.h"

#include "CPU/CpuDeviceContext.h"
#include "CUDA/CUDADeviceContext.h"

#include <string.h>

namespace M::AsyncRT {
//
// ===== Device ===== //
//
Device::~Device() {
  // Nothing to be done currently.
}

//
// ===== DeviceBuffer ===== //
//
DeviceBuffer::DeviceBuffer(const DeviceContext &context, size_t len,
                           size_t elem_size)
    : context_(DeviceContextRef::copy(&context)), len_(len),
      elem_size_(elem_size) {}

DeviceBuffer::~DeviceBuffer() {
  // Nothing to be done currently.
}

const DeviceContext &DeviceBuffer::context() const { return *context_; }

size_t DeviceBuffer::len() const { return len_; }

size_t DeviceBuffer::elem_size() const { return elem_size_; }

//
// ===== DeviceFunctionArguments ===== //
//
DeviceFunctionArguments::DeviceFunctionArguments() {}

DeviceFunctionArguments::~DeviceFunctionArguments() {
  // Nothing to be done currently.
}

//
// ===== DeviceFunction ===== //
//
DeviceFunction::DeviceFunction(DeviceContextRef context)
    : context_(std::move(context)) {}

DeviceFunction::~DeviceFunction() {
  // Nothing to be done currently.
}

const DeviceContextRef &DeviceFunction::context() const { return context_; }

//
// ===== DeviceContext ===== //
//
DeviceContext::DeviceContext(const Device &device)
    : device_(DeviceRef::copy(&device)) {}

DeviceContext::~DeviceContext() {
  // Nothing to be done currently.
}

const Device &DeviceContext::device() const { return *device_; }

ErrorOr<DeviceContextRef> DeviceContext::create(const char *kind) {
  // TODO(iposva): Implement dynamic registration mechanism.
  if (strcmp(kind, "cuda") == 0) {
    return CUDADeviceContext::create();
  } else if (strcmp(kind, "cpu") == 0) {
    return CpuDeviceContext::create();
  }

  return DeviceContextRef();
}

} // namespace M::AsyncRT
