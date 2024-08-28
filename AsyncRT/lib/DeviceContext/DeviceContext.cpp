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

#include "AsyncRT/lib/DeviceContext/CPU/CpuDeviceContext.h"
#include "AsyncRT/lib/DeviceContext/CUDA/CUDADeviceContext.h"

#include <string.h>

namespace M::AsyncRT {
//
// ===== Device ===== //
//
Device::~Device() {}

//
// ===== DeviceBuffer ===== //
//
DeviceBuffer::DeviceBuffer(DeviceContextRef context, size_t len,
                           size_t elem_size)
    : context_(std::move(context)), len_(len), elem_size_(elem_size) {}

DeviceBuffer::~DeviceBuffer() {}

const DeviceContextRef &DeviceBuffer::context() const { return context_; }

size_t DeviceBuffer::len() const { return len_; }

size_t DeviceBuffer::elem_size() const { return elem_size_; }

//
// ===== DeviceContext ===== //
//
DeviceContext::DeviceContext(DeviceRef device) : device_(std::move(device)) {}

DeviceContext::~DeviceContext() {}

const Device *DeviceContext::device() const { return device_.getPointer(); }

DeviceContextRef DeviceContext::create(const char *kind) {
  // TODO(iposva): Implement dynamic registration mechanism.
  if (strcmp(kind, "cuda") == 0) {
    return CUDADeviceContext::create();
  } else if (strcmp(kind, "cpu") == 0) {
    return CpuDeviceContext::create();
  }

  return DeviceContextRef();
}

} // namespace M::AsyncRT
