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
#include "llvm/ADT/Twine.h"

#include <string.h>

namespace M::AsyncRT {
//
// ===== DeviceBuffer ===== //
//
DeviceBuffer::DeviceBuffer(const DeviceContext &context, size_t len,
                           size_t elem_size)
    : context_(DeviceContextRef::copy(&context)), len_(len),
      elem_size_(elem_size) {}

DeviceBuffer::DeviceBuffer(const DeviceContext &context,
                           const DeviceBuffer &parent, size_t len,
                           size_t elem_size)
    : context_(DeviceContextRef::copy(&context)),
      parent_(DeviceBufferRef::copy(&parent)), len_(len),
      elem_size_(elem_size) {}

const DeviceContext &DeviceBuffer::context() const { return *context_; }

const DeviceBuffer *DeviceBuffer::parent() const {
  return parent_.getPointer();
}

size_t DeviceBuffer::len() const { return len_; }

size_t DeviceBuffer::elem_size() const { return elem_size_; }

//
// ===== DeviceFunction ===== //
//
enum DeviceFunction::ArgumentType : uint8_t {
  UNSET = 0, // This entry must be zero and first.
  UINT8,
  UINT16,
  UINT32,
  UINT64,
  INT8,
  INT16,
  INT32,
  INT64,
  FP4,
  FP8,
  FP16,
  FP32,
  FP64,
  BUFFER,
  _NUM_ENTRIES_ // This entry must be last.
};

const char *DeviceFunction::typeName(ArgumentType arg_type) {
  const char *type_names[] = {"unset",   "uint8",   "uint16",      "uint32",
                              "uint64",  "int8",    "int16",       "int32",
                              "int64",   "float4",  "float8",      "float16",
                              "float32", "float64", "DeviceBuffer"};
  static_assert(sizeof(type_names) ==
                (static_cast<int>(_NUM_ENTRIES_) * sizeof(*type_names)));
  int index = static_cast<int>(arg_type);
  return ((index >= 0) && (index < _NUM_ENTRIES_)) ? type_names[index]
                                                   : "unknown";
}

DeviceFunction::DeviceFunction(
    const DeviceContext &context,
    const DeviceFunction::SignatureBuilder &signature)
    : context_(DeviceContextRef::copy(&context)), signature_(signature.types_) {
}

const DeviceContext &DeviceFunction::context() const { return *context_; }

ErrorOrSuccess DeviceFunction::checkSignatureAt(int index,
                                                ArgumentType arg_type) const {
  int expected_size = signature_.size();
  if ((index < 0) || index >= expected_size) {
    return Error("Argument count mismatch at " + Twine(index + 1) +
                 "th position: Expected " + Twine(expected_size) +
                 " arguments.");
  }
  ArgumentType expected_type = signature_.at(index);
  if (expected_type != arg_type) {
    return Error(llvm::Twine("Argument mismatch at ") + Twine(index + 1) +
                 "th position: Expected " + typeName(expected_type) + ", got " +
                 typeName(arg_type) + ".");
  }
  return success();
}

//
// ===== DeviceFunction::SignatureBuilder ===== //
//
DeviceFunction::SignatureBuilder &
DeviceFunction::SignatureBuilder::addUInt64() {
  types_.push_back(UINT64);
  return *this;
}

DeviceFunction::SignatureBuilder &
DeviceFunction::SignatureBuilder::addBuffer() {
  types_.push_back(BUFFER);
  return *this;
}

//
// ===== DeviceFunction::Arguments ===== //
//
DeviceFunction::Arguments::Arguments(const DeviceFunction &fn)
    : fn_(DeviceFunctionRef::copy(&fn)) {}

bool DeviceFunction::Arguments::hasError() const { return err_.isError(); }

const ErrorOrSuccess &DeviceFunction::Arguments::getError() const {
  return err_;
}

void DeviceFunction::Arguments::checkArg(int index, const uint64_t &val) {
  if (err_)
    return;
  err_ = fn_->checkSignatureAt(index, UINT64);
  pushArg(index, val);
}

void DeviceFunction::Arguments::checkArg(int index, const DeviceBuffer &val) {
  if (err_)
    return;
  err_ = fn_->checkSignatureAt(index, BUFFER);
  pushArg(index, val);
}

//
// ===== DeviceTimer ===== //
//
DeviceTimer::DeviceTimer(const DeviceContext &context)
    : context_(DeviceContextRef::copy(&context)) {}

const DeviceContext &DeviceTimer::context() const { return *context_; }

//
// ===== DeviceContext ===== //
//
DeviceContext::DeviceContext(const Device &device)
    : device_(DeviceRef::copy(&device)) {}

const Device &DeviceContext::device() const { return *device_; }

ErrorOrSuccess DeviceContext::isCompatibleWithMAX() const { return success(); }

ErrorOr<int> DeviceContext::computeCapability() const {
  // RUNP-494 Remove this, compute capability is a CUDA-specific concept.
  return 0;
}

ErrorOr<DeviceContextRef> DeviceContext::create(const char *kind, int id) {
  // TODO(iposva): Implement dynamic registration mechanism.
  if (strcmp(kind, "cuda") == 0) {
    return CUDADeviceContext::create(id);
  } else if (strcmp(kind, "cpu") == 0) {
    return CpuDeviceContext::create(id);
  }

  return Error(llvm::Twine("Unsupported device kind: ") + kind);
}

ErrorOr<int> Device::numberOfDevices(const char *kind) {
  // TODO(iposva): Implement dynamic registration mechanism.
  if (strcmp(kind, "cuda") == 0) {
    return CUDADevice::numberOfDevices();
  } else if (strcmp(kind, "cpu") == 0) {
    return CpuDevice::numberOfDevices();
  }

  return Error(llvm::Twine("Unsupported device kind: ") + kind);
}

} // namespace M::AsyncRT
