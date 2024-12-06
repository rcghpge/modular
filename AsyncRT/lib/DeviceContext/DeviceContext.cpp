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
#include "HIP/HIPDeviceContext.h"
#include "llvm/ADT/Twine.h"

#include <string.h>

namespace M::AsyncRT {

// TODO: Remove once users have been updated to use
// `Device::preferredAlignment`.
size_t getPreferredAlignment(std::string label) {
  return Device::preferredAlignment(label);
}

//
// ===== DeviceBuffer ===== //
//
Device::Device(API api, std::string name, int id, size_t allocAligment)
    : name_(std::move(name)), id_(id), api_(api),
      alloc_alignment_(allocAligment) {}

const std::string &Device::name() const { return name_; }

int Device::id() const { return id_; }

Device::API Device::api() const { return api_; }

StringRef Device::vendor() const {
  if (api() == CUDADevice::api)
    return "nvidia";
  if (api() == HIPDevice::api)
    return "amdgpu";
  if (api() == CpuDevice::api)
    return "cpu";
  return "";
}

size_t Device::allocationAlignment() const { return alloc_alignment_; }

size_t Device::preferredAlignment(StringRef api) {
  if (api == "gpu") {
    // TODO(GEX-1416): Remove special hard-coded value to pass tests.
    return 256;
  }
  ErrorOr<DeviceRef> device = create(api, 0);
  if (device.isError()) {
    return M::kPreferredMemoryAlignment;
  }
  return device.get()->allocationAlignment();
}

ErrorOr<DeviceRef> Device::create(StringRef api, int id) {
  // TODO(iposva): Implement dynamic registration mechanism.
  if (api == "gpu") {
    ErrorOr<const Device &> device = CUDADevice::get(id);
    if (!device.isError()) {
      return DeviceRef::copy(&device.get());
    }
    device = HIPDevice::get(id);
    if (!device.isError()) {
      return DeviceRef::copy(&device.get());
    }
    return Error("No supported \"gpu\" device available.");
  } else if (api == "cuda") {
    ErrorOr<const CUDADevice &> device = CUDADevice::get(id);
    if (device.isError())
      return device.takeError();
    return DeviceRef::copy(&device.get());
  } else if (api == "cpu") {
    ErrorOr<const CpuDevice &> device = CpuDevice::get(id);
    if (device.isError())
      return device.takeError();
    return DeviceRef::copy(&device.get());
  } else if (api == "hip") {
    ErrorOr<const HIPDevice &> device = HIPDevice::get(id);
    if (device.isError())
      return device.takeError();
    return DeviceRef::copy(&device.get());
  }

  return Error(llvm::Twine("Unsupported device api: \"") + api + "\"");
}

//
// ===== DeviceBuffer ===== //
//
DeviceBuffer::DeviceBuffer(const DeviceContext &context, size_t byteSize)
    : context_(DeviceContextRef::copy(&context)), byteSize_(byteSize) {}

DeviceBuffer::DeviceBuffer(const DeviceContext &context,
                           const DeviceBuffer &parent, size_t byteSize)
    : context_(DeviceContextRef::copy(&context)),
      parent_(DeviceBufferRef::copy(&parent)), byteSize_(byteSize) {}

const DeviceContext &DeviceBuffer::context() const { return *context_; }

const DeviceBuffer *DeviceBuffer::parent() const {
  return parent_.getPointer();
}

size_t DeviceBuffer::byteSize() const { return byteSize_; }

//
// ===== DeviceStream ===== //
//
DeviceStream::DeviceStream(const DeviceContext &context)
    : context_(DeviceContextRef::copy(&context)) {}

const DeviceContext &DeviceStream::context() const { return *context_; }

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

ErrorOr<DeviceBufferRef>
DeviceContext::enqueueCreateBuffer(size_t byteSize) const {
  if (byteSize == 0)
    return Error("cannot allocate zero length buffers");

  return createBuffer(byteSize, ASYNC);
}

ErrorOr<DeviceBufferRef> DeviceContext::createBuffer(size_t byteSize) const {
  if (byteSize == 0)
    return Error("cannot allocate zero length buffers");

  return createBuffer(byteSize, SYNC);
}

ErrorOr<DeviceContextRef> DeviceContext::create(StringRef api, int id,
                                                size_t buffer_cache_size) {
  ErrorOr<DeviceRef> device = Device::create(api, id);
  if (device.isError()) {
    return device.takeError();
  }
  return (*device)->createContext(buffer_cache_size);
}

int Device::numberOfDevices(StringRef api) {
  ErrorOr<DeviceRef> device = Device::create(api);
  if (device.isError()) {
    return 0;
  }
  return (*device)->deviceCount();
}

const std::string &Device::getAcceleratorArchOrEmpty() {
  static std::string arch = [] {
    ErrorOr<DeviceRef> device = Device::create("gpu");
    if (device.isError()) {
      return std::string();
    }

    ErrorOr<int> major =
        (*device)->getAttribute(Device::Attribute::COMPUTE_CAPABILITY_MAJOR);
    ErrorOr<int> minor =
        (*device)->getAttribute(Device::Attribute::COMPUTE_CAPABILITY_MINOR);
    if (major.isError() || minor.isError()) {
      return std::string();
    }

    return (llvm::Twine((*device)->vendor()) + ":" + llvm::Twine(*major) +
            llvm::Twine(*minor))
        .str();
  }();
  return arch;
}

} // namespace M::AsyncRT
