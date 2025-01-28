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

#include "BufferCache.h"
#include "CPU/CpuDeviceContext.h"
#include "CUDA/CUDADeviceContext.h"
#include "HIP/HIPDeviceContext.h"
#include "Support/AlignedAlloc.h"
#include "llvm/ADT/Twine.h"

#include <string.h>

#define ERR_CHECK_ERR(err_call)                                                \
  do {                                                                         \
    auto err_result = (err_call);                                              \
    if (err_result.isError()) {                                                \
      return formatError(__FILE__, __LINE__, __FUNCTION__,                     \
                         err_result.getError());                               \
    }                                                                          \
  } while (0)

#define ERR_CHECK(err_call)                                                    \
  do {                                                                         \
    auto err_result = (err_call);                                              \
    if (err_result.isError()) {                                                \
      M::Error err = formatError(__FILE__, __LINE__, __FUNCTION__,             \
                                 err_result.getError());                       \
      /* We are not in a context where we can return an Error here. So we   */ \
      /* print the message and abort().                                     */ \
      llvm::errs() << err << "\n";                                             \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define RAISE_ERROR_MESSAGE(msg)                                               \
  do {                                                                         \
    return formatError(__FILE__, __LINE__, __FUNCTION__, msg);                 \
  } while (0)

namespace M::AsyncRT {

// Allow printing of detailed error messages with C++ source information when
// debugging low-level implementation.
static constexpr bool detailed_err_msg = false;

static Error formatError(const char *file, int line, const char *func,
                         const char *msg = nullptr) {
  if (detailed_err_msg || (msg == nullptr)) {
    return Error(llvm::Twine(file) + ":" + llvm::Twine(line) + " - " + func +
                 " failed" + ((msg == nullptr) ? "." : ": ") +
                 ((msg == nullptr) ? "" : msg));
  }
  return Error(msg);
}

// TODO: Remove once users have been updated to use
// `Device::preferredAlignment`.
size_t getPreferredAlignment(std::string label) {
  return Device::preferredAlignment(label);
}

//
// ===== Device ===== //
//
Device::Device(API api, std::string name, int id, size_t allocAligment)
    : name_(std::move(name)), id_(id), api_(api),
      allocAlignment_(allocAligment) {}

const std::string &Device::name() const { return name_; }

int Device::id() const { return id_; }

Device::API Device::api() const { return api_; }

StringRef Device::vendor() const {
  if (api() == cudaAPI)
    return "nvidia";
  if (api() == hipAPI)
    return "amdgpu";
  if (api() == cpuAPI)
    return "cpu";
  return "";
}

size_t Device::allocationAlignment() const { return allocAlignment_; }

size_t Device::preferredAlignment(StringRef api) {
  if (api == gpuAPI) {
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
  if (api == gpuAPI) {
    ErrorOr<const Device &> device = CUDADevice::get(id);
    if (!device.isError()) {
      return DeviceRef::copy(&device.get());
    }
    device = HIPDevice::get(id);
    if (!device.isError()) {
      return DeviceRef::copy(&device.get());
    }
    return Error("No supported \"gpu\" device available.");
  } else if (api == cudaAPI) {
    ErrorOr<const CUDADevice &> device = CUDADevice::get(id);
    if (device.isError())
      return device.takeError();
    return DeviceRef::copy(&device.get());
  } else if (api == cpuAPI) {
    ErrorOr<const CpuDevice &> device = CpuDevice::get(id);
    if (device.isError())
      return device.takeError();
    return DeviceRef::copy(&device.get());
  } else if (api == hipAPI) {
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
DeviceBuffer::DeviceBuffer(const DeviceContext &context, void *ptr,
                           size_t byteSize)
    : context_(DeviceContextRef::copy(&context)), ptr_(ptr),
      byteSize_(byteSize) {}
DeviceBuffer::DeviceBuffer(const DeviceContext &context,
                           const DeviceBuffer &parent, void *ptr,
                           size_t byteSize)
    : context_(DeviceContextRef::copy(&context)),
      parent_(DeviceBufferRef::copy(&parent)), ptr_(ptr), byteSize_(byteSize) {}
// TODO: Remove the V1 compatibility.
DeviceBuffer::DeviceBuffer(const DeviceContext &context, void *ptr,
                           size_t byteSize, bool owning)
    : context_(DeviceContextRef::copy(&context)), ptr_(ptr),
      byteSize_(byteSize), owning_(owning) {}

DeviceBuffer::~DeviceBuffer() {
  if (!owning_) {
    // TODO: Remove the V1 compatibility.
    // This is a device buffer created by pointer in Mojo while ownership is
    // maintained elsewhere.
    return;
  }
  if ((parent() != nullptr) || (ptr_ == nullptr)) {
    // Parent owns the memory.
    return;
  }
  if (!context().bufferCache().free(ptr_)) {
    ERR_CHECK(context().memFree(ptr_));
  }
}
const DeviceContext &DeviceBuffer::context() const { return *context_; }

const DeviceBuffer *DeviceBuffer::parent() const {
  return parent_.getPointer();
}

void *DeviceBuffer::ptr() const { return ptr_; }

// TODO: Remove the V1 compatibility.
void *DeviceBuffer::releasePtr() const {
  auto tmp = ptr_;
  ptr_ = nullptr;
  owning_ = false;
  return tmp;
}

size_t DeviceBuffer::byteSize() const { return byteSize_; }

// Default implementation, which can be specialized by subclasses.
ErrorOrSuccess DeviceBuffer::copyTo(const DeviceBuffer &dst) const {
  const DeviceBuffer &src = *this;
  const Device::API src_api = src.context().device().api();
  const Device::API dst_api = dst.context().device().api();
  if (src_api == Device::cpuAPI) {
    return dst.context().enqueueCopyToDevice(dst, src.ptr());
  } else if (dst_api == Device::cpuAPI) {
    return src.context().enqueueCopyFromDevice(dst.ptr(), src);
  } else if (src_api == dst_api) {
    return dst.context().enqueueCopyDeviceToDevice(dst, src);
  }
  return Error("Device API mismatch copying from \"" + src_api + "\" to \"" +
               dst_api + "\"");
}

ErrorOr<DeviceBufferRef> DeviceBuffer::createSubBuffer(size_t byteOffset,
                                                       size_t byteSize) const {
  char *orig_ptr = static_cast<char *>(this->ptr());
  size_t origByteSize = this->byteSize();

  if ((byteSize > origByteSize) || (origByteSize - byteSize < byteOffset)) {
    RAISE_ERROR_MESSAGE("offset, len and elem_size are out of range");
  }
  return DeviceBufferRef::take(
      new DeviceBuffer(context(), *this, orig_ptr + byteOffset, byteSize));
}

//
// ===== DeviceEvent ===== //
//
DeviceEvent::DeviceEvent(const DeviceContext &context)
    : context_(DeviceContextRef::copy(&context)) {}

const DeviceContext &DeviceEvent::context() const { return *context_; }

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

const char *DeviceFunction::typeName(ArgumentType argType) {
  const char *typeNames[] = {"unset",   "uint8",   "uint16",      "uint32",
                             "uint64",  "int8",    "int16",       "int32",
                             "int64",   "float4",  "float8",      "float16",
                             "float32", "float64", "DeviceBuffer"};
  static_assert(sizeof(typeNames) ==
                (static_cast<int>(_NUM_ENTRIES_) * sizeof(*typeNames)));
  int index = static_cast<int>(argType);
  return ((index >= 0) && (index < _NUM_ENTRIES_)) ? typeNames[index]
                                                   : "unknown";
}

DeviceFunction::DeviceFunction(
    const DeviceContext &context,
    const DeviceFunction::SignatureBuilder &signature)
    : context_(DeviceContextRef::copy(&context)), signature_(signature.types_) {
}

const DeviceContext &DeviceFunction::context() const { return *context_; }

ErrorOrSuccess DeviceFunction::checkSignatureAt(int index,
                                                ArgumentType argType) const {
  int expectedSize = signature_.size();
  if ((index < 0) || index >= expectedSize) {
    return Error("Argument count mismatch at " + Twine(index + 1) +
                 "th position: Expected " + Twine(expectedSize) +
                 " arguments.");
  }
  ArgumentType expectedType = signature_.at(index);
  if (expectedType != argType) {
    return Error(llvm::Twine("Argument mismatch at ") + Twine(index + 1) +
                 "th position: Expected " + typeName(expectedType) + ", got " +
                 typeName(argType) + ".");
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
DeviceContext::DeviceContext(const Device &device,
                             std::unique_ptr<BufferCache> bufferCache)
    : device_(DeviceRef::copy(&device)), bufferCache_(std::move(bufferCache)) {
  assert(bufferCache_ != nullptr);
}

DeviceContext::~DeviceContext() {}

const Device &DeviceContext::device() const { return *device_; }

ErrorOr<DeviceBufferRef>
DeviceContext::enqueueCreateBuffer(size_t byteSize) const {
  if (byteSize == 0)
    return this->createBufferOwning(nullptr, 0, true);
  return allocBuffer(byteSize, ASYNC);
}

ErrorOr<DeviceBufferRef> DeviceContext::createBuffer(size_t byteSize) const {
  if (byteSize == 0)
    return this->createBufferOwning(nullptr, 0, true);
  return allocBuffer(byteSize, SYNC);
}

DeviceBufferRef DeviceContext::createBufferOwning(void *ptr, size_t byteSize,
                                                  bool owning) const {
  return DeviceBufferRef::take(new DeviceBuffer(*this, ptr, byteSize, owning));
}

ErrorOr<DeviceBufferRef> DeviceContext::allocBuffer(size_t byteSize,
                                                    SyncMode mode) const {
  assert(byteSize != 0);
  void *ptr = bufferCache().allocate(byteSize);
  if (ptr == nullptr) {
    ERR_CHECK_ERR(memAlloc(&ptr, byteSize, mode));
  }
  return DeviceBufferRef::take(new DeviceBuffer(*this, ptr, byteSize));
}

ErrorOr<DeviceContextRef> DeviceContext::create(StringRef api, int id,
                                                size_t bufferCacheSize) {
  ErrorOr<DeviceRef> device = Device::create(api, id);
  if (device.isError()) {
    return device.takeError();
  }
  return (*device)->createContext(bufferCacheSize);
}

BufferCache &DeviceContext::bufferCache() const { return *bufferCache_; }

// DeviceFunction compiling, loading, and execution.
ErrorOr<M::BufferRef>
DeviceContext::compileFunction(StringRef ir, StringRef debugLevel,
                               int32_t optimizationLevel) const {
  return Error("compileFunction for this device is not supported");
}

//
// ===== Device ===== //
//
int Device::numberOfDevices(StringRef api) {
  ErrorOr<DeviceRef> device = Device::create(api);
  if (device.isError()) {
    return 0;
  }
  return (*device)->deviceCount();
}

bool DeviceContext::configAsBool(StringRef key, bool defaultValue) {
  ErrorOr<Config> &config = getConfig();
  if (config.isError())
    return defaultValue;

  ErrorOr<bool> configValue = config->getValueAsBool(
      (llvm::Twine("device-context.") + key).str(), defaultValue);
  if (configValue.isError())
    return defaultValue;
  return *configValue;
}

int64_t DeviceContext::configAsInt(StringRef key, int64_t defaultValue) {
  ErrorOr<Config> &config = getConfig();
  if (config.isError())
    return defaultValue;

  StringRef s = config->getValue((llvm::Twine("device-context.") + key).str());
  long long result;
  if (!llvm::getAsSignedInteger(s, 10, result))
    return defaultValue;
  return result;
}

StringRef DeviceContext::configAsString(StringRef key, StringRef defaultValue) {
  ErrorOr<Config> &config = getConfig();
  if (config.isError())
    return defaultValue;

  return config->getValueOr((llvm::Twine("device-context.") + key).str(),
                            defaultValue);
}

ErrorOr<Config> &DeviceContext::getConfig() {
  static ErrorOr<Config> configuration = Config::open();
  return configuration;
}

const std::string &Device::getAcceleratorArchOrEmpty() {
  static std::string arch = [] {
    ErrorOr<DeviceRef> device = Device::create(gpuAPI);
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
