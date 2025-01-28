//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CUDA device context.
//
//===----------------------------------------------------------------------===//

#include "CUDADeviceContext.h"

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "AsyncRT/lib/DeviceContext/BufferCache.h"
#include "CUDADriver.h"
#include "Support/Configuration.h"
#include "Support/ErrorOr.h"
#include "Support/PlatformUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <mutex>
#include <string>

namespace M::AsyncRT {
namespace {

// The helper class CUDAContextScope is used to set and restore the CUcontext
// around interactions with the CUDA driver.
class CUDAContextScope {
public:
  CUDAContextScope(const CUcontext ctx, const CUDADriver &driver)
      : driver_(driver), ctx_(ctx), prevCtx_(nullptr),
        status_(CUDADriver::SUCCESS) {
    driver_.CtxGetCurrent(&prevCtx_);
    if (prevCtx_ == ctx_)
      return;
    status_ = driver_.CtxSetCurrent(ctx_);
  }

  ~CUDAContextScope() {
    if ((prevCtx_ == ctx_) || (status_ != CUDADriver::SUCCESS))
      return;

    // Only pop the context if push was necessary and successful.
    CUcontext ctx = nullptr;
    driver_.CtxGetCurrent(&ctx);
    driver_.CtxSetCurrent(prevCtx_);
    assert(ctx == ctx_);
  }

  CUDADriver::Result status() { return status_; }

private:
  const CUDADriver &driver_;
  const CUcontext ctx_;
  CUcontext prevCtx_;
  CUDADriver::Result status_;
};

} // namespace
} // namespace M::AsyncRT

#define CUDA_CHECK_ERR(cuda_call)                                              \
  do {                                                                         \
    CUDADriver::Result cuda_result = cuda_call;                                \
    if (cuda_result != CUDADriver::SUCCESS) {                                  \
      return formatError(__FILE__, __LINE__, __FUNCTION__, cuda_result);       \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_ERR_MSG(cuda_call, msg)                                     \
  do {                                                                         \
    CUDADriver::Result cuda_result = cuda_call;                                \
    if (cuda_result != CUDADriver::SUCCESS) {                                  \
      return formatError(__FILE__, __LINE__, __FUNCTION__, cuda_result, msg);  \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(cuda_call)                                                  \
  do {                                                                         \
    CUDADriver::Result cuda_result = cuda_call;                                \
    if (cuda_result != CUDADriver::SUCCESS) {                                  \
      M::Error err =                                                           \
          formatError(__FILE__, __LINE__, __FUNCTION__, cuda_result);          \
      /* We are not in a context where we can return an Error here. So we   */ \
      /* print the message and abort().                                     */ \
      llvm::errs() << err << "\n";                                             \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define RAISE_ERROR(msg)                                                       \
  do {                                                                         \
    return formatError(__FILE__, __LINE__, __FUNCTION__, msg);                 \
  } while (0)

#define CUDA_WITH_CTX_ERR(ctx)                                                 \
  CUDAContextScope __cudaWithContext((ctx).cudaContext(), (ctx).cudaDriver()); \
  CUDA_CHECK_ERR(__cudaWithContext.status());

#define CUDA_WITH_CTX(ctx)                                                     \
  CUDAContextScope __cudaWithContext((ctx).cudaContext(), (ctx).cudaDriver()); \
  CUDA_CHECK(__cudaWithContext.status());

#define CHECK_IS_CUDA_ERR(dev)                                                 \
  do {                                                                         \
    if ((dev).api() != Device::cudaAPI) {                                      \
      RAISE_ERROR("Not a CUDA device.");                                       \
    }                                                                          \
  } while (0)

namespace M::AsyncRT {

// Allow printing of detailed error messages with C++ source information when
// debugging low-level implementation.
static constexpr bool detailed_err_msg = false;

// TODO(iposva) : Switch error handling to using `std::source_location` once we
// are using C++ 20.
static Error formatError(const char *file, int line, const char *func,
                         const llvm::Twine &msg) {
  if (detailed_err_msg) {
    return Error(llvm::Twine(file) + ":" + llvm::Twine(line) + " - " + func +
                 " - " + msg);
  }
  return Error(msg);
}

static Error formatError(const char *file, int line, const char *func,
                         CUDADriver::Result result,
                         std::string extra_msg = "") {
  ErrorOr<const CUDADriver &> err_driver = CUDADriver::get();
  const char *err_name = nullptr;
  const char *err_str = nullptr;

  if (!err_driver) {
    // Ignore errors and use nullptr of char* instead.
    err_driver->GetErrorName(result, &err_name);
    err_driver->GetErrorString(result, &err_str);
  }

  if (!extra_msg.empty()) {
    extra_msg = std::string(" - ") + extra_msg;
  }

  if ((err_name != nullptr) && (err_str != nullptr)) {
    return formatError(file, line, func,
                       llvm::Twine("CUDA call failed: ") + err_name + " (" +
                           err_str + ")" + extra_msg);
  } else if ((err_name != nullptr) || (err_str != nullptr)) {
    return formatError(file, line, func,
                       llvm::Twine("CUDA call failed: ") +
                           (err_name != nullptr ? err_name : err_str) +
                           extra_msg);
  } else {
    return formatError(file, line, func,
                       llvm::Twine("CUDA call failed: ") + llvm::Twine(result) +
                           extra_msg);
  }
}

// ===== CUDADevice ===== //
CUDADevice::CUDADevice(const CUDADriver &driver, int id, CUdevice device,
                       std::string name, size_t allocAlignment)
    // GEX-1416: Do not expect hardcoded number.
    : Device(api, std::move(name), id, 256), driver_(driver), device_(device),
      context_(nullptr) {}

CUDADevice::~CUDADevice() {
  if (context_ != nullptr)
    // We ignore the result here as there is not much we can do if it fails
    // since in most cases the CUDA driver is in the process of shutting down.
    cudaDriver().DevicePrimaryCtxRelease(cudaDevice());
}

const CUDADriver &CUDADevice::cudaDriver() const { return driver_; }

CUdevice CUDADevice::cudaDevice() const { return device_; }

ErrorOr<CUcontext> CUDADevice::cudaContext() const {
  std::lock_guard<std::mutex> lock(mu_);
  if (context_ == nullptr) {
    CUDA_CHECK_ERR(
        cudaDriver().DevicePrimaryCtxRetain(&context_, cudaDevice()));
  }
  return context_;
}

ErrorOr<int> CUDADevice::getDriverVersion() const {
  int driver_version = 0;
  CUDA_CHECK_ERR(driver_.DriverGetVersion(&driver_version));
  return driver_version;
}

// The value returned by `cuDriverGetVersion_` from the CUDA API is an integer
// with the driver version embedded. The major version number is in the
// thousands place and the minor version number is in the hundreds place, scaled
// up by ten.
static std::string driverVersionToString(int driver_version) {
  int major_version = driver_version / 1000;
  int minor_version = (driver_version % 1000) / 10;

  return (llvm::Twine(major_version) + "." + llvm::Twine(minor_version)).str();
}

ErrorOrSuccess CUDADevice::isCompatible() const {
#ifndef MODULAR_LINUX
  return Error("MAX with GPUs is only supported on Linux based systems.");
#endif

  ErrorOr<int> compute = computeCapability();
  if (compute.isError())
    return compute.takeError();
  if (*compute < MIN_COMPUTE_CAPABILITY) {
    RAISE_ERROR("MAX only supports CUDA Ampere architectures or higher");
  }

  int driver_version = 0;
  CUDA_CHECK_ERR(driver_.DriverGetVersion(&driver_version));
  if (driver_version < MIN_CUDA_DRIVER_VERSION) {
    RAISE_ERROR(
        "MAX is not supported using current GPU driver. MAX "
        "requires a minimum driver version of " +
        llvm::Twine(MIN_NVIDIA_DRIVER_VERSION) + " and CUDA version " +
        driverVersionToString(MIN_CUDA_DRIVER_VERSION) +
        ". See https://modul.ar/nvidia-driver-version for "
        "details about how to install the proper NVIDIA driver. The current "
        "CUDA version MAX detected is: " +
        driverVersionToString(driver_version));
  }

  return success();
}

ErrorOr<int> CUDADevice::computeCapability() const {
  int compute_major = 0, compute_minor = 0;
  CUDA_CHECK_ERR(driver_.DeviceGetAttribute(
      &compute_major, CUdevice_attribute::COMPUTE_CAPABILITY_MAJOR,
      cudaDevice()));
  CUDA_CHECK_ERR(driver_.DeviceGetAttribute(
      &compute_minor, CUdevice_attribute::COMPUTE_CAPABILITY_MINOR,
      cudaDevice()));
  return 10 * compute_major + compute_minor;
}

ErrorOr<bool> CUDADevice::canAccess(const Device &peer) const {
  int canAccessPeer = 0;
  CUDA_CHECK_ERR(driver_.DeviceCanAccessPeer(
      &canAccessPeer, cudaDevice(), CUDADevice::cast(peer).cudaDevice()));
  return canAccessPeer;
}

ErrorOr<int> CUDADevice::getAttribute(Attribute attr) const {
  int value = 0;
  CUDA_CHECK_ERR(driver_.DeviceGetAttribute(
      &value, static_cast<CUdevice_attribute>(attr), cudaDevice()));
  return value;
}

ErrorOr<DeviceContextRef>
CUDADevice::createContext(size_t bufferCacheSize) const {
  ErrorOr<CUcontext> context = cudaContext();
  if (context.isError())
    return context.takeError();

  // For the remainder of this method we use the context scope to ensure that
  // the context is properly removed when an error happens or then at the
  // return.
  CUDAContextScope withCtx(*context, this->cudaDriver());

  CUstream stream;
  CUDA_CHECK_ERR(this->cudaDriver().StreamCreate(&stream, 0));

  CUdeviceptr cache_base = nullptr;
  if (bufferCacheSize != 0) {
    CUDA_CHECK_ERR(this->cudaDriver().MemAlloc(&cache_base, bufferCacheSize));
  }

  return CUDADeviceContextRef::create(*this, *context, stream, cache_base,
                                      bufferCacheSize);
}

int CUDADevice::deviceCount() const {
  ErrorOr<const DeviceList &> err_or_devices = deviceList();
  if (err_or_devices.isError()) {
    return 0;
  }
  return err_or_devices.get().size();
}

const CUDADevice &CUDADevice::cast(const Device &device) {
  return static_cast<const CUDADevice &>(device);
};

ErrorOr<const CUDADevice &> CUDADevice::get(int id) {
  ErrorOr<const DeviceList &> err_or_devices = deviceList();
  if (err_or_devices) {
    return err_or_devices.takeError();
  }
  return *err_or_devices.get()[id];
}

ErrorOr<const CUDADevice::DeviceList &> CUDADevice::deviceList() {
  // Initialization of these two statics only happens once on the first call. We
  // ensure this by initializing the err using a lambda.
  static DeviceList devices;
  static const ErrorOrSuccess err = [&]() -> ErrorOrSuccess {
    ErrorOr<const CUDADriver &> err_driver = CUDADriver::get();
    if (err_driver) {
      RAISE_ERROR(llvm::Twine("CUDA Driver initialization failed: ") +
                  err_driver.getError());
    }
    int count = 0;
    CUDA_CHECK_ERR(err_driver.get().DeviceGetCount(&count));

    char tmp_name[256];
    int tmp_alignment;
    CUdevice tmp_device;

    for (int i = 0; i < count; i++) {
      CUDA_CHECK_ERR(err_driver.get().DeviceGet(&tmp_device, i));
      CUDA_CHECK_ERR(err_driver.get().DeviceGetName(tmp_name, sizeof(tmp_name),
                                                    tmp_device));
      CUDA_CHECK_ERR(err_driver.get().DeviceGetAttribute(
          &tmp_alignment, CUdevice_attribute::TEXTURE_ALIGNMENT, tmp_device));
      devices.emplace_back(CUDADeviceRef::create(
          err_driver.get(), i, tmp_device, tmp_name, tmp_alignment));
    }
    return success();
  }();

  if (err) {
    RAISE_ERROR(err.getError());
  }
  return devices;
}

// ===== CUDADeviceEvent ===== //
CUDADeviceEvent::CUDADeviceEvent(const CUDADeviceContext &context,
                                 CUevent event)
    : DeviceEvent(context), event_(event) {}

CUDADeviceEvent::~CUDADeviceEvent() {
  const auto &cudaContext = CUDADeviceContext::cast(context());
  CUDA_WITH_CTX(cudaContext);
  CUDA_CHECK(cudaContext.cudaDriver().EventDestroy(event_));
}

ErrorOrSuccess CUDADeviceEvent::synchronize() const {
  const auto &cudaContext = CUDADeviceContext::cast(context());
  CUDA_WITH_CTX_ERR(cudaContext);
  CUDA_CHECK_ERR(cudaContext.cudaDriver().EventSynchronize(event_));
  return success();
}

CUevent CUDADeviceEvent::cudaEvent() const { return event_; }

const CUDADeviceEvent &CUDADeviceEvent::cast(const DeviceEvent &event) {
  return static_cast<const CUDADeviceEvent &>(event);
}

// ===== CUDADeviceStream ===== //
CUDADeviceStream::CUDADeviceStream(const CUDADeviceContext &context,
                                   CUstream stream)
    : DeviceStream(context), stream_(stream) {}

ErrorOrSuccess CUDADeviceStream::synchronize() const {
  const auto &cudaContext = CUDADeviceContext::cast(context());

  CUDA_WITH_CTX_ERR(cudaContext);
  CUDA_CHECK_ERR(cudaContext.cudaDriver().StreamSynchronize(stream_));
  return success();
}

CUstream CUDADeviceStream::cudaStream() const { return stream_; }

const CUDADeviceStream *
CUDADeviceStream::verified_cast(const DeviceStream *stream) {
  if (stream->context().device().api() != Device::cudaAPI) {
    return nullptr;
  }
  return static_cast<const CUDADeviceStream *>(stream);
}

// ===== CUDADeviceFunctionArguments ===== //
CUDADeviceFunctionArguments::CUDADeviceFunctionArguments(
    const CUDADeviceFunction &fn)
    : DeviceFunction::Arguments(fn) {}

void CUDADeviceFunctionArguments::pushArg(int index, const uint64_t &val) {
  argValues_.push_back(reinterpret_cast<void *>(val));
}

void CUDADeviceFunctionArguments::pushArg(int index, const DeviceBuffer &val) {
  // Ensure buffer is preserved by increasing the reference count through
  // copy().
  assert(val.context().device().api() == Device::cudaAPI);
  refs_.emplace_back(DeviceBufferRef::copy(&val));
  argValues_.push_back(val.ptr());
}

std::vector<void *> CUDADeviceFunctionArguments::cudaArgs() const {
  std::vector<void *> launch_args;
  launch_args.reserve(argValues_.size());
  for (size_t i = 0; i < argValues_.size(); i++) {
    launch_args.push_back(const_cast<void **>(&argValues_[i]));
  }
  return launch_args;
}

const CUDADeviceFunctionArguments &
CUDADeviceFunctionArguments::cast(const DeviceFunction::Arguments &args) {
  return static_cast<const CUDADeviceFunctionArguments &>(args);
}

// ===== CUDAModuleFunction ===== //
CUDAModuleFunction::CUDAModuleFunction(const std::string name,
                                       const CUDADriver &driver,
                                       CUmodule module, CUfunction function)
    : driver_(driver), module_(module), function_(function),
      name_(std::move(name)) {}

CUDAModuleFunction::~CUDAModuleFunction() {
  CUDA_CHECK(cudaDriver().ModuleUnload(module_));
}

const std::string &CUDAModuleFunction::name() const { return name_; }
const CUDADriver &CUDAModuleFunction::cudaDriver() const { return driver_; }
CUmodule CUDAModuleFunction::cudaModule() const { return module_; }
CUfunction CUDAModuleFunction::cudaFunction() const { return function_; }

// ===== CUDADeviceFunction ===== //
CUDADeviceFunction::CUDADeviceFunction(
    const CUDADeviceContext &context,
    const DeviceFunction::SignatureBuilder &signature,
    CUDAModuleFunctionRef modFunc)
    : DeviceFunction(context, signature), moduleFunction_(std::move(modFunc)) {}
CUDADeviceFunction::~CUDADeviceFunction() {}

DeviceFunction::Arguments *
CUDADeviceFunction::allocateArguments(const DeviceFunction &fn) const {
  return new CUDADeviceFunctionArguments(CUDADeviceFunction::cast(fn));
}

ErrorOrSuccess CUDADeviceFunction::cuda_callKernel(
    uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
    uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    const CUDADeviceFunctionArguments &args, CUstream stream) const {
  CUDA_CHECK_ERR(moduleFunction_->cudaDriver().LaunchKernel(
      moduleFunction_->cudaFunction(), gridX, gridY, gridZ, blockX, blockY,
      blockZ, sharedMemBytes, stream, args.cudaArgs().data(), nullptr));
  return success();
}

ErrorOrSuccess CUDADeviceFunction::cudaCallKernelDirect(
    uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
    uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    CUlaunchAttribute *attrs, uint32_t numAttrs, void **args,
    CUstream stream) const {
  CUlaunchConfig config;
  config.gridDimX = gridX;
  config.gridDimY = gridY;
  config.gridDimZ = gridZ;
  config.blockDimX = blockX;
  config.blockDimY = blockY;
  config.blockDimZ = blockZ;
  config.sharedMemBytes = sharedMemBytes;
  config.hStream = stream;
  config.attrs = attrs;
  config.numAttrs = numAttrs;

  CUDA_CHECK_ERR(moduleFunction_->cudaDriver().LaunchKernelEx(
      &config, moduleFunction_->cudaFunction(), args, nullptr));
  return success();
}

const std::string &CUDADeviceFunction::name() const {
  return moduleFunction_->name();
}

ErrorOrSuccess CUDADeviceFunction::copyToNamedConstant(const char *name,
                                                       const void *data,
                                                       size_t byteSize) const {
  CUdeviceptr device_ptr;
  size_t device_size;
  const auto &cudaContext = CUDADeviceContext::cast(context());
  CUDA_CHECK_ERR(cudaContext.cudaDriver().ModuleGetGlobal(
      &device_ptr, &device_size, moduleFunction_->cudaModule(), name));

  if (device_size < byteSize) {
    RAISE_ERROR("byteSize too big");
  }

  CUDA_CHECK_ERR(cudaContext.cudaDriver().MemcpyHtoDAsync(
      device_ptr, data, byteSize, cudaContext.cudaStream()));
  return success();
}

ErrorOr<int32_t> CUDADeviceFunction::getAttribute(int32_t code) const {
  int32_t result;
  CUDA_CHECK_ERR(moduleFunction_->cudaDriver().FuncGetAttribute(
      &result, static_cast<CUfunction_attribute>(code),
      moduleFunction_->cudaFunction()));
  return result;
}

const CUDADeviceFunction &CUDADeviceFunction::cast(const DeviceFunction &func) {
  return static_cast<const CUDADeviceFunction &>(func);
}

// ===== CUDADeviceTimer ===== //
CUDADeviceTimer::CUDADeviceTimer(const CUDADeviceContext &context,
                                 CUevent start, CUevent stop)
    : DeviceTimer(context), start_(start), stop_(stop) {}

CUDADeviceTimer::~CUDADeviceTimer() {
  const auto &cudaContext = CUDADeviceContext::cast(context());

  CUDA_CHECK(cudaContext.cudaDriver().EventDestroy(start_));
  CUDA_CHECK(cudaContext.cudaDriver().EventDestroy(stop_));
}

CUevent CUDADeviceTimer::cudaStartEvent() const { return start_; }
CUevent CUDADeviceTimer::cudaStopEvent() const { return stop_; }

const CUDADeviceTimer &CUDADeviceTimer::cast(const DeviceTimer &timer) {
  return static_cast<const CUDADeviceTimer &>(timer);
}

// ===== CUDADeviceContext ===== //
CUDADeviceContext::CUDADeviceContext(const CUDADevice &device,
                                     CUcontext context, CUstream stream,
                                     void *bufferCacheBase,
                                     size_t bufferCacheSize)
    : DeviceContext(device, std::make_unique<BufferCache>(
                                bufferCacheBase, bufferCacheSize,
                                device.allocationAlignment())),
      driver_(device.cudaDriver()), context_(context), stream_(stream) {
  ErrorOr<Config> configOr = Config::open();
  if (!configOr.isError()) {
    ErrorOr<bool> envVal =
        configOr->getValueAsBool("device-context.log_load_function", false);
    if (!envVal.isError())
      logLoadFunction_ = *envVal;
    envVal = configOr->getValueAsBool("device-context.log_verbose", false);
    if (!envVal.isError())
      logVerbose_ = *envVal;
  }
}

CUDADeviceContext::~CUDADeviceContext() {
  {
    // Make sure we have the correct context setup for the cleanup operations.
    // We will exit the CUDAContextScope before we force a push of the context
    // to be destroyed.
    CUDA_WITH_CTX(*this);
    // Destroy the function map before destroying the context.
    funcMap_.clear();
    CUDA_CHECK(driver_.StreamSynchronize(stream_));
    bufferCache().clear();
    // We do not need to synchronize on possible work queued on this stream.
    // Resources will be released automatically once work has been completed.
    CUDA_CHECK(driver_.StreamDestroy(stream_));
  }
}

ErrorOrSuccess CUDADeviceContext::memAlloc(void **ptr, size_t byteSize,
                                           SyncMode mode) const {
  assert(byteSize != 0);
  CUDA_WITH_CTX_ERR(*this);
  if (mode == ASYNC) {
    CUDA_CHECK_ERR(driver_.MemAllocAsync(ptr, byteSize, stream_));
  } else {
    CUDA_CHECK_ERR(driver_.MemAlloc(ptr, byteSize));
  }
  return success();
}

ErrorOrSuccess CUDADeviceContext::memFree(void *ptr) const {
  CUDA_WITH_CTX(*this);
  CUDA_CHECK_ERR(driver_.MemFreeAsync(ptr, cudaStream()));

  return success();
}

ErrorOr<void *> CUDADeviceContext::mallocHost(size_t size) const {
  void *ptr;
  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.MemAllocHost(&ptr, size));
  return ptr;
}

ErrorOrSuccess CUDADeviceContext::freeHost(void *ptr) const {
  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.MemFreeHost(ptr));
  return success();
}

ErrorOrSuccess CUDADeviceContext::copyToDevice(const DeviceBuffer &dst,
                                               const void *src,
                                               SyncMode mode) const {
  CHECK_IS_CUDA_ERR(dst.context().device());
  CUDA_WITH_CTX_ERR(*this);

  CUdeviceptr dst_ptr = dst.ptr();
  size_t byteSize = dst.byteSize();
  if (mode == ASYNC) {
    CUDA_CHECK_ERR(driver_.MemcpyHtoDAsync(dst_ptr, src, byteSize, stream_));
  } else {
    CUDA_CHECK_ERR(driver_.MemcpyHtoD(dst_ptr, src, byteSize));
  }
  return success();
}

ErrorOrSuccess CUDADeviceContext::copyFromDevice(void *dst,
                                                 const DeviceBuffer &src,
                                                 SyncMode mode) const {
  CHECK_IS_CUDA_ERR(src.context().device());
  CUDA_WITH_CTX_ERR(*this);

  CUdeviceptr src_ptr = src.ptr();
  size_t byteSize = src.byteSize();
  if (mode == ASYNC) {
    CUDA_CHECK_ERR(driver_.MemcpyDtoHAsync(dst, src_ptr, byteSize, stream_));
  } else {
    CUDA_CHECK_ERR(driver_.MemcpyDtoH(dst, src_ptr, byteSize));
  }
  return success();
}

ErrorOrSuccess CUDADeviceContext::copyDeviceToDevice(const DeviceBuffer &dst,
                                                     const DeviceBuffer &src,
                                                     SyncMode mode) const {
  CHECK_IS_CUDA_ERR(src.context().device());
  CHECK_IS_CUDA_ERR(dst.context().device());
  CUDA_WITH_CTX_ERR(*this);

  const auto &src_ctx = CUDADeviceContext::cast(src.context());
  const auto &dst_ctx = CUDADeviceContext::cast(dst.context());

  if (this->context_ != dst_ctx.context_) {
    RAISE_ERROR("device context does not match context of dst");
  }

  CUdeviceptr src_ptr = src.ptr();
  CUdeviceptr dst_ptr = dst.ptr();
  size_t dstByteSize = dst.byteSize();
  size_t srcByteSize = src.byteSize();
  if (srcByteSize < dstByteSize) {
    RAISE_ERROR("not enough data in src");
  }
  if (src_ctx.context_ != dst_ctx.context_) {
    if (mode == ASYNC) {
      CUDA_CHECK_ERR(driver_.MemcpyPeerAsync(dst_ptr, dst_ctx.context_, src_ptr,
                                             src_ctx.context_, dstByteSize,
                                             stream_));
    } else {
      CUDA_CHECK_ERR(driver_.MemcpyPeer(dst_ptr, dst_ctx.context_, src_ptr,
                                        src_ctx.context_, dstByteSize));
    }
  } else {
    if (mode == ASYNC) {
      CUDA_CHECK_ERR(
          driver_.MemcpyDtoDAsync(dst_ptr, src_ptr, dstByteSize, stream_));
    } else {
      CUDA_CHECK_ERR(driver_.MemcpyDtoD(dst_ptr, src_ptr, dstByteSize));
    }
  }
  return success();
}

ErrorOrSuccess CUDADeviceContext::setMemory(const DeviceBuffer &dst,
                                            uint32_t val, size_t valSize,
                                            SyncMode mode) const {
  CHECK_IS_CUDA_ERR(dst.context().device());
  CUDA_WITH_CTX_ERR(*this);
  void *ptr = dst.ptr();
  size_t byteSize = dst.byteSize();
  if (valSize == 1) {
    if (mode == ASYNC) {
      CUDA_CHECK_ERR(driver_.MemsetD8Async(ptr, static_cast<uint8_t>(val),
                                           byteSize, stream_));
    } else {
      CUDA_CHECK_ERR(
          driver_.MemsetD8(ptr, static_cast<uint8_t>(val), byteSize));
    }
  } else if (valSize == 2) {
    if ((byteSize & 1) != 0) {
      return Error("byteSize of buffer is misaliged");
    }
    size_t len = byteSize >> 1;
    if (mode == ASYNC) {
      CUDA_CHECK_ERR(driver_.MemsetD16Async(ptr, static_cast<uint16_t>(val),
                                            len, stream_));
    } else {
      CUDA_CHECK_ERR(driver_.MemsetD16(ptr, static_cast<uint16_t>(val), len));
    }
  } else {
    if (valSize != 4) {
      return Error("unsupported valSize");
    }
    if ((byteSize & 3) != 0) {
      return Error("byteSize of buffer is misaliged");
    }
    size_t len = byteSize >> 2;
    if (mode == ASYNC) {
      CUDA_CHECK_ERR(driver_.MemsetD32Async(ptr, val, len, stream_));
    } else {
      CUDA_CHECK_ERR(driver_.MemsetD32(ptr, val, len));
    }
  }
  return success();
}

static ErrorOr<std::pair<SmallVector<CUjit_option>, SmallVector<uintptr_t>>>
setupCompilationOptions(StringRef debugLevel, int32_t optimizationLevel) {
  static constexpr int kDefaultOptimizationLevel = 4;

  SmallVector<CUjit_option> optionKind;
  SmallVector<uintptr_t> optionValues;

  optionKind.reserve(2);
  optionValues.reserve(2);

  // The default optimization level for NVIDIA GPUs is 4.
  if (optimizationLevel < 0)
    optimizationLevel = kDefaultOptimizationLevel;

  if (debugLevel == "full") {
    // Note that the PTX compiler complains if debugLevel=full and optimization
    // level is something other than 0 or 4 (the default).
    if (!llvm::is_contained({0, kDefaultOptimizationLevel},
                            optimizationLevel)) {
      RAISE_ERROR("debugLevel of \"full\" requires optimizationLevel of "
                  "zero (or default)");
    }

    optionKind.push_back(CUjit_option::JIT_GENERATE_DEBUG_INFO);
    optionValues.push_back(1);
  } else if (debugLevel == "line-tables") {
    optionKind.push_back(CUjit_option::JIT_GENERATE_LINE_INFO);
    optionValues.push_back(1);
  }

  if (optimizationLevel != kDefaultOptimizationLevel) {
    optionKind.push_back(CUjit_option::JIT_OPTIMIZATION_LEVEL);
    optionValues.push_back(optimizationLevel);
  }

  return std::make_pair(optionKind, optionValues);
}

ErrorOr<M::BufferRef>
CUDADeviceContext::compileFunction(StringRef ptx, StringRef debugLevel,
                                   int32_t optimizationLevel) const {
  auto errorOrOpts = setupCompilationOptions(debugLevel, optimizationLevel);
  if (errorOrOpts)
    return errorOrOpts.takeError();
  auto [optionKind, optionValues] = errorOrOpts.takeValue();

  CUDA_WITH_CTX_ERR(*this);

  CUlinkState linkState;
  CUDA_CHECK_ERR(driver_.LinkCreate(optionKind.size(), optionKind.data(),
                                    (void **)optionValues.data(), &linkState));

  CUDA_CHECK_ERR(driver_.LinkAddData(
      linkState, CUjitInputType::CU_JIT_INPUT_PTX, (void *)ptx.data(),
      ptx.size(), "", 0, nullptr, nullptr));

  void *cubin;
  size_t cubinSize;
  CUDA_CHECK_ERR(driver_.LinkComplete(linkState, &cubin, &cubinSize));

  return M::BufferRef::create(StringRef((char *)cubin, cubinSize));
}

ErrorOr<DeviceFunctionRef> CUDADeviceContext::loadFunction(
    StringRef moduleName, StringRef functionName,
    const DeviceFunction::SignatureBuilder &signature, const void *data,
    int32_t maxDynamicSharedBytes, StringRef debugLevel,
    int32_t optimizationLevel) const {
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto iter = funcMap_.find(moduleName);
    if (iter != funcMap_.end()) {
      CUDAModuleFunctionRef module_func = iter->second;
      return CUDADeviceFunctionRef::create(*this, signature,
                                           std::move(module_func));
    }
  }

  if (logLoadFunction_) {
    llvm::errs() << "CUDADeviceContext::loadFunction(" << functionName
                 << (logVerbose_ ? "):\n" : ")")
                 << (logVerbose_ ? reinterpret_cast<const char *>(data) : "")
                 << "\n";
  }
  auto errorOrOpts = setupCompilationOptions(debugLevel, optimizationLevel);
  if (errorOrOpts)
    return errorOrOpts.takeError();
  auto [optionKind, optionValues] = errorOrOpts.takeValue();

  CUmodule module;
  CUDA_WITH_CTX_ERR(*this);

  CUDA_CHECK_ERR(driver_.ModuleLoadDataEx(
      &module, data, optionKind.size(), optionKind.data(),
      reinterpret_cast<void **>(optionValues.data())));

  CUfunction function;
  CUDA_CHECK_ERR_MSG(
      driver_.ModuleGetFunction(&function, module, functionName.data()),
      "function name: " + functionName.str());

  if (maxDynamicSharedBytes >= 0) {
    CUDA_CHECK_ERR(driver_.FuncSetAttribute(
        function,
        CUfunction_attribute::FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        maxDynamicSharedBytes));
  }

  // Allocate func outside of the lock region to reduce the time spent under
  // lock. If this functionName has been registered concurrently, then we will
  // release this ref outside of the lock region before returning.
  auto module_func = CUDAModuleFunctionRef::create(
      std::string(functionName), this->cudaDriver(), module, function);
  CUDADeviceFunctionRef result;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto iter = funcMap_.find(moduleName);
    if (iter != funcMap_.end()) {
      result = CUDADeviceFunctionRef::create(
          *this, signature,
          CUDAModuleFunctionRef::copy(iter->second.getPointer()));
    } else {
      funcMap_.insert({moduleName, module_func});
      result = CUDADeviceFunctionRef::create(*this, signature,
                                             std::move(module_func));
    }
  }
  return result;
}

ErrorOrSuccess CUDADeviceContext::enqueueFunctionExec(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    const DeviceFunction::Arguments &args) const {
  if (args.hasError()) {
    RAISE_ERROR(args.getError().getError());
  }

  const auto &cuda_func = CUDADeviceFunction::cast(func);

  CUDA_WITH_CTX_ERR(*this);
  return cuda_func.cuda_callKernel(
      gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
      CUDADeviceFunctionArguments::cast(args), stream_);
}

ErrorOrSuccess CUDADeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args) const {
  const auto &cuda_func = CUDADeviceFunction::cast(func);

  CUDA_WITH_CTX_ERR(*this);
  return cuda_func.cudaCallKernelDirect(
      gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
      reinterpret_cast<CUlaunchAttribute *>(attrs), numAttrs, args, stream_);
}

ErrorOr<DeviceStreamRef> CUDADeviceContext::stream() const {
  return CUDADeviceStreamRef::create(*this, stream_);
}

ErrorOrSuccess CUDADeviceContext::synchronize() const {
  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.StreamSynchronize(stream_));
  return success();
}

ErrorOr<DeviceEventRef> CUDADeviceContext::enqueueEvent() const {
  CUDA_WITH_CTX_ERR(*this);
  CUevent event;
  CUDA_CHECK_ERR(driver_.EventCreate(&event, 0));
  CUDA_CHECK_ERR(driver_.EventRecord(event, stream_));
  return CUDADeviceEventRef::create(*this, event);
}

ErrorOrSuccess
CUDADeviceContext::enqueueWaitForEvent(const DeviceEvent &event) const {
  const auto &cudaEvent = CUDADeviceEvent::cast(event);
  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.StreamWaitEvent(stream_, cudaEvent.cudaEvent(), 0));
  return success();
}

ErrorOr<DeviceTimerRef> CUDADeviceContext::startTimer() const {
  CUevent start;
  CUevent stop;

  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.EventCreate(&start, 0));
  CUDA_CHECK_ERR(driver_.EventCreate(&stop, 0));

  auto timer = CUDADeviceTimerRef::create(*this, start, stop);

  CUDA_CHECK_ERR(driver_.EventRecord(timer->cudaStartEvent(), stream_));

  return timer;
}

ErrorOr<std::chrono::nanoseconds>
CUDADeviceContext::stopTimer(const DeviceTimer &timer) const {
  const auto &cuda_timer = CUDADeviceTimer::cast(timer);

  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.EventRecord(cuda_timer.cudaStopEvent(), stream_));
  CUDA_CHECK_ERR(driver_.EventSynchronize(cuda_timer.cudaStopEvent()));

  float millis;
  CUDA_CHECK_ERR(driver_.EventElapsedTime(&millis, cuda_timer.cudaStartEvent(),
                                          cuda_timer.cudaStopEvent()));

  constexpr double nanos_per_millisecond = 1000.0 * 1000.0;
  double nanos = static_cast<double>(millis) * nanos_per_millisecond;
  return std::chrono::nanoseconds(static_cast<int64_t>(nanos));
}

ErrorOrSuccess
CUDADeviceContext::enablePeerAccess(const DeviceContext &peer) const {
  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.CtxEnablePeerAccess(
      CUDADeviceContext::cast(peer).cudaContext(), 0));
  return success();
}

ErrorOr<DeviceContext::MemInfo> CUDADeviceContext::getMemoryInfo() const {
  MemInfo mem_info;
  CUDA_WITH_CTX(*this);
  CUDA_CHECK_ERR(driver_.GetMemInfo(&mem_info.free, &mem_info.total));
  return mem_info;
}

const CUDADriver &CUDADeviceContext::cudaDriver() const { return driver_; }

CUstream CUDADeviceContext::cudaStream() const { return stream_; }

CUcontext CUDADeviceContext::cudaContext() const { return context_; }

ErrorOrSuccess CUDADeviceContext::cudaTensorMapEncodeTiled(
    void *tensorMap, CUtensorMapDataType tensorDataType, uint32_t tensorRank,
    const DeviceBuffer &globalAddress, const uint64_t *globalDim,
    const uint64_t *globalStrides, const uint32_t *boxDim,
    const uint32_t *elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) const {
  CHECK_IS_CUDA_ERR(globalAddress.context().device());
  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.TensorMapEncodeTiled(
      tensorMap, tensorDataType, tensorRank, globalAddress.ptr(), globalDim,
      globalStrides, boxDim, elementStrides, interleave, swizzle, l2Promotion,
      oobFill));
  return success();
}

const CUDADeviceContext &CUDADeviceContext::cast(const DeviceContext &ctx) {
  return static_cast<const CUDADeviceContext &>(ctx);
}

const CUDADeviceContext *
CUDADeviceContext::verified_cast(const DeviceContext *ctx) {
  if (ctx->device().api() != CUDADevice::api) {
    return nullptr;
  }
  return static_cast<const CUDADeviceContext *>(ctx);
}

} // namespace M::AsyncRT
