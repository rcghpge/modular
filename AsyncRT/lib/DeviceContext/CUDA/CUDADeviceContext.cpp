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
#include "Support/ErrorOr.h"
#include "Support/PlatformUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"
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
      return formatCUDAError(__FILE__, __LINE__, __FUNCTION__, cuda_result);   \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_ERR_MSG(cuda_call, msg)                                     \
  do {                                                                         \
    CUDADriver::Result cuda_result = cuda_call;                                \
    if (cuda_result != CUDADriver::SUCCESS) {                                  \
      return formatCUDAError(__FILE__, __LINE__, __FUNCTION__, cuda_result,    \
                             msg);                                             \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(cuda_call)                                                  \
  do {                                                                         \
    CUDADriver::Result cuda_result = cuda_call;                                \
    if (cuda_result != CUDADriver::SUCCESS) {                                  \
      M::Error err =                                                           \
          formatCUDAError(__FILE__, __LINE__, __FUNCTION__, cuda_result);      \
      /* We are not in a context where we can return an Error here. So we   */ \
      /* print the message and abort().                                     */ \
      llvm::errs() << err << "\n";                                             \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define RAISE_ERROR(msg)                                                       \
  do {                                                                         \
    return DeviceContext::formatError(__FILE__, __LINE__, __FUNCTION__, msg);  \
  } while (0)

#define ASSIGN_OR_RAISE(var, call)                                             \
  auto _err_or_var_ = call;                                                    \
  if (_err_or_var_.isError()) {                                                \
    return DeviceContext::formatError(__FILE__, __LINE__, __FUNCTION__,        \
                                      _err_or_var_.getError());                \
  }                                                                            \
  var = _err_or_var_.takeValue()

#define DEFINE_OR_RAISE(type, var, call)                                       \
  auto _err_or_##var = call;                                                   \
  if (_err_or_##var.isError()) {                                               \
    return DeviceContext::formatError(__FILE__, __LINE__, __FUNCTION__,        \
                                      _err_or_##var.getError());               \
  }                                                                            \
  type var = _err_or_##var.takeValue()

#define ERR_CHECK(err_call)                                                    \
  do {                                                                         \
    auto err_result = (err_call);                                              \
    if (err_result.isError()) {                                                \
      M::Error err = DeviceContext::formatError(                               \
          __FILE__, __LINE__, __FUNCTION__, err_result.getError());            \
      /* We are not in a context where we can return an Error here. So we   */ \
      /* print the message and abort().                                     */ \
      llvm::errs() << err << "\n";                                             \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define CUDA_WITH_CTX_ERR(ctx)                                                 \
  CUDAContextScope __cudaWithContext((ctx).cudaContext(), (ctx).cudaDriver()); \
  CUDA_CHECK_ERR(__cudaWithContext.status());

#define CUDA_WITH_CTX(ctx)                                                     \
  CUDAContextScope __cudaWithContext((ctx).cudaContext(), (ctx).cudaDriver()); \
  CUDA_CHECK(__cudaWithContext.status());

#define CUDA_WITH_DEV_ERR(dev)                                                 \
  CUDAContextScope __cudaWithContext(*(dev).cudaContext(),                     \
                                     (dev).cudaDriver());                      \
  CUDA_CHECK_ERR(__cudaWithContext.status());

#define CUDA_WITH_DEV(dev)                                                     \
  CUDAContextScope __cudaWithContext(*(dev).cudaContext(),                     \
                                     (dev).cudaDriver());                      \
  CUDA_CHECK(__cudaWithContext.status());

#define CHECK_IS_CUDA_ERR(dev)                                                 \
  do {                                                                         \
    if ((dev).api() != Device::cudaAPI) {                                      \
      RAISE_ERROR("Not a CUDA device.");                                       \
    }                                                                          \
  } while (0)

namespace M::AsyncRT {

static Error formatCUDAError(const char *file, int line, const char *func,
                             CUDADriver::Result result,
                             std::string extra_msg = "") {
  ErrorOr<const CUDADriver &> errDriver = CUDADriver::get();
  const char *errName = nullptr;
  const char *errStr = nullptr;

  if (!errDriver) {
    // Ignore errors and use nullptr of char* instead.
    errDriver->GetErrorName(result, &errName);
    errDriver->GetErrorString(result, &errStr);
  }

  if (!extra_msg.empty()) {
    extra_msg = std::string(" - ") + extra_msg;
  }

  if ((errName != nullptr) && (errStr != nullptr)) {
    return DeviceContext::formatError(file, line, func,
                                      llvm::Twine("CUDA call failed: ") +
                                          errName + " (" + errStr + ")" +
                                          extra_msg);
  } else if ((errName != nullptr) || (errStr != nullptr)) {
    return DeviceContext::formatError(
        file, line, func,
        llvm::Twine("CUDA call failed: ") +
            (errName != nullptr ? errName : errStr) + extra_msg);
  } else {
    return DeviceContext::formatError(file, line, func,
                                      llvm::Twine("CUDA call failed: ") +
                                          llvm::Twine(result) + extra_msg);
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
  int driverVersion = 0;
  CUDA_CHECK_ERR(driver_.DriverGetVersion(&driverVersion));
  return driverVersion;
}

// The value returned by `cuDriverGetVersion_` from the CUDA API is an integer
// with the driver version embedded. The major version number is in the
// thousands place and the minor version number is in the hundreds place, scaled
// up by ten.
static std::string driverVersionToString(int driver_version) {
  int majorVersion = driver_version / 1000;
  int minorVersion = (driver_version % 1000) / 10;

  return (llvm::Twine(majorVersion) + "." + llvm::Twine(minorVersion)).str();
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

  int driverVersion = 0;
  CUDA_CHECK_ERR(driver_.DriverGetVersion(&driverVersion));
  if (driverVersion < MIN_CUDA_DRIVER_VERSION) {
    RAISE_ERROR(
        "MAX is not supported using current GPU driver. MAX "
        "requires a minimum driver version of " +
        llvm::Twine(MIN_NVIDIA_DRIVER_VERSION) + " and CUDA version " +
        driverVersionToString(MIN_CUDA_DRIVER_VERSION) +
        ". See https://modul.ar/nvidia-driver-version for "
        "details about how to install the proper NVIDIA driver. The current "
        "CUDA version MAX detected is: " +
        driverVersionToString(driverVersion));
  }

  return success();
}

ErrorOr<int> CUDADevice::computeCapability() const {
  int computeMajor = 0, computeMinor = 0;
  CUDA_CHECK_ERR(driver_.DeviceGetAttribute(
      &computeMajor, CUdevice_attribute::COMPUTE_CAPABILITY_MAJOR,
      cudaDevice()));
  CUDA_CHECK_ERR(driver_.DeviceGetAttribute(
      &computeMinor, CUdevice_attribute::COMPUTE_CAPABILITY_MINOR,
      cudaDevice()));
  return 10 * computeMajor + computeMinor;
}

ErrorOr<bool> CUDADevice::canAccess(const Device &peer) const {
  if (id() == peer.id()) {
    // Always disable self access (same behaviour as cuDeviceCanAccessPeer).
    return false;
  }

  int canAccessPeer = 0;
  CUDA_CHECK_ERR(driver_.DeviceCanAccessPeer(
      &canAccessPeer, cudaDevice(), CUDADevice::cast(peer).cudaDevice()));

  // Use support for native atomics as a proxy for NVLink here.
  // On a PCIe-link A10G system connected through a PCIe host bridge, we
  // observed that direct P2P access with stream-ordered allocations returns
  // zeros.
  // Furthermore, the very act of calling cuMemPoolSetAccess causes
  // `cuMemcpyPeerAsync` to return zeroes.
  // So force the cuMemcpyPeerAsync path and skipping MemPoolSetAccess by
  // returning cannot access here.
  int isNativeAtomicSupported = 0;
  CUDA_CHECK_ERR(driver_.DeviceGetP2PAttribute(
      &isNativeAtomicSupported,
      CUdevice_P2PAttribute::CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED,
      cudaDevice(), CUDADevice::cast(peer).cudaDevice()));

  return canAccessPeer && isNativeAtomicSupported;
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

  size_t cacheChunkSize = 0;
  size_t maxCacheSize = 0;
  if (bufferCacheSize != 0) {
    constexpr size_t KB = 1024; // NOLINT
    DeviceContext::MemInfo memInfo;
    CUDA_CHECK_ERR(
        this->cudaDriver().GetMemInfo(&memInfo.free, &memInfo.total));
    // Make the buffer cache disabled by setting percentages to zero.
    // Limit to 90% (default) of available memory.
    int64_t cachePercent =
        DeviceContext::configAsInt("buffer_cache_size_percent", 0);
    maxCacheSize = std::min(
        bufferCacheSize, static_cast<size_t>(llvm::alignToPowerOf2(
                             (memInfo.free * cachePercent) / 100, 256 * KB)));
    // Allocate chunks of 2% (default) of maximum cache size.
    int64_t chunkPercent =
        DeviceContext::configAsInt("buffer_cache_chunk_percent", 0);
    cacheChunkSize = std::min(
        maxCacheSize, static_cast<size_t>(llvm::alignToPowerOf2(
                          (maxCacheSize * chunkPercent) / 100, 256 * KB)));
  }

  return CUDADeviceContextRef::create(*this, *context, stream, cacheChunkSize,
                                      maxCacheSize);
}

ErrorOr<DeviceStreamRef>
CUDADevice::createExternalStream(void *externalStream) const {
  return CUDADeviceStreamRef::create(
      *this, reinterpret_cast<CUstream>(externalStream));
}

int CUDADevice::deviceCount() const {
  ErrorOr<const DeviceList &> errOrDevices = deviceList();
  if (errOrDevices.isError()) {
    return 0;
  }
  return errOrDevices.get().size();
}

const CUDADevice &CUDADevice::cast(const Device &device) {
  return static_cast<const CUDADevice &>(device);
};

ErrorOr<const CUDADevice &> CUDADevice::get(int id) {
  ErrorOr<const DeviceList &> errOrDevices = deviceList();
  if (errOrDevices) {
    return errOrDevices.takeError();
  }
  return *errOrDevices.get()[id];
}

ErrorOr<const CUDADevice::DeviceList &> CUDADevice::deviceList() {
  // Initialization of these two statics only happens once on the first call. We
  // ensure this by initializing the err using a lambda.
  static DeviceList devices;
  static const ErrorOrSuccess err = [&]() -> ErrorOrSuccess {
    ErrorOr<const CUDADriver &> errDriver = CUDADriver::get();
    if (errDriver) {
      RAISE_ERROR(llvm::Twine("CUDA Driver initialization failed: ") +
                  errDriver.getError());
    }
    int count = 0;
    CUDA_CHECK_ERR(errDriver.get().DeviceGetCount(&count));

    char tmpName[256];
    int tmpAlignment;
    CUdevice tmpDevice;

    for (int i = 0; i < count; i++) {
      CUDA_CHECK_ERR(errDriver.get().DeviceGet(&tmpDevice, i));
      CUDA_CHECK_ERR(
          errDriver.get().DeviceGetName(tmpName, sizeof(tmpName), tmpDevice));
      CUDA_CHECK_ERR(errDriver.get().DeviceGetAttribute(
          &tmpAlignment, CUdevice_attribute::TEXTURE_ALIGNMENT, tmpDevice));
      devices.emplace_back(CUDADeviceRef::create(errDriver.get(), i, tmpDevice,
                                                 tmpName, tmpAlignment));
    }
    return success();
  }();

  if (err) {
    RAISE_ERROR(err.getError());
  }
  return devices;
}

// ===== CUDADeviceEvent ===== //
CUDADeviceEvent::CUDADeviceEvent(const CUDADevice &device, CUevent event)
    : DeviceEvent(device), event_(event) {}

CUDADeviceEvent::~CUDADeviceEvent() {
  const auto &cudaDevice = CUDADevice::cast(device());
  CUDA_WITH_DEV(cudaDevice);
  CUDA_CHECK(cudaDevice.cudaDriver().EventDestroy(event_));
}

ErrorOrSuccess CUDADeviceEvent::synchronize() const {
  const auto &cudaDevice = CUDADevice::cast(device());
  CUDA_WITH_DEV_ERR(cudaDevice);
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().EventSynchronize(event_));
  return success();
}

CUevent CUDADeviceEvent::cudaEvent() const { return event_; }

const CUDADeviceEvent &CUDADeviceEvent::cast(const DeviceEvent &event) {
  return static_cast<const CUDADeviceEvent &>(event);
}

// ===== CUDADeviceStream ===== //
CUDADeviceStream::CUDADeviceStream(const CUDADevice &device, CUstream stream)
    : DeviceStream(device), stream_(stream) {}

ErrorOrSuccess CUDADeviceStream::synchronize() const {
  const auto &cudaDevice = CUDADevice::cast(device());

  CUDA_WITH_DEV_ERR(cudaDevice);
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().StreamSynchronize(stream_));
  return success();
}

ErrorOr<DeviceEventRef> CUDADeviceStream::enqueueEvent() const {
  const auto &cudaDevice = CUDADevice::cast(device());
  CUevent event;
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().EventCreate(&event, 0));
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().EventRecord(event, stream_));
  return CUDADeviceEventRef::create(CUDADevice::cast(this->device()), event);
}

ErrorOrSuccess
CUDADeviceStream::enqueueWaitForEvent(const DeviceEvent &event) const {
  const auto &cudaEvent = CUDADeviceEvent::cast(event);
  const auto &cudaDevice = CUDADevice::cast(device());
  CUDA_WITH_DEV_ERR(cudaDevice);
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().StreamWaitEvent(
      stream_, cudaEvent.cudaEvent(), 0));
  return success();
}

void *CUDADeviceStream::ptr() const { return cudaStream(); }

CUstream CUDADeviceStream::cudaStream() const { return stream_; }

const CUDADeviceStream *
CUDADeviceStream::verified_cast(const DeviceStream *stream) {
  if (stream->device().api() != Device::cudaAPI) {
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
  std::vector<void *> launchArgs;
  launchArgs.reserve(argValues_.size());
  for (size_t i = 0; i < argValues_.size(); i++) {
    launchArgs.push_back(const_cast<void **>(&argValues_[i]));
  }
  return launchArgs;
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
  CUdeviceptr devicePtr;
  size_t deviceSize;
  const auto &cudaContext = CUDADeviceContext::cast(context());
  CUDA_CHECK_ERR(cudaContext.cudaDriver().ModuleGetGlobal(
      &devicePtr, &deviceSize, moduleFunction_->cudaModule(), name));

  if (deviceSize < byteSize) {
    RAISE_ERROR("byteSize too big");
  }

  CUDA_CHECK_ERR(cudaContext.cudaDriver().MemcpyHtoDAsync(
      devicePtr, data, byteSize, cudaContext.cudaStream()));
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
                                     size_t cacheChunkSize, size_t maxCacheSize)
    : DeviceContext(device, std::make_unique<BufferCache>(
                                *this, cacheChunkSize, maxCacheSize,
                                device.allocationAlignment())),
      driver_(device.cudaDriver()), context_(context), stream_(stream) {
  logLoadFunction_ = DeviceContext::configAsBool("log_load_function", false);
  logVerbose_ = DeviceContext::configAsBool("log_verbose", false);
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
    // Synchronize to ensure that all of the chunks have been freed.
    CUDA_CHECK(driver_.StreamSynchronize(stream_));
    // We do not need to synchronize on possible work queued on this stream.
    // Resources will be released automatically once work has been completed.
    CUDA_CHECK(driver_.StreamDestroy(stream_));
  }
}

ErrorOrSuccess CUDADeviceContext::memAlloc(void **ptr, size_t byteSize,
                                           AllocMode mode) const {
  assert(byteSize != 0);
  CUDA_WITH_CTX_ERR(*this);
  switch (mode) {
  case AllocMode::ASYNC: {
    CUDA_CHECK_ERR(driver_.MemAllocAsync(ptr, byteSize, stream_));
    break;
  }
  case AllocMode::HOST: {
    CUDA_CHECK_ERR(driver_.MemAllocHost(ptr, byteSize));
    break;
  }
  case AllocMode::SYNC: {
    CUDA_CHECK_ERR(driver_.MemAlloc(ptr, byteSize));
    break;
  }
  }
  return success();
}

ErrorOrSuccess CUDADeviceContext::memFree(void *ptr, bool isHost) const {
  CUDA_WITH_CTX(*this);
  if (isHost) {
    CUDA_CHECK_ERR(driver_.MemFreeHost(ptr));
  } else {
    CUDA_CHECK_ERR(driver_.MemFreeAsync(ptr, cudaStream()));
  }
  return success();
}

ErrorOrSuccess CUDADeviceContext::copyToDevice(const DeviceBuffer &dst,
                                               const void *src,
                                               SyncMode mode) const {
  CHECK_IS_CUDA_ERR(dst.context().device());
  CUDA_WITH_CTX_ERR(*this);

  CUdeviceptr dstPtr = dst.ptr();
  size_t byteSize = dst.byteSize();
  if (mode == ASYNC) {
    CUDA_CHECK_ERR(driver_.MemcpyHtoDAsync(dstPtr, src, byteSize, stream_));
  } else {
    CUDA_CHECK_ERR(driver_.MemcpyHtoD(dstPtr, src, byteSize));
  }
  return success();
}

ErrorOrSuccess CUDADeviceContext::copyFromDevice(void *dst,
                                                 const DeviceBuffer &src,
                                                 SyncMode mode) const {
  CHECK_IS_CUDA_ERR(src.context().device());
  CUDA_WITH_CTX_ERR(*this);

  CUdeviceptr srcPtr = src.ptr();
  size_t byteSize = src.byteSize();
  if (mode == ASYNC) {
    CUDA_CHECK_ERR(driver_.MemcpyDtoHAsync(dst, srcPtr, byteSize, stream_));
  } else {
    CUDA_CHECK_ERR(driver_.MemcpyDtoH(dst, srcPtr, byteSize));
  }
  return success();
}

ErrorOrSuccess CUDADeviceContext::copyDeviceToDevice(const DeviceBuffer &dst,
                                                     const DeviceBuffer &src,
                                                     SyncMode mode) const {
  CHECK_IS_CUDA_ERR(src.context().device());
  CHECK_IS_CUDA_ERR(dst.context().device());
  CUDA_WITH_CTX_ERR(*this);

  const auto &srcCtx = CUDADeviceContext::cast(src.context());
  const auto &dstCtx = CUDADeviceContext::cast(dst.context());

  if (this->context_ != dstCtx.context_) {
    RAISE_ERROR("device context does not match context of dst");
  }

  CUdeviceptr srcPtr = src.ptr();
  CUdeviceptr dstPtr = dst.ptr();
  size_t dstByteSize = dst.byteSize();
  size_t srcByteSize = src.byteSize();
  if (srcByteSize < dstByteSize) {
    RAISE_ERROR("not enough data in src");
  }
  if (srcCtx.context_ != dstCtx.context_) {
    if (mode == ASYNC) {
      // Ensure that we are waiting for the source data to be ready.
      DEFINE_OR_RAISE(DeviceEventRef, srcEvent, srcCtx.enqueueEvent());
      ERR_CHECK(dstCtx.enqueueWaitForEvent(*srcEvent));

      CUDA_CHECK_ERR(driver_.MemcpyPeerAsync(dstPtr, dstCtx.context_, srcPtr,
                                             srcCtx.context_, dstByteSize,
                                             stream_));

      // Wait for the copy to complete before proceeding with the source stream
      // (e.g. freeing or overwriting memory).
      DEFINE_OR_RAISE(DeviceEventRef, dstEvent, dstCtx.enqueueEvent());
      ERR_CHECK(srcCtx.enqueueWaitForEvent(*dstEvent));
    } else {
      CUDA_CHECK_ERR(driver_.MemcpyPeer(dstPtr, dstCtx.context_, srcPtr,
                                        srcCtx.context_, dstByteSize));
    }
  } else {
    if (mode == ASYNC) {
      CUDA_CHECK_ERR(
          driver_.MemcpyDtoDAsync(dstPtr, srcPtr, dstByteSize, stream_));
    } else {
      CUDA_CHECK_ERR(driver_.MemcpyDtoD(dstPtr, srcPtr, dstByteSize));
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

  // The Mojo compiler uses optimization level 3 as the highest optimization
  // level, but 4 is the highest for CUDA.
  if (optimizationLevel == 3)
    optimizationLevel = kDefaultOptimizationLevel;

  if (debugLevel == "full") {
    // Note that the PTX compiler complains if debugLevel=full and optimization
    // level is something other than 0 or 4 (the default).
    if (!llvm::is_contained({0, kDefaultOptimizationLevel},
                            optimizationLevel)) {
      RAISE_ERROR((("debugLevel of \"full\" requires optimizationLevel of "
                    "either '0' or '" +
                    llvm::Twine(kDefaultOptimizationLevel) + "' but got '" +
                    llvm::Twine(optimizationLevel) + "'"))
                      .str());
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
  // Creates a pending JIT linker invocation.
  CUDA_CHECK_ERR(driver_.LinkCreate(optionKind.size(), optionKind.data(),
                                    (void **)optionValues.data(), &linkState));

  // Add an input to a pending linker invocation.
  CUDA_CHECK_ERR(driver_.LinkAddData(
      linkState, CUjitInputType::CU_JIT_INPUT_PTX, (void *)ptx.data(),
      ptx.size(), "", 0, nullptr, nullptr));

  void *cubin;
  size_t cubinSize;
  // Completes the pending linker action and returns the cubin image
  // for the linked device code. The cubin is owned by state, so it
  // should be used before state is destroyed.
  // This call does not destroy state.
  CUDA_CHECK_ERR(driver_.LinkComplete(linkState, &cubin, &cubinSize));

  // Copy cubin to a BufferRef so that the original one can be destoried
  // with the link state to avoid leaking memory.
  auto result = M::Buffer::get(StringRef((char *)cubin, cubinSize));

  // Destroys state for a JIT linker invocation.
  CUDA_CHECK_ERR(driver_.LinkDestroy(linkState));
  return result;
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
      CUDAModuleFunctionRef moduleFunc = iter->second;
      return CUDADeviceFunctionRef::create(*this, signature,
                                           std::move(moduleFunc));
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
  auto moduleFunc = CUDAModuleFunctionRef::create(
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
      funcMap_.insert({moduleName, moduleFunc});
      result = CUDADeviceFunctionRef::create(*this, signature,
                                             std::move(moduleFunc));
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

  const auto &cudaFunc = CUDADeviceFunction::cast(func);

  CUDA_WITH_CTX_ERR(*this);
  return cudaFunc.cuda_callKernel(
      gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
      CUDADeviceFunctionArguments::cast(args), stream_);
}

ErrorOrSuccess CUDADeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args) const {
  const auto &cudaFunc = CUDADeviceFunction::cast(func);

  CUDA_WITH_CTX_ERR(*this);
  return cudaFunc.cudaCallKernelDirect(
      gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
      reinterpret_cast<CUlaunchAttribute *>(attrs), numAttrs, args, stream_);
}

void CUDADeviceContext::clearReadyEvents() const {
  std::lock_guard<std::mutex> lock(mu_);
  clearReadyEventsLocked();
}

void CUDADeviceContext::clearReadyEventsLocked() const {
  for (auto it = pendingEvents_.begin(); it != pendingEvents_.end();) {
    CUDADriver::Result ready = cudaDriver().EventQuery((*it)->cudaEvent());
    if (ready == CUDADriver::SUCCESS) {
      it = pendingEvents_.erase(it);
    } else {
      // Events were enqueued in order. We can stop at the first non-ready
      // event.
      break;
    }
  }
}

ErrorOr<DeviceStreamRef> CUDADeviceContext::stream() const {
  return CUDADeviceStreamRef::create(CUDADevice::cast(this->device()), stream_);
}

ErrorOrSuccess CUDADeviceContext::synchronize() const {
  CUDA_WITH_CTX_ERR(*this);
  // Attempt to clear the pending events before possibly blocking on the
  // stream synchronization.
  clearReadyEvents();
  CUDA_CHECK_ERR(driver_.StreamSynchronize(stream_));
  // Cleanup the remainder of the pending events.
  clearReadyEvents();
  return success();
}

ErrorOr<DeviceEventRef> CUDADeviceContext::enqueueEvent() const {
  CUDA_WITH_CTX_ERR(*this);
  CUevent event;
  CUDA_CHECK_ERR(driver_.EventCreate(&event, 0));
  CUDA_CHECK_ERR(driver_.EventRecord(event, stream_));
  auto result =
      CUDADeviceEventRef::create(CUDADevice::cast(this->device()), event);
  {
    std::lock_guard<std::mutex> lock(mu_);
    // First remove any outstanding ready events.
    clearReadyEventsLocked();
    pendingEvents_.push_back(result);
  }
  return result;
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
  const auto &cudaTimer = CUDADeviceTimer::cast(timer);

  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.EventRecord(cudaTimer.cudaStopEvent(), stream_));
  CUDA_CHECK_ERR(driver_.EventSynchronize(cudaTimer.cudaStopEvent()));

  float millis;
  CUDA_CHECK_ERR(driver_.EventElapsedTime(&millis, cudaTimer.cudaStartEvent(),
                                          cudaTimer.cudaStopEvent()));

  constexpr double nanosPerMillisecond = 1000.0 * 1000.0;
  double nanos = static_cast<double>(millis) * nanosPerMillisecond;
  return std::chrono::nanoseconds(static_cast<int64_t>(nanos));
}

ErrorOrSuccess
CUDADeviceContext::enablePeerAccess(const DeviceContext &peer) const {
  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.CtxEnablePeerAccess(
      CUDADeviceContext::cast(peer).cudaContext(), 0));

  // Also "enable peer access" for stream-ordered allocations.
  // Do so by allowing this device to access the peer's current memory pool.
  // NOTE: this only enables access to allocations from the peer's current
  // memory pool, so other memory pools won't be accessible.
  CUmemoryPool pool;
  CUDA_CHECK_ERR(driver_.DeviceGetMemPool(
      &pool, CUDADevice::cast(peer.device()).cudaDevice()));

  CUmemAccessDesc desc{};
  desc.location.type = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = device().id();
  desc.flags = CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUDA_CHECK_ERR(driver_.MemPoolSetAccess(pool, &desc, /*count=*/1));

  return success();
}

ErrorOr<DeviceContext::MemInfo> CUDADeviceContext::getMemoryInfo() const {
  MemInfo memInfo;
  CUDA_WITH_CTX(*this);
  // To get more accurate memory stats, ensure that at least all pending memory
  // allocations and frees have completed for this context's stream.
  CUDA_CHECK_ERR(driver_.StreamSynchronize(stream_));

  CUDA_CHECK_ERR(driver_.GetMemInfo(&memInfo.free, &memInfo.total));
  return memInfo;
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
