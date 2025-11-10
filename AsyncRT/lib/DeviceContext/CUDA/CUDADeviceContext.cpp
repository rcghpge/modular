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
#include "CUDACompilationDriver.h"
#include "CUDADriver.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/PlatformUtils.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Process.h"
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
    // No need to restore the previous context if no previous context is to be
    // restored, if the context was already the desired context or if there was
    // an error setting the context.
    if ((prevCtx_ == ctx_) || (prevCtx_ == nullptr) ||
        (status_ != CUDADriver::SUCCESS))
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

#define NVPTXCOMPILER_SAFE_CALL(nvptxcompiler_call)                            \
  do {                                                                         \
    CUDACompilationDriver::Result result = nvptxcompiler_call;                 \
    if (result != CUDACompilationDriver::SUCCESS) {                            \
      return formatCUDAError(__FILE__, __LINE__, __FUNCTION__, result);        \
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

#define ERR_CHECK_ERR(err_call)                                                \
  do {                                                                         \
    auto err_result = (err_call);                                              \
    if (err_result.isError()) {                                                \
      return DeviceContext::formatError(__FILE__, __LINE__, __FUNCTION__,      \
                                        err_result.getError());                \
    }                                                                          \
  } while (0)

#define CUDA_WITH_CTX_ERR(ctx)                                                 \
  ErrorOrSuccess __pendingErr = (ctx).checkPendingWork();                      \
  if (__pendingErr.isError()) {                                                \
    return __pendingErr.takeError();                                           \
  }                                                                            \
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

#define MAYBE_SYNC(ctx)                                                        \
  do {                                                                         \
    const CUDADeviceContext &cuda_ctx = (ctx);                                 \
    if (cuda_ctx.syncMode()) {                                                 \
      CUDA_CHECK_ERR(                                                          \
          cuda_ctx.cudaDriver().StreamSynchronize(cuda_ctx.cudaStream()));     \
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

#ifdef MODULAR_LINUX
static Error formatCUDAError(const char *file, int line, const char *func,
                             CUDACompilationDriver::Result result,
                             std::string extra_msg = "") {
  if (!extra_msg.empty()) {
    extra_msg = std::string(" - ") + extra_msg;
  }
  return DeviceContext::formatError(
      file, line, func,
      llvm::Twine("NVPTX Compiler call failed: ") + llvm::Twine(result) +
          extra_msg);
}
#endif

// ===== CUDADevice ===== //
CUDADevice::CUDADevice(const CUDADriver &driver, int id, CUdevice device,
                       CUcontext context, std::string name,
                       Device::MemInfo memInfo, size_t allocAlignment,
                       int64_t deviceMemoryLimit)
    : Device(api, std::move(name), id, memInfo, allocAlignment,
             deviceMemoryLimit),
      driver_(driver), device_(device), context_(context) {}

CUDADevice::~CUDADevice() {
  // Destroy the function map before destroying the context.
  clearFunctionCache();
  bufferCache().clear();
  if (context_ != nullptr)
    // We ignore the result here as there is not much we can do if it fails
    // since in most cases the CUDA driver is in the process of shutting down.
    cudaDriver().DevicePrimaryCtxRelease(cudaDevice());
}

const CUDADriver &CUDADevice::cudaDriver() const { return driver_; }

CUdevice CUDADevice::cudaDevice() const { return device_; }

ErrorOr<CUcontext> CUDADevice::cudaContext() const { return context_; }

ErrorOr<int> CUDADevice::getApiVersion() const {
  int driverVersion = 0;
  CUDA_CHECK_ERR(driver_.DriverGetVersion(&driverVersion));
  return driverVersion;
}

ErrorOr<DriverVersion &> CUDADevice::getDriverVersion() const {
  // Initialize the driver version once on first call of this method
  // or remember the error.
  static ErrorOr<std::string> errOrVersion = []() -> ErrorOr<std::string> {
    ErrorOr<const NVML &> errOrNVML = NVML::get();
    if (errOrNVML.isError()) {
      return errOrNVML.takeError();
    }
    ErrorOr<std::string> errOrVersion = errOrNVML->GetDriverVersion();
    if (errOrVersion.isError()) {
      return errOrVersion.takeError();
    }
    return *errOrVersion;
  }();

  // Propagate the error if getting the version produced an error.
  if (errOrVersion.isError()) {
    return Error(errOrVersion.getError());
  }
  // Return the version string.
  static DriverVersion version(*errOrVersion);
  return version;
}

// The value returned by `cuDriverGetVersion_` from the CUDA API is an integer
// with the API version embedded. The major version number is in the
// thousands place and the minor version number is in the hundreds place, scaled
// up by ten.
static std::string apiVersionToString(int apiVersion) {
  int majorVersion = apiVersion / 1000;
  int minorVersion = (apiVersion % 1000) / 10;

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
    RAISE_ERROR("MAX only supports CUDA Pascal architectures or higher");
  }

  int driverVersion = 0;
  CUDA_CHECK_ERR(driver_.DriverGetVersion(&driverVersion));
  if (driverVersion < MIN_CUDA_DRIVER_VERSION) {
    RAISE_ERROR(
        "MAX doesn't support your current NVIDIA GPU driver. MAX "
        "requires a minimum driver version of " +
        llvm::Twine(MIN_NVIDIA_DRIVER_VERSION) + " and CUDA version " +
        apiVersionToString(MIN_CUDA_DRIVER_VERSION) +
        ". The driver version MAX detected is: " +
        apiVersionToString(driverVersion) +
        ".\nFor more information, see https://modul.ar/gpu-requirements.");
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

ErrorOrSuccess CUDADevice::enablePeerAccess(const Device &peer) const {
  CUDA_WITH_DEV_ERR(*this);

  const CUDADevice &peerCudaDev = CUDADevice::cast(peer);
  ErrorOr<CUcontext> peerCtxOr = peerCudaDev.cudaContext();
  if (peerCtxOr.isError())
    return peerCtxOr.takeError();

  ErrorOr<bool> canAccessOr = canAccess(peerCudaDev);
  if (canAccessOr.isError())
    return canAccessOr.takeError();
  if (!*canAccessOr) {
    return Error("Cannot enable peer access from GPU " + Twine(id()) +
                 " to GPU " + Twine(peer.id()) +
                 ": hardware does not support P2P access");
  }

  // Tolerate "already enabled" as success to keep idempotent.
  CUDADriver::Result res = driver_.CtxEnablePeerAccess(*peerCtxOr, /*flags=*/0);
  if (res != CUDADriver::SUCCESS) {
    const char *name = nullptr;
    driver_.GetErrorName(res, &name);
    if (!(name && (Twine(name) == "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED")))
      return formatCUDAError(__FILE__, __LINE__, __FUNCTION__, res);
  }

  // Also allow access to the peer's stream-ordered allocation pool.
  CUmemoryPool pool;
  CUDA_CHECK_ERR(driver_.DeviceGetMemPool(&pool, peerCudaDev.cudaDevice()));

  CUmemAccessDesc desc{};
  desc.location.type = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = id();
  desc.flags = CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUDA_CHECK_ERR(driver_.MemPoolSetAccess(pool, &desc, /*count=*/1));

  return success();
}

ErrorOr<int> CUDADevice::getAttribute(Attribute attr) const {
  int value = 0;
  CUDA_CHECK_ERR(driver_.DeviceGetAttribute(
      &value, static_cast<CUdevice_attribute>(attr), cudaDevice()));
  return value;
}

ErrorOr<bool> CUDADevice::supportsMulticast() const {
  // Check if multicast memory is supported.
  int multicastSupported = 0;
  CUDA_CHECK_ERR(cudaDriver().DeviceGetAttribute(
      &multicastSupported, CUdevice_attribute::MULTICAST_SUPPORTED,
      cudaDevice()));
  return multicastSupported != 0;
}

ErrorOr<DeviceContextRef> CUDADevice::createContext() const {
  ErrorOr<CUcontext> context = cudaContext();
  if (context.isError())
    return context.takeError();

  // For the remainder of this method we use the context scope to ensure that
  // the context is properly removed when an error happens or then at the
  // return.
  CUDAContextScope withCtx(*context, this->cudaDriver());

  CUstream stream;
  CUDA_CHECK_ERR(this->cudaDriver().StreamCreate(&stream, 0));

  return CUDADeviceContextRef::create(*this, *context, stream);
}

ErrorOr<DeviceStreamRef>
CUDADevice::createExternalStream(void *externalStream) const {
  return CUDADeviceStreamRef::create(
      *this, reinterpret_cast<CUstream>(externalStream));
}

ErrorOr<int> CUDADevice::numberOfGPUs() {
  static ErrorOr<int> errOrCount = []() -> ErrorOr<int> {
    ErrorOr<const CUDADriver &> errDriver = CUDADriver::get();
    if (errDriver.isError())
      return errDriver.takeError();

    int result;
    CUDA_CHECK_ERR((*errDriver).DeviceGetCount(&result));
    return result;
  }();

  if (errOrCount.isError()) {
    return errOrCount.copy();
  }
  return *errOrCount;
}

int CUDADevice::deviceCount() const {
  ErrorOr<int> errOrCount = numberOfGPUs();
  if (errOrCount.isError()) {
    // Do not propagate an error.
    return 0;
  }
  return *errOrCount;
}

ErrorOrSuccess CUDADevice::chunkAlloc(void **ptr, size_t byteSize,
                                      bool isHost) const {
  assert(byteSize != 0);
  CUDA_WITH_DEV_ERR(*this);
  if (isHost) {
    CUDA_CHECK_ERR(driver_.MemAllocHost(ptr, byteSize));
  } else {
    CUDA_CHECK_ERR(driver_.MemAlloc(ptr, byteSize));
  }
  return success();
}

ErrorOrSuccess CUDADevice::chunkFree(void *ptr, bool isHost) const {
  CUDA_WITH_DEV_ERR(*this);
  if (isHost) {
    CUDA_CHECK_ERR(driver_.MemFreeHost(ptr));
  } else {
    CUDA_CHECK_ERR(driver_.MemFree(ptr));
  }
  return success();
}

ErrorOr<Device::MemInfo> CUDADevice::getMemoryInfo() const {
  Device::MemInfo memInfo;
  CUDA_WITH_DEV_ERR(*this);
  CUDA_CHECK_ERR(driver_.GetMemInfo(&memInfo.free, &memInfo.total));
  return memInfo;
}

ErrorOr<void *> CUDADevice::getCurrentDriverContext() const {
  CUcontext result = nullptr;
  CUDA_CHECK_ERR(driver_.CtxGetCurrent(&result));
  return result;
}
ErrorOrSuccess CUDADevice::setCurrentDriverContext(void *ctx) const {
  CUDA_CHECK_ERR(driver_.CtxSetCurrent(reinterpret_cast<CUcontext>(ctx)));
  return success();
}

const CUDADevice &CUDADevice::cast(const Device &device) {
  return static_cast<const CUDADevice &>(device);
}

ErrorOr<const CUDADevice &> CUDADevice::get(int id, int64_t deviceMemoryLimit) {
  // On first call initialize devices and err. devices is only set if there was
  // no error during setup.
  static std::mutex mu;
  static DeviceList *devices = nullptr;
  static const ErrorOrSuccess err = [&]() -> ErrorOrSuccess {
    ErrorOr<int> errOrCount = numberOfGPUs();
    if (errOrCount.isError()) {
      return errOrCount.takeError();
    }
    devices = new DeviceList(*errOrCount, nullptr);
    return success();
  }();

  // If we did get an error during setup, skip accessing the device and forward
  // the error.
  if (err.isError()) {
    RAISE_ERROR(err.getError());
  }

  assert(devices != nullptr);
  assert((id >= 0) && (static_cast<size_t>(id) < devices->size()));

  // Get or initialize the Device object.
  std::lock_guard<std::mutex> lock(mu);
  CUDADevice *device = (*devices)[id];
  if (device == nullptr) {
    ErrorOr<const CUDADriver &> errDriver = CUDADriver::get();
    if (errDriver) {
      RAISE_ERROR(llvm::Twine("CUDA Driver initialization failed: ") +
                  errDriver.getError());
    }
    const CUDADriver &driver = *errDriver;
    char name[256];
    int alignment;
    CUdevice cuDevice;
    CUcontext cuContext;

    CUDA_CHECK_ERR(driver.DeviceGet(&cuDevice, id));
    CUDA_CHECK_ERR(driver.DeviceGetName(name, sizeof(name), cuDevice));
    CUDA_CHECK_ERR(driver.DeviceGetAttribute(
        &alignment, CUdevice_attribute::TEXTURE_ALIGNMENT, cuDevice));

    // Use the primary context for this device, with the effect that we share
    // the same context with other users of the CUDA Runtime API.
    CUDA_CHECK_ERR(driver.DevicePrimaryCtxRetain(&cuContext, cuDevice));

    CUDAContextScope withCtx(cuContext, driver);
    Device::MemInfo memInfo;
    CUDA_CHECK_ERR(driver.GetMemInfo(&memInfo.free, &memInfo.total));

    device = new CUDADevice(driver, id, cuDevice, cuContext, name, memInfo,
                            alignment, deviceMemoryLimit);

    // Once the Device has been created remember it in the devices list.
    (*devices)[id] = device;
  }
  return *device;
}

void CUDADevice::eventCacheClear() const {
  std::lock_guard<std::mutex> lock(mu_);
  for (const auto &event : eventCache_) {
    // Ignore errors during destruction of device object.
    cudaDriver().EventDestroy(event);
  }
}

ErrorOr<CUevent> CUDADevice::eventCachePop() const {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!eventCache_.empty()) {
      CUevent result = std::move(eventCache_.back());
      eventCache_.pop_back();
      return result;
    }
  }
  CUevent event;
  CUDA_CHECK_ERR(
      cudaDriver().EventCreate(&event, CUevent_flags::CU_EVENT_DISABLE_TIMING));
  return event;
}

void CUDADevice::eventCachePush(CUevent event) const {
  std::lock_guard<std::mutex> lock(mu_);
  eventCache_.emplace_back(event);
}

// ===== CUDADeviceMulticastBuffer ===== //
CUDADeviceMulticastBuffer::CUDADeviceMulticastBuffer(
    size_t byteSize, size_t paddedByteSize, const CUDADriver &driver,
    CUmemGenericAllocationHandle multicastHandle, CUdeviceptr multicastPtr,
    std::vector<CUDADeviceMulticastBuffer::DeviceToBuffer> unicastBuffers)
    : DeviceMulticastBuffer(byteSize), driver_(driver),
      multicastHandle_(multicastHandle), multicastPtr_(multicastPtr),
      paddedByteSize_(paddedByteSize),
      unicastBuffers_(std::move(unicastBuffers)) {}

CUDADeviceMulticastBuffer::~CUDADeviceMulticastBuffer() {
  CUDA_CHECK(driver_.MemUnmap(multicastPtr_, paddedByteSize_));
  CUDA_CHECK(driver_.MemAddressFree(multicastPtr_, paddedByteSize_));

  for (const auto &it : unicastBuffers_) {
    CUDA_CHECK(driver_.MulticastUnbind(multicastHandle_, it.device_, 0,
                                       paddedByteSize_));
    CUDA_CHECK(driver_.MemUnmap(it.ptr_, paddedByteSize_));
    CUDA_CHECK(driver_.MemAddressFree(it.ptr_, paddedByteSize_));
    CUDA_CHECK(driver_.MemRelease(it.handle_));
  }
  CUDA_CHECK(driver_.MemRelease(multicastHandle_));
}

ErrorOr<DeviceBufferRef>
CUDADeviceMulticastBuffer::multicastBuffer(const DeviceContext &ctx) const {
  return ctx.createBufferOwning(multicastPtr_, byteSize(),
                                /* owning = */ false);
}

ErrorOr<DeviceBufferRef>
CUDADeviceMulticastBuffer::unicastBufferFor(const DeviceContext &ctx) const {
  CUdevice dev = CUDADevice::cast(ctx.device()).cudaDevice();
  for (const auto &it : unicastBuffers_) {
    if (it.device_ == dev) {
      return ctx.createBufferOwning(it.ptr_, byteSize(), /* owning = */ false);
    }
  }
  RAISE_ERROR("device is not part of this DeviceMulticastBuffer");
}

size_t CUDADeviceMulticastBuffer::numberOfDevices() const {
  return unicastBuffers_.size();
}

ErrorOr<DeviceMulticastBufferRef>
CUDADeviceMulticastBuffer::allocate(const std::vector<const Device *> &devices,
                                    size_t byteSize) {
  if (devices.size() < 2)
    RAISE_ERROR(
        "At least two DeviceContexts needed for device multicast allocation.");
  for (const auto &it : devices) {
    if (it->api() != CUDADevice::api)
      RAISE_ERROR("Only CUDADeviceContext supports device multicast access.");
  }
  const CUDADevice &dev0 = CUDADevice::cast(*devices[0]);
  const CUDADriver &driver = dev0.cudaDriver();

  // Verify that multicast memory is supported.
  int multicastSupported = 0;
  CUDA_CHECK_ERR(driver.DeviceGetAttribute(
      &multicastSupported, CUdevice_attribute::MULTICAST_SUPPORTED,
      dev0.cudaDevice()));
  if (multicastSupported == 0)
    RAISE_ERROR("Multicast buffer not supported.");

  // Get minimum multicast and allocation granularity, adjust requested size up.
  CUmulticastObjectProp multiProp = {};
  multiProp.numDevices = devices.size();
  multiProp.handleTypes =
      //    static_cast<unsigned long long>(CUmemAllocationHandleType::NONE);
      static_cast<unsigned long long>(
          CUmemAllocationHandleType::POSIX_FILE_DESCRIPTOR);
  multiProp.size = byteSize;

  size_t granularity = 0;
  CUDA_CHECK_ERR(driver.MulticastGetGranularity(
      &granularity, &multiProp,
      CUmulticastGranularity_flags::GRANULARITY_MINIMUM));

  size_t paddedByteSize = llvm::alignTo(byteSize, granularity);

  CUmemAllocationProp memProp = {};
  memProp.type = CUmemAllocationType::PINNED;
  memProp.location.type = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
  memProp.requestedHandleTypes =
      CUmemAllocationHandleType::POSIX_FILE_DESCRIPTOR;
  memProp.location.id = dev0.cudaDevice();

  granularity = 0;
  driver.MemGetAllocationGranularity(
      &granularity, &memProp,
      CUmemAllocationGranularity_flags::GRANULARITY_MINIMUM);
  paddedByteSize = llvm::alignTo(paddedByteSize, granularity);
  multiProp.size = paddedByteSize;

  // Allocate the multicast memory handle.
  CUmemGenericAllocationHandle multiHandle;
  CUDA_CHECK_ERR(driver.MulticastCreate(&multiHandle, &multiProp));

  // First add all the devices to the multicast handle to avoid blocking when
  // adding physical memory for each participating device.
  for (const auto &it : devices) {
    const CUDADevice &device = CUDADevice::cast(*it);
    CUDA_CHECK_ERR(driver.MulticastAddDevice(multiHandle, device.cudaDevice()));
  }

  // Only after all devices have been added we can allocate and map virtual
  // address space on the individual devices. Then bind the physical memory to
  // the multicast memory handle.
  std::vector<DeviceToBuffer> unicastBuffers;

  CUmemAccessDesc memDesc = {};
  memDesc.location.type = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
  memDesc.flags = CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  for (const auto &it : devices) {
    const CUDADevice &device = CUDADevice::cast(*it);
    memProp.location.id = device.cudaDevice();
    memDesc.location.id = device.cudaDevice();

    CUmemGenericAllocationHandle memHandle;
    CUdeviceptr memPtr;
    CUDA_CHECK_ERR(driver.MemCreate(&memHandle, paddedByteSize, &memProp, 0));
    CUDA_CHECK_ERR(
        driver.MemAddressReserve(&memPtr, paddedByteSize, 0, nullptr, 0));
    CUDA_CHECK_ERR(driver.MemMap(memPtr, paddedByteSize, 0, memHandle, 0));
    CUDA_CHECK_ERR(driver.MemSetAccess(memPtr, paddedByteSize, &memDesc, 1));
    CUDA_CHECK_ERR(driver.MulticastBindMem(multiHandle, 0, memHandle, 0,
                                           paddedByteSize, 0));
    unicastBuffers.push_back(
        DeviceToBuffer{device.cudaDevice(), memHandle, memPtr});
  }

  // Map the multicast memory handle to a virtual address range and allow access
  // from all participating devices.
  CUdeviceptr multiPtr = nullptr;
  CUDA_CHECK_ERR(
      driver.MemAddressReserve(&multiPtr, paddedByteSize, 0, nullptr, 0));
  CUDA_CHECK_ERR(driver.MemMap(multiPtr, paddedByteSize, 0, multiHandle, 0));
  for (const auto &it : devices) {
    const CUDADevice &device = CUDADevice::cast(*it);
    memDesc.location.id = device.cudaDevice();
    CUDA_CHECK_ERR(driver.MemSetAccess(multiPtr, paddedByteSize, &memDesc, 1));
  }

  return CUDADeviceMulticastBufferRef::create(byteSize, paddedByteSize, driver,
                                              multiHandle, multiPtr,
                                              std::move(unicastBuffers));
}

// ===== CUDADeviceEvent ===== //
CUDADeviceEvent::CUDADeviceEvent(const CUDADevice &device, CUevent event)
    : DeviceEvent(device), event_(event) {}

CUDADeviceEvent::~CUDADeviceEvent() {
  const auto &cudaDevice = CUDADevice::cast(device());
  CUDA_WITH_DEV(cudaDevice);
  cudaDevice.eventCachePush(event_);
}

ErrorOrSuccess CUDADeviceEvent::synchronize() const {
  const auto &cudaDevice = CUDADevice::cast(device());
  CUDA_WITH_DEV_ERR(cudaDevice);
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().EventSynchronize(event_));
  return success();
}

ErrorOr<bool> CUDADeviceEvent::isReady() const {
  const auto &cudaDevice = CUDADevice::cast(device());
  CUDA_WITH_DEV_ERR(cudaDevice);
  CUDADriver::Result res = cudaDevice.cudaDriver().EventQuery(event_);
  if (res == CUDADriver::SUCCESS)
    return true;
  else if (res == CUDADriver::NOT_READY) {
    return false;
  }
  CUDA_CHECK_ERR(res); // All other errors get raised here.
  // Not reachable because there is an error status in res, but compiler
  // complains if not returning a value.
  llvm_unreachable("Not Reachable");
  return Error("Not Reachable.");
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
  DEFINE_OR_RAISE(CUevent, event, cudaDevice.eventCachePop());
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

ErrorOrSuccess CUDADeviceStream::eventRecord(const DeviceEvent &event) const {
  const auto &cudaEvent = CUDADeviceEvent::cast(event);
  const auto &cudaDevice = CUDADevice::cast(device());
  CUDA_WITH_DEV_ERR(cudaDevice);
  CUDA_CHECK_ERR(
      cudaDevice.cudaDriver().EventRecord(cudaEvent.cudaEvent(), stream_));
  return success();
}

ErrorOrSuccess CUDADeviceStream::waitEvent(const DeviceEvent &event,
                                           unsigned int flags) const {
  const auto &cudaEvent = CUDADeviceEvent::cast(event);
  const auto &cudaDevice = CUDADevice::cast(device());
  CUDA_WITH_DEV_ERR(cudaDevice);
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().StreamWaitEvent(
      stream_, cudaEvent.cudaEvent(), flags));
  return success();
}

ErrorOrSuccess CUDADeviceStream::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args) const {
  const auto &cudaFunc = CUDADeviceFunction::cast(func);
  const auto &cudaDevice = CUDADevice::cast(device());
  CUDA_WITH_DEV_ERR(cudaDevice);
  return cudaFunc.cudaCallKernelDirect(
      gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
      reinterpret_cast<CUlaunchAttribute *>(attrs), numAttrs, args, stream_);
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
    : DeviceFunctionCacheEntry(std::move(name)), driver_(driver),
      module_(module), function_(function) {}

CUDAModuleFunction::~CUDAModuleFunction() {
  CUDA_CHECK(cudaDriver().ModuleUnload(module_));
}

const CUDADriver &CUDAModuleFunction::cudaDriver() const { return driver_; }
CUmodule CUDAModuleFunction::cudaModule() const { return module_; }
CUfunction CUDAModuleFunction::cudaFunction() const { return function_; }

const CUDAModuleFunction *
CUDAModuleFunction::cast(const DeviceFunctionCacheEntry *entry) {
  return static_cast<const CUDAModuleFunction *>(entry);
}

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
  MAYBE_SYNC(CUDADeviceContext::cast(context()));
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
  MAYBE_SYNC(CUDADeviceContext::cast(context()));
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
  MAYBE_SYNC(CUDADeviceContext::cast(context()));
  return success();
}

ErrorOr<int32_t> CUDADeviceFunction::getAttribute(int32_t code) const {
  int32_t result;
  CUDA_CHECK_ERR(moduleFunction_->cudaDriver().FuncGetAttribute(
      &result, static_cast<CUfunction_attribute>(code),
      moduleFunction_->cudaFunction()));
  return result;
}

CUmodule CUDADeviceFunction::cudaModule() const {
  return moduleFunction_->cudaModule();
}

const CUDADeviceFunction &CUDADeviceFunction::cast(const DeviceFunction &func) {
  return static_cast<const CUDADeviceFunction &>(func);
}

ErrorOrSuccess CUDADeviceFunction::occupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, int blockSize, size_t dynamicSharedMemSize) const {
  const auto &cudaContext = CUDADeviceContext::cast(context());
  CUDA_WITH_CTX_ERR(cudaContext);
  CUDA_CHECK_ERR(
      moduleFunction_->cudaDriver().OccupancyMaxActiveBlocksPerMultiprocessor(
          numBlocks, moduleFunction_->cudaFunction(), blockSize,
          dynamicSharedMemSize));
  return success();
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
                                     CUcontext context, CUstream stream)
    : DeviceContext(device), driver_(device.cudaDriver()), context_(context),
      stream_(stream) {
  logLoadFunction_ = DeviceContext::configAsBool("log_load_function", false);
  logVerbose_ = DeviceContext::configAsBool("log_verbose", false);
}

CUDADeviceContext::~CUDADeviceContext() {
  // Make sure we have the correct context setup for the cleanup operations.
  // We will exit the CUDAContextScope before we force a push of the context
  // to be destroyed.
  CUDA_WITH_CTX(*this);
  // Synchronize to ensure that all of the chunks have been freed.
  CUDA_CHECK(driver_.StreamSynchronize(stream_));
  // We do not need to synchronize on possible work queued on this stream.
  // Resources will be released automatically once work has been completed.
  CUDA_CHECK(driver_.StreamDestroy(stream_));
}

ErrorOrSuccess CUDADeviceContext::memAlloc(void **ptr, size_t byteSize,
                                           Device::AllocMode mode) const {
  assert(byteSize != 0);
  CUDA_WITH_CTX_ERR(*this);
  switch (mode) {
  case Device::AllocMode::ASYNC: {
    CUDA_CHECK_ERR(driver_.MemAllocAsync(ptr, byteSize, stream_));
    MAYBE_SYNC(*this);
    break;
  }
  case Device::AllocMode::HOST: {
    CUDA_CHECK_ERR(driver_.MemAllocHost(ptr, byteSize));
    break;
  }
  case Device::AllocMode::SYNC: {
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
    MAYBE_SYNC(*this);
  }
  return success();
}

void *CUDADeviceContext::driverContext() const { return cudaContext(); }

ErrorOrSuccess CUDADeviceContext::enqueueCopyToDevice(const DeviceBuffer &dst,
                                                      const void *src) const {
  CHECK_IS_CUDA_ERR(dst.context().device());
  CUDA_WITH_CTX_ERR(*this);

  CUdeviceptr dstPtr = dst.ptr();
  size_t byteSize = dst.byteSize();
  CUDA_CHECK_ERR(driver_.MemcpyHtoDAsync(dstPtr, src, byteSize, stream_));
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess
CUDADeviceContext::enqueueCopyFromDevice(void *dst,
                                         const DeviceBuffer &src) const {
  CHECK_IS_CUDA_ERR(src.context().device());
  CUDA_WITH_CTX_ERR(*this);

  CUdeviceptr srcPtr = src.ptr();
  size_t byteSize = src.byteSize();
  CUDA_CHECK_ERR(driver_.MemcpyDtoHAsync(dst, srcPtr, byteSize, stream_));
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess
CUDADeviceContext::enqueueCopyToDeviceSized(void *dst, void *src,
                                            size_t byteSize) const {
  CUDA_WITH_CTX_ERR(*this);

  CUDA_CHECK_ERR(driver_.MemcpyHtoDAsync(dst, src, byteSize, stream_));
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess
CUDADeviceContext::enqueueCopyFromDeviceSized(void *dst, void *src,
                                              size_t byteSize) const {
  CUDA_WITH_CTX_ERR(*this);

  CUDA_CHECK_ERR(driver_.MemcpyDtoHAsync(dst, src, byteSize, stream_));
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess CUDADeviceContext::memCopyDtoD(const DeviceBuffer &dst,
                                              const DeviceBuffer &src) const {
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
  assert(src.byteSize() >= dstByteSize);
  if (srcCtx.cudaStream() != dstCtx.cudaStream()) {
    // Ensure that we are waiting for the source data to be ready.
    DEFINE_OR_RAISE(DeviceEventRef, srcEvent, srcCtx.enqueueEvent());
    ERR_CHECK_ERR(dstCtx.enqueueWaitForEvent(*srcEvent));

    if (srcCtx.cudaContext() != dstCtx.cudaContext()) {
      CUDA_CHECK_ERR(driver_.MemcpyPeerAsync(dstPtr, dstCtx.context_, srcPtr,
                                             srcCtx.context_, dstByteSize,
                                             stream_));
    } else {
      CUDA_CHECK_ERR(
          driver_.MemcpyDtoDAsync(dstPtr, srcPtr, dstByteSize, stream_));
    }

    // Wait for the copy to complete before proceeding with the source stream
    // (e.g. freeing or overwriting memory).
    DEFINE_OR_RAISE(DeviceEventRef, dstEvent, dstCtx.enqueueEvent());
    ERR_CHECK_ERR(srcCtx.enqueueWaitForEvent(*dstEvent));
  } else {
    CUDA_CHECK_ERR(
        driver_.MemcpyDtoDAsync(dstPtr, srcPtr, dstByteSize, stream_));
  }
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess CUDADeviceContext::enqueueSetMemory(const DeviceBuffer &dst,
                                                   uint64_t val,
                                                   size_t valSize) const {
  CHECK_IS_CUDA_ERR(dst.context().device());
  CUDA_WITH_CTX_ERR(*this);
  void *ptr = dst.ptr();
  size_t byteSize = dst.byteSize();
  if (valSize == 1) {
    CUDA_CHECK_ERR(driver_.MemsetD8Async(ptr, static_cast<uint8_t>(val),
                                         byteSize, stream_));
  } else if (valSize == 2) {
    if ((byteSize & 1) != 0) {
      return Error("byteSize of buffer is misaliged");
    }
    size_t len = byteSize >> 1;
    CUDA_CHECK_ERR(
        driver_.MemsetD16Async(ptr, static_cast<uint16_t>(val), len, stream_));
  } else if (valSize == 4) {
    if ((byteSize & 3) != 0) {
      return Error("byteSize of buffer is misaliged");
    }
    size_t len = byteSize >> 2;
    CUDA_CHECK_ERR(
        driver_.MemsetD32Async(ptr, static_cast<uint32_t>(val), len, stream_));
  } else {
    if (valSize != 8) {
      return Error("unsupported valSize");
    }
    if ((byteSize & 7) != 0) {
      return Error("byteSize of buffer is misaliged");
    }
    uint32_t hiBits = static_cast<uint32_t>(val >> 32);
    uint32_t loBits = static_cast<uint32_t>(val & 0xfffffffful);
    if (hiBits == loBits) {
      size_t len = byteSize >> 2; // Divide by 4 as we are using 32-bit memset.
      CUDA_CHECK_ERR(driver_.MemsetD32Async(ptr, loBits, len, stream_));
    } else {
      size_t len = byteSize >> 3;
      void *hiPtr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ptr) +
                                             sizeof(uint32_t));
      CUDA_CHECK_ERR(driver_.MemsetD2D32Async(ptr, 8, loBits, 1, len, stream_));
      CUDA_CHECK_ERR(
          driver_.MemsetD2D32Async(hiPtr, 8, hiBits, 1, len, stream_));
    }
  }
  MAYBE_SYNC(*this);
  return success();
}

DeviceContext::MMapFileImplResult
CUDADeviceContext::memmapFileImpl(const std::filesystem::path &filename) const {
  return Error("Unimplemented for CUDA devices");
}

static ErrorOr<std::pair<SmallVector<CUjit_option>, SmallVector<uintptr_t>>>
setupCompilationOptions(StringRef debugLevel, int32_t optimizationLevel) {
  static constexpr int kDefaultOptimizationLevel = 4;

  SmallVector<CUjit_option, 2> optionKind;
  SmallVector<uintptr_t, 2> optionValues;

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

SmallVector<std::string> setupCompilationOptionsNVPTXCompiler(
    StringRef debugLevel, int32_t optimizationLevel, StringRef gpuName) {
  // TODO: add more options needed.
  // https://docs.nvidia.com/cuda/ptx-compiler-api/index.html#compilation-options

  static constexpr int kDefaultOptimizationLevel = 4;
  // The default optimization level for NVIDIA GPUs is 4.
  if (optimizationLevel < 0)
    optimizationLevel = kDefaultOptimizationLevel;

  // The Mojo compiler uses optimization level 3 as the highest optimization
  // level, but 4 is the highest for CUDA.
  if (optimizationLevel == 3)
    optimizationLevel = kDefaultOptimizationLevel;

  SmallVector<std::string, 2> options;
  options.push_back("--gpu-name=" + gpuName.str());

  if (debugLevel == "full") {
    options.push_back("-g");
    optimizationLevel = 0;
  } else if (debugLevel == "line-tables") {
    options.push_back("--generate-line-info");
  }

  options.push_back("--opt-level=" + std::to_string(optimizationLevel));

  return options;
}

ErrorOr<M::BufferRef>
CUDADeviceContext::compileFunction(StringRef ptx, StringRef debugLevel,
                                   int32_t optimizationLevel,
                                   StringRef targetAccelerator) const {
#ifdef MODULAR_LINUX
  auto options = setupCompilationOptionsNVPTXCompiler(
      debugLevel, optimizationLevel, targetAccelerator);

  CUDA_WITH_CTX_ERR(*this);

  nvPTXCompilerHandle compiler = nullptr;

  size_t elfSize, infoSize, errorSize;
  unsigned int minorVer, majorVer;

  NVPTXCOMPILER_SAFE_CALL(
      CUDACompilationDriver::NVPTXCompilerGetVersion(&majorVer, &minorVer));

  NVPTXCOMPILER_SAFE_CALL(CUDACompilationDriver::NVPTXCompilerCreate(
      &compiler, (size_t)ptx.size(), (const char *)ptx.data()));

  // FIXME: resize this if needed
  constexpr int maxNumOptions = 16;
  const char *compileOptions[maxNumOptions];
  for (auto [idx, option] : llvm::enumerate(options)) {
    compileOptions[idx] = option.c_str();
  }

  CUDACompilationDriver::Result result =
      CUDACompilationDriver::NVPTXCompilerCompile(
          compiler, options.size(), /* numCompileOptions */
          compileOptions);          /* compileOptions */

  if (result != CUDACompilationDriver::SUCCESS) {
    NVPTXCOMPILER_SAFE_CALL(CUDACompilationDriver::NVPTXCompilerGetErrorLogSize(
        compiler, &errorSize));

    std::string errorLog;
    if (errorSize != 0) {
      errorLog.resize(errorSize + 1);
      CUDACompilationDriver::Result getErrorResult =
          CUDACompilationDriver::NVPTXCompilerGetErrorLog(compiler,
                                                          errorLog.data());

      if (getErrorResult != CUDACompilationDriver::SUCCESS)
        return Error("compile to CUBIN failed and can't get error log\n");
    }

    return Error(errorLog + "compile to CUBIN failed\n");
  }

  NVPTXCOMPILER_SAFE_CALL(
      CUDACompilationDriver::NVPTXCompilerGetCompiledProgramSize(compiler,
                                                                 &elfSize));

  auto resultBuf = M::WriteableBuffer::get(elfSize);

  NVPTXCOMPILER_SAFE_CALL(
      CUDACompilationDriver::NVPTXCompilerGetCompiledProgram(
          compiler, (void *)resultBuf->getBufferStart()));

  NVPTXCOMPILER_SAFE_CALL(
      CUDACompilationDriver::NVPTXCompilerGetInfoLogSize(compiler, &infoSize));

  std::optional<std::string> printWarning =
      llvm::sys::Process::GetEnv("MODULAR_NVPTX_COMPILER_WARN");
  std::optional<std::string> werror =
      llvm::sys::Process::GetEnv("MODULAR_NVPTX_COMPILER_WERROR");

  if (infoSize != 0 && (printWarning || werror)) {
    std::string infoLog;
    infoLog.resize(infoSize + 1);
    CUDACompilationDriver::Result getLogResult =
        CUDACompilationDriver::NVPTXCompilerGetInfoLog(compiler,
                                                       infoLog.data());

    if (getLogResult != CUDACompilationDriver::SUCCESS)
      return Error("compile to CUBIN has log info but can't get it\n");

    if (werror)
      return Error(infoLog + "compile to CUBIN failed due to warnings\n");
    else
      llvm::errs() << infoLog << "\n";
  }

  NVPTXCOMPILER_SAFE_CALL(
      CUDACompilationDriver::NVPTXCompilerDestroy(&compiler));

  return resultBuf;
#else  // MODULAR_LINUX
  llvm_unreachable("NVPTX compiler not supported on this platform");
#endif // MODULAR_LINUX
}

ErrorOr<DeviceFunctionRef> CUDADeviceContext::loadFunction(
    StringRef moduleName, StringRef functionName,
    const DeviceFunction::SignatureBuilder &signature, StringRef code,
    int32_t maxDynamicSharedBytes, StringRef debugLevel,
    int32_t optimizationLevel) const {
  DeviceFunctionCacheEntryRef entry =
      device().findInFunctionCache(moduleName.empty() ? code : moduleName);
  if (entry.getPointer() != nullptr) {
    CUDAModuleFunctionRef moduleFunc =
        CUDAModuleFunctionRef::take(CUDAModuleFunction::cast(entry.release()));
    return CUDADeviceFunctionRef::create(*this, signature,
                                         std::move(moduleFunc));
  }

  if (logLoadFunction_) {
    llvm::errs() << "CUDADeviceContext::loadFunction(" << functionName
                 << (logVerbose_ ? "):\n" : ")") << (logVerbose_ ? code : "")
                 << "\n";
  }
  auto errorOrOpts = setupCompilationOptions(debugLevel, optimizationLevel);
  if (errorOrOpts)
    return errorOrOpts.takeError();
  auto [optionKind, optionValues] = errorOrOpts.takeValue();

  CUmodule module;
  CUDA_WITH_CTX_ERR(*this);

  CUDA_CHECK_ERR(driver_.ModuleLoadDataEx(
      &module, code.data(), optionKind.size(), optionKind.data(),
      reinterpret_cast<void **>(optionValues.data())));

  CUfunction function;
  CUDA_CHECK_ERR_MSG(
      driver_.ModuleGetFunction(&function, module, functionName.data()),
      "function name: " + functionName.str());

  CUDA_CHECK_ERR(driver_.FuncSetAttribute(
      function,
      CUfunction_attribute::FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
      1));

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
  entry = device().insertInFunctionCache(moduleName.empty() ? code : moduleName,
                                         *moduleFunc);
  result = CUDADeviceFunctionRef::create(
      *this, signature,
      CUDAModuleFunctionRef::take(CUDAModuleFunction::cast(entry.release())));
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
    void *attrs, uint32_t numAttrs, void **args, const size_t *argSizes) const {
  const auto &cudaFunc = CUDADeviceFunction::cast(func);

  CUDA_WITH_CTX_ERR(*this);
  return cudaFunc.cudaCallKernelDirect(
      gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
      reinterpret_cast<CUlaunchAttribute *>(attrs), numAttrs, args, stream_);
}

ErrorOr<DeviceStreamRef> CUDADeviceContext::stream() const {
  return CUDADeviceStreamRef::create(CUDADevice::cast(this->device()), stream_);
}

ErrorOr<DeviceStreamRef>
CUDADeviceContext::streamCreate(unsigned int flags) const {
  CUstream stream;
  CUDA_CHECK_ERR(this->cudaDriver().StreamCreate(&stream, flags));

  return CUDADeviceStreamRef::create(CUDADevice::cast(this->device()), stream);
}

ErrorOr<DeviceStreamRef>
CUDADeviceContext::streamCreateWithPriority(unsigned int flags,
                                            int priority) const {
  CUstream stream;
  CUDA_CHECK_ERR(
      this->cudaDriver().StreamCreateWithPriority(&stream, flags, priority));

  return CUDADeviceStreamRef::create(CUDADevice::cast(this->device()), stream);
}

ErrorOrSuccess
CUDADeviceContext::streamPriorityRange(int *leastPriority,
                                       int *greatestPriority) const {
  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(
      driver_.CtxGetStreamPriorityRange(leastPriority, greatestPriority));
  return success();
}

ErrorOrSuccess CUDADeviceContext::synchronize() const {
  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.StreamSynchronize(stream_));
  // Handle any pending events after synchronization.
  ERR_CHECK_ERR(checkPendingWork());
  return success();
}

ErrorOr<DeviceEventRef> CUDADeviceContext::enqueueEvent() const {
  CUDA_WITH_CTX_ERR(*this);
  DEFINE_OR_RAISE(CUevent, event, CUDADevice::cast(device()).eventCachePop());
  CUDA_CHECK_ERR(driver_.EventRecord(event, stream_));
  return CUDADeviceEventRef::create(CUDADevice::cast(this->device()), event);
}

ErrorOr<DeviceEventRef>
CUDADeviceContext::eventCreate(unsigned int flags) const {
  CUDA_WITH_CTX_ERR(*this);
  CUevent event;
  CUDA_CHECK_ERR(
      driver_.EventCreate(&event, static_cast<CUevent_flags>(flags)));
  return CUDADeviceEventRef::create(CUDADevice::cast(this->device()), event);
}

ErrorOrSuccess
CUDADeviceContext::enqueueWaitFor(const DeviceEvent &event) const {
  const auto &cudaEvent = CUDADeviceEvent::cast(event);
  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.StreamWaitEvent(stream_, cudaEvent.cudaEvent(), 0));
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess
CUDADeviceContext::enqueueWaitFor(const DeviceContext &ctx) const {
  DEFINE_OR_RAISE(DeviceEventRef, event, ctx.enqueueEvent());
  ERR_CHECK_ERR(enqueueWaitFor(*event));
  return success();
}

ErrorOr<DeviceTimerRef> CUDADeviceContext::startTimer() const {
  CUevent start;
  CUevent stop;

  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.EventCreate(&start));
  CUDA_CHECK_ERR(driver_.EventCreate(&stop));

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

ErrorOr<Device::MemInfo> CUDADeviceContext::getDeviceMemoryInfo() const {
  CUDA_WITH_CTX(*this);
  // To get more accurate memory stats, ensure that at least all pending memory
  // allocations and frees have completed for this context's stream.
  CUDA_CHECK_ERR(driver_.StreamSynchronize(stream_));

  return device().getMemoryInfo();
}

ErrorOrSuccess CUDADeviceContext::occupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, const DeviceFunction &func, int blockSize,
    size_t dynamicSharedMemSize) const {
  const auto &cudaFunc = CUDADeviceFunction::cast(func);
  CUDA_WITH_CTX_ERR(*this);
  return cudaFunc.occupancyMaxActiveBlocksPerMultiprocessor(
      numBlocks, blockSize, dynamicSharedMemSize);
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
