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
#include "CUDACompilationDevice.h"
#include "CUDACompilationDriver.h"
#include "CUDADriver.h"
#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "Support/Profiling/Tracy.h"

using namespace M;
using namespace M::Driver;

namespace {

// The helper class CUDAContextScope is used to set and restore the CUcontext
// around interactions with the CUDA driver.
class CUDAContextScope {
public:
  CUDAContextScope(const CUcontext ctx, const CUDADriver &driver)
      : driver_(driver), ctx_(ctx), prevCtx_(nullptr),
        status_(CUDADriver::SUCCESS) {
    driver_.cuCtxGetCurrent(&prevCtx_);
    if (prevCtx_ == ctx_)
      return;
    status_ = driver_.cuCtxSetCurrent(ctx_);
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
    driver_.cuCtxGetCurrent(&ctx);
    driver_.cuCtxSetCurrent(prevCtx_);
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
          cuda_ctx.cudaDriver().cuStreamSynchronize(cuda_ctx.cudaStream()));   \
    }                                                                          \
  } while (0)

[[maybe_unused]]
static Error formatCUDAError(const char *file, int line, const char *func,
                             CUDADriver::Result result,
                             std::string extra_msg = "") {
  ErrorOr<const CUDADriver &> errDriver = CUDADriver::get();
  const char *errName = nullptr;
  const char *errStr = nullptr;

  if (!errDriver) {
    // Ignore errors and use nullptr of char* instead.
    errDriver->cuGetErrorName(result, &errName);
    errDriver->cuGetErrorString(result, &errStr);
  }

  if (!extra_msg.empty())
    extra_msg = std::string(" - ") + extra_msg;

  if (errName && errStr) {
    return DeviceContext::formatError(file, line, func,
                                      llvm::Twine("CUDA call failed: ") +
                                          errName + " (" + errStr + ")" +
                                          extra_msg);
  }

  if (errName || errStr) {
    return DeviceContext::formatError(
        file, line, func,
        llvm::Twine("CUDA call failed: ") +
            (errName != nullptr ? errName : errStr) + extra_msg);
  }

  return DeviceContext::formatError(
      file, line, func, "CUDA call failed: " + llvm::Twine(result) + extra_msg);
}

[[maybe_unused]]
static Error formatCUDAError(const char *file, int line, const char *func,
                             CUDACompilationDriver::Result result,
                             std::string extra_msg = "") {
  if (!extra_msg.empty())
    extra_msg = std::string(" - ") + extra_msg;

  return DeviceContext::formatError(
      file, line, func,
      "NVPTX Compiler call failed: " + llvm::Twine(result) + extra_msg);
}

static int cudaCCMonolithic(int major, int minor) { return major * 10 + minor; }

static std::string cudaCCToSM(int major, int minor) {
  std::string sm = "sm_";
  int cc = cudaCCMonolithic(major, minor);
  sm += std::to_string(cc);

  // For architectures above 90 we use the "a" variants to enable hardware
  // specific features at the cost of backward and forward compatibility.
  if (cc >= 90)
    sm += "a";

  return sm;
}

// ===== CUDADevice ===== //
CUDADevice::CUDADevice(const CUDADriver &driver, int id, CUdevice device,
                       CUcontext context, std::string name,
                       int computeCapabilityMajor, int computeCapabilityMinor,
                       Device::MemInfo memInfo, size_t allocAlignment,
                       int64_t deviceMemoryLimit)
    : Device(api, std::move(name), id, memInfo, allocAlignment,
             deviceMemoryLimit),
      CUDACompilationDevice(
          cudaCCToSM(computeCapabilityMajor, computeCapabilityMinor)),
      driver_(driver), device_(device), context_(context),
      computeCapability_(
          cudaCCMonolithic(computeCapabilityMajor, computeCapabilityMinor)) {}

CUDADevice::~CUDADevice() {
  // Destroy the function map before destroying the context.
  clearFunctionCache();
  memoryManager().clear();
  if (context_ != nullptr)
    // We ignore the result here as there is not much we can do if it fails
    // since in most cases the CUDA driver is in the process of shutting down.
    cudaDriver().cuDevicePrimaryCtxRelease_v2(cudaDevice());
}

const CUDADriver &CUDADevice::cudaDriver() const { return driver_; }

CUdevice CUDADevice::cudaDevice() const { return device_; }

ErrorOr<CUcontext> CUDADevice::cudaContext() const { return context_; }

const CUDACompilationDevice *CUDADevice::compilationDevice() const {
  return this;
}

ErrorOr<int> CUDADevice::getApiVersion() const {
  // CUDADevice always has a compilation device - delegate to it
  return CUDACompilationDevice::getApiVersion();
}

ErrorOr<DriverVersion &> CUDADevice::getDriverVersion() const {
  // CUDADevice always has a compilation device - delegate to it
  return CUDACompilationDevice::getDriverVersion();
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

  if (computeCapability_ < MIN_COMPUTE_CAPABILITY) {
    RAISE_ERROR("MAX only supports CUDA Pascal architectures or higher");
  }

  int driverVersion = 0;
  CUDA_CHECK_ERR(driver_.cuDriverGetVersion(&driverVersion));
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
  return computeCapability_;
}

ErrorOr<bool> CUDADevice::canAccess(const Device &peer) const {
  if (id() == peer.id()) {
    // Always disable self access (same behaviour as cuDeviceCanAccessPeer).
    return false;
  }

  int canAccessPeer = 0;
  CUDA_CHECK_ERR(driver_.cuDeviceCanAccessPeer(
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
  CUDA_CHECK_ERR(driver_.cuDeviceGetP2PAttribute(
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
  CUDADriver::Result res =
      driver_.cuCtxEnablePeerAccess(*peerCtxOr, /*flags=*/0);
  if (res != CUDADriver::SUCCESS) {
    const char *name = nullptr;
    driver_.cuGetErrorName(res, &name);
    if (!(name && (Twine(name) == "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED")))
      return formatCUDAError(__FILE__, __LINE__, __FUNCTION__, res);
  }

  // Also allow access to the peer's stream-ordered allocation pool.
  CUmemoryPool pool;
  CUDA_CHECK_ERR(driver_.cuDeviceGetMemPool(&pool, peerCudaDev.cudaDevice()));

  CUmemAccessDesc desc{};
  desc.location.type = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = id();
  desc.flags = CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUDA_CHECK_ERR(driver_.cuMemPoolSetAccess(pool, &desc, /*count=*/1));

  return success();
}

ErrorOr<int> CUDADevice::getAttribute(Attribute attr) const {
  int value = 0;
  CUDA_CHECK_ERR(driver_.cuDeviceGetAttribute(
      &value, static_cast<CUdevice_attribute>(attr), cudaDevice()));
  return value;
}

ErrorOr<bool> CUDADevice::supportsMulticast() const {
  // Check if multicast memory is supported.
  int multicastSupported = 0;
  CUDA_CHECK_ERR(cudaDriver().cuDeviceGetAttribute(
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
  CUDA_CHECK_ERR(this->cudaDriver().cuStreamCreate(&stream, 0));

  return CUDADeviceContextRef::create(*this, *context, stream);
}

ErrorOr<int> CUDADevice::numberOfGPUs() {
  static ErrorOr<unsigned int> errOrCount = []() -> ErrorOr<int> {
    ErrorOr<const NVML &> errDriver = NVML::get();
    if (errDriver.isError())
      return errDriver.takeError();

    unsigned int result;
    if (errDriver->DeviceGetCount_v2(&result) != NVML::SUCCESS)
      RAISE_ERROR("NVML: Unable to get device count");
    return result;
  }();

  if (errOrCount.isError()) {
    return errOrCount.copy();
  }
  return static_cast<int>(*errOrCount);
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
    CUDA_CHECK_ERR(driver_.cuMemAllocHost_v2(ptr, byteSize));
  } else {
    CUDA_CHECK_ERR(driver_.cuMemAlloc_v2(ptr, byteSize));
  }
  return success();
}

ErrorOrSuccess CUDADevice::chunkFree(void *ptr, bool isHost) const {
  CUDA_WITH_DEV_ERR(*this);
  if (isHost) {
    CUDA_CHECK_ERR(driver_.cuMemFreeHost(ptr));
  } else {
    CUDA_CHECK_ERR(driver_.cuMemFree_v2(ptr));
  }
  return success();
}

ErrorOr<Device::MemInfo> CUDADevice::getMemoryInfo() const {
  Device::MemInfo memInfo;
  CUDA_WITH_DEV_ERR(*this);
  CUDA_CHECK_ERR(driver_.cuMemGetInfo_v2(&memInfo.free, &memInfo.total));
  return memInfo;
}

ErrorOr<void *> CUDADevice::getCurrentDriverContext() const {
  CUcontext result = nullptr;
  CUDA_CHECK_ERR(driver_.cuCtxGetCurrent(&result));
  return result;
}
ErrorOrSuccess CUDADevice::setCurrentDriverContext(void *ctx) const {
  CUDA_CHECK_ERR(driver_.cuCtxSetCurrent(reinterpret_cast<CUcontext>(ctx)));
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

    CUDA_CHECK_ERR(driver.cuDeviceGet(&cuDevice, id));
    CUDA_CHECK_ERR(driver.cuDeviceGetName(name, sizeof(name), cuDevice));
    CUDA_CHECK_ERR(driver.cuDeviceGetAttribute(
        &alignment, CUdevice_attribute::TEXTURE_ALIGNMENT, cuDevice));

    // Use the primary context for this device, with the effect that we share
    // the same context with other users of the CUDA Runtime API.
    CUDA_CHECK_ERR(driver.cuDevicePrimaryCtxRetain(&cuContext, cuDevice));

    CUDAContextScope withCtx(cuContext, driver);
    Device::MemInfo memInfo;
    CUDA_CHECK_ERR(driver.cuMemGetInfo_v2(&memInfo.free, &memInfo.total));

    // Get compute capability to create compilation device
    int computeCapabilityMajor, computeCapabilityMinor;
    CUDA_CHECK_ERR(driver.cuDeviceGetAttribute(
        &computeCapabilityMajor, CUdevice_attribute::COMPUTE_CAPABILITY_MAJOR,
        cuDevice));
    CUDA_CHECK_ERR(driver.cuDeviceGetAttribute(
        &computeCapabilityMinor, CUdevice_attribute::COMPUTE_CAPABILITY_MINOR,
        cuDevice));

    device = new CUDADevice(driver, id, cuDevice, cuContext, name,
                            computeCapabilityMajor, computeCapabilityMinor,
                            memInfo, alignment, deviceMemoryLimit);

    // Once the Device has been created remember it in the devices list.
    (*devices)[id] = device;
  }
  return *device;
}

ErrorOr<const char *> CUDADevice::getArchName() const {
  return CUDACompilationDevice::getArchName();
}

void CUDADevice::eventCacheClear() const {
  std::lock_guard<std::mutex> lock(mu_);
  for (const auto &event : eventCache_) {
    // Ignore errors during destruction of device object.
    cudaDriver().cuEventDestroy(event);
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
  CUDA_CHECK_ERR(cudaDriver().cuEventCreate(
      &event,
      static_cast<unsigned int>(CUevent_flags::CU_EVENT_DISABLE_TIMING)));
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
  CUDA_CHECK(driver_.cuMemUnmap(multicastPtr_, paddedByteSize_));
  CUDA_CHECK(driver_.cuMemAddressFree(multicastPtr_, paddedByteSize_));

  for (const auto &it : unicastBuffers_) {
    CUDA_CHECK(driver_.cuMulticastUnbind(multicastHandle_, it.device_, 0,
                                         paddedByteSize_));
    CUDA_CHECK(driver_.cuMemUnmap(it.ptr_, paddedByteSize_));
    CUDA_CHECK(driver_.cuMemAddressFree(it.ptr_, paddedByteSize_));
    CUDA_CHECK(driver_.cuMemRelease(it.handle_));
  }
  CUDA_CHECK(driver_.cuMemRelease(multicastHandle_));
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
  CUDA_CHECK_ERR(driver.cuDeviceGetAttribute(
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
  CUDA_CHECK_ERR(driver.cuMulticastGetGranularity(
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
  driver.cuMemGetAllocationGranularity(
      &granularity, &memProp,
      CUmemAllocationGranularity_flags::GRANULARITY_MINIMUM);
  paddedByteSize = llvm::alignTo(paddedByteSize, granularity);
  multiProp.size = paddedByteSize;

  // Allocate the multicast memory handle.
  CUmemGenericAllocationHandle multiHandle;
  CUDA_CHECK_ERR(driver.cuMulticastCreate(&multiHandle, &multiProp));

  // First add all the devices to the multicast handle to avoid blocking when
  // adding physical memory for each participating device.
  for (const auto &it : devices) {
    const CUDADevice &device = CUDADevice::cast(*it);
    CUDA_CHECK_ERR(
        driver.cuMulticastAddDevice(multiHandle, device.cudaDevice()));
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
    CUDA_CHECK_ERR(driver.cuMemCreate(&memHandle, paddedByteSize, &memProp, 0));
    CUDA_CHECK_ERR(
        driver.cuMemAddressReserve(&memPtr, paddedByteSize, 0, nullptr, 0));
    CUDA_CHECK_ERR(driver.cuMemMap(memPtr, paddedByteSize, 0, memHandle, 0));
    CUDA_CHECK_ERR(driver.cuMemSetAccess(memPtr, paddedByteSize, &memDesc, 1));
    CUDA_CHECK_ERR(driver.cuMulticastBindMem(multiHandle, 0, memHandle, 0,
                                             paddedByteSize, 0));
    unicastBuffers.push_back(
        DeviceToBuffer{device.cudaDevice(), memHandle, memPtr});
  }

  // Map the multicast memory handle to a virtual address range and allow access
  // from all participating devices.
  CUdeviceptr multiPtr = nullptr;
  CUDA_CHECK_ERR(
      driver.cuMemAddressReserve(&multiPtr, paddedByteSize, 0, nullptr, 0));
  CUDA_CHECK_ERR(driver.cuMemMap(multiPtr, paddedByteSize, 0, multiHandle, 0));
  for (const auto &it : devices) {
    const CUDADevice &device = CUDADevice::cast(*it);
    memDesc.location.id = device.cudaDevice();
    CUDA_CHECK_ERR(
        driver.cuMemSetAccess(multiPtr, paddedByteSize, &memDesc, 1));
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
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().cuEventSynchronize(event_));
  return success();
}

ErrorOr<bool> CUDADeviceEvent::isReady() const {
  const auto &cudaDevice = CUDADevice::cast(device());
  CUDA_WITH_DEV_ERR(cudaDevice);
  CUDADriver::Result res = cudaDevice.cudaDriver().cuEventQuery(event_);
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
CUDADeviceStream::CUDADeviceStream(const CUDADeviceContext &ctx,
                                   CUstream stream, bool isOwner)
    : DeviceStream(ctx), stream_(stream), isOwner_(isOwner) {}

CUDADeviceStream::~CUDADeviceStream() {
  if (!isOwner_) {
    return;
  }
  CUDA_WITH_CTX(CUDADeviceContext::cast(context_()));
  const auto &cudaDevice = CUDADevice::cast(this->device());
  CUDA_CHECK(cudaDevice.cudaDriver().cuStreamDestroy(stream_));
  isOwner_ = false;
}

ErrorOrSuccess CUDADeviceStream::synchronize() const {
  const auto &cudaDevice = CUDADevice::cast(this->device());

  CUDA_WITH_DEV_ERR(cudaDevice);
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().cuStreamSynchronize(stream_));
  return success();
}

ErrorOr<DeviceEventRef> CUDADeviceStream::enqueueEvent() const {
  const auto &cudaDevice = CUDADevice::cast(this->device());
  DEFINE_OR_RAISE(CUevent, event, cudaDevice.eventCachePop());
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().cuEventRecord(event, stream_));
  return CUDADeviceEventRef::create(cudaDevice, event);
}

ErrorOrSuccess
CUDADeviceStream::enqueueWaitForEvent(const DeviceEvent &event) const {
  const auto &cudaEvent = CUDADeviceEvent::cast(event);
  const auto &cudaDevice = CUDADevice::cast(this->device());
  CUDA_WITH_DEV_ERR(cudaDevice);
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().cuStreamWaitEvent(
      stream_, cudaEvent.cudaEvent(), 0));
  return success();
}

ErrorOrSuccess CUDADeviceStream::eventRecord(const DeviceEvent &event) const {
  const auto &cudaEvent = CUDADeviceEvent::cast(event);
  const auto &cudaDevice = CUDADevice::cast(this->device());
  CUDA_WITH_DEV_ERR(cudaDevice);
  CUDA_CHECK_ERR(
      cudaDevice.cudaDriver().cuEventRecord(cudaEvent.cudaEvent(), stream_));
  return success();
}

ErrorOrSuccess CUDADeviceStream::waitEvent(const DeviceEvent &event,
                                           unsigned int flags) const {
  const auto &cudaEvent = CUDADeviceEvent::cast(event);
  const auto &cudaDevice = CUDADevice::cast(this->device());
  CUDA_WITH_DEV_ERR(cudaDevice);
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().cuStreamWaitEvent(
      stream_, cudaEvent.cudaEvent(), flags));
  return success();
}

ErrorOrSuccess CUDADeviceStream::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args) const {
  const auto &cudaFunc = CUDADeviceFunction::cast(func);
  const auto &cudaDevice = CUDADevice::cast(this->device());
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

// ===== CUDAModuleFunction ===== //
CUDAModuleFunction::CUDAModuleFunction(const std::string name,
                                       const CUDADriver &driver,
                                       CUmodule module, CUfunction function)
    : DeviceFunctionCacheEntry(std::move(name)), driver_(driver),
      module_(module), function_(function) {}

CUDAModuleFunction::~CUDAModuleFunction() {
  CUDA_CHECK(cudaDriver().cuModuleUnload(module_));
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

  CUDA_CHECK_ERR(moduleFunction_->cudaDriver().cuLaunchKernelEx(
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
  CUDA_CHECK_ERR(cudaContext.cudaDriver().cuModuleGetGlobal_v2(
      &devicePtr, &deviceSize, moduleFunction_->cudaModule(), name));

  if (deviceSize < byteSize) {
    RAISE_ERROR("byteSize too big");
  }

  CUDA_CHECK_ERR(cudaContext.cudaDriver().cuMemcpyHtoDAsync_v2(
      devicePtr, data, byteSize, cudaContext.cudaStream()));
  MAYBE_SYNC(CUDADeviceContext::cast(context()));
  return success();
}

ErrorOr<int32_t> CUDADeviceFunction::getAttribute(int32_t code) const {
  int32_t result;
  CUDA_CHECK_ERR(moduleFunction_->cudaDriver().cuFuncGetAttribute(
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
      moduleFunction_->cudaDriver().cuOccupancyMaxActiveBlocksPerMultiprocessor(
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

  CUDA_CHECK(cudaContext.cudaDriver().cuEventDestroy(start_));
  CUDA_CHECK(cudaContext.cudaDriver().cuEventDestroy(stop_));
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
  CUDA_CHECK(driver_.cuStreamSynchronize(stream_));
  CUDA_CHECK(driver_.cuStreamDestroy(stream_));
}

ErrorOrSuccess CUDADeviceContext::memAlloc(void **ptr, size_t byteSize,
                                           Device::AllocMode mode) const {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::memAlloc", TRACY_COLOR_PURPLE);

  assert(byteSize != 0);
  CUDA_WITH_CTX_ERR(*this);
  switch (mode) {
  case Device::AllocMode::ASYNC: {
    CUDA_CHECK_ERR(driver_.cuMemAllocAsync(ptr, byteSize, stream_));
    MAYBE_SYNC(*this);
    break;
  }
  case Device::AllocMode::HOST: {
    CUDA_CHECK_ERR(driver_.cuMemAllocHost_v2(ptr, byteSize));
    break;
  }
  case Device::AllocMode::SYNC: {
    CUDA_CHECK_ERR(driver_.cuMemAlloc_v2(ptr, byteSize));
    break;
  }
  }
  return success();
}

ErrorOrSuccess CUDADeviceContext::memFree(void *ptr, bool isHost) const {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::memFree", TRACY_COLOR_PURPLE);

  CUDA_WITH_CTX(*this);
  if (isHost) {
    CUDA_CHECK_ERR(driver_.cuMemFreeHost(ptr));
  } else {
    CUDA_CHECK_ERR(driver_.cuMemFreeAsync(ptr, stream_));
    MAYBE_SYNC(*this);
  }
  return success();
}

void *CUDADeviceContext::driverContext() const { return cudaContext(); }

ErrorOrSuccess CUDADeviceContext::enqueueCopyToDevice(const DeviceBuffer &dst,
                                                      const void *src) const {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::enqueueCopyToDevice",
                       TRACY_COLOR_PURPLE);

  CHECK_IS_CUDA_ERR(dst.context().device());
  CUDA_WITH_CTX_ERR(*this);

  CUdeviceptr dstPtr = dst.ptr();
  size_t byteSize = dst.byteSize();
  CUDA_CHECK_ERR(driver_.cuMemcpyHtoDAsync_v2(dstPtr, src, byteSize, stream_));
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess
CUDADeviceContext::enqueueCopyFromDevice(void *dst,
                                         const DeviceBuffer &src) const {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::enqueueCopyFromDevice",
                       TRACY_COLOR_PURPLE);

  CHECK_IS_CUDA_ERR(src.context().device());
  CUDA_WITH_CTX_ERR(*this);

  CUdeviceptr srcPtr = src.ptr();
  size_t byteSize = src.byteSize();
  CUDA_CHECK_ERR(driver_.cuMemcpyDtoHAsync_v2(dst, srcPtr, byteSize, stream_));
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess
CUDADeviceContext::enqueueCopyToDeviceSized(void *dst, void *src,
                                            size_t byteSize) const {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::enqueueCopyToDeviceSized",
                       TRACY_COLOR_PURPLE);

  CUDA_WITH_CTX_ERR(*this);

  CUDA_CHECK_ERR(driver_.cuMemcpyHtoDAsync_v2(dst, src, byteSize, stream_));
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess
CUDADeviceContext::enqueueCopyFromDeviceSized(void *dst, void *src,
                                              size_t byteSize) const {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::enqueueCopyFromDeviceSized",
                       TRACY_COLOR_PURPLE);

  CUDA_WITH_CTX_ERR(*this);

  CUDA_CHECK_ERR(driver_.cuMemcpyDtoHAsync_v2(dst, src, byteSize, stream_));
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess CUDADeviceContext::memCopyDtoD(const DeviceBuffer &dst,
                                              const DeviceBuffer &src) const {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::memCopyDtoD", TRACY_COLOR_PURPLE);

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
      CUDA_CHECK_ERR(driver_.cuMemcpyPeerAsync(dstPtr, dstCtx.context_, srcPtr,
                                               srcCtx.context_, dstByteSize,
                                               stream_));
    } else {
      CUDA_CHECK_ERR(
          driver_.cuMemcpyDtoDAsync_v2(dstPtr, srcPtr, dstByteSize, stream_));
    }

    // Wait for the copy to complete before proceeding with the source stream
    // (e.g. freeing or overwriting memory).
    DEFINE_OR_RAISE(DeviceEventRef, dstEvent, dstCtx.enqueueEvent());
    ERR_CHECK_ERR(srcCtx.enqueueWaitForEvent(*dstEvent));
  } else {
    CUDA_CHECK_ERR(
        driver_.cuMemcpyDtoDAsync_v2(dstPtr, srcPtr, dstByteSize, stream_));
  }
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess CUDADeviceContext::enqueueSetMemory(const DeviceBuffer &dst,
                                                   uint64_t val,
                                                   size_t valSize) const {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::enqueueSetMemory",
                       TRACY_COLOR_PURPLE);

  CHECK_IS_CUDA_ERR(dst.context().device());
  CUDA_WITH_CTX_ERR(*this);
  void *ptr = dst.ptr();
  size_t byteSize = dst.byteSize();
  if (valSize == 1) {
    CUDA_CHECK_ERR(driver_.cuMemsetD8Async(ptr, static_cast<uint8_t>(val),
                                           byteSize, stream_));
  } else if (valSize == 2) {
    if ((byteSize & 1) != 0) {
      return Error("byteSize of buffer is misaliged");
    }
    size_t len = byteSize >> 1;
    CUDA_CHECK_ERR(driver_.cuMemsetD16Async(ptr, static_cast<uint16_t>(val),
                                            len, stream_));
  } else if (valSize == 4) {
    if ((byteSize & 3) != 0) {
      return Error("byteSize of buffer is misaliged");
    }
    size_t len = byteSize >> 2;
    CUDA_CHECK_ERR(driver_.cuMemsetD32Async(ptr, static_cast<uint32_t>(val),
                                            len, stream_));
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
      CUDA_CHECK_ERR(driver_.cuMemsetD32Async(ptr, loBits, len, stream_));
    } else {
      size_t len = byteSize >> 3;
      void *hiPtr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ptr) +
                                             sizeof(uint32_t));
      CUDA_CHECK_ERR(
          driver_.cuMemsetD2D32Async(ptr, 8, loBits, 1, len, stream_));
      CUDA_CHECK_ERR(
          driver_.cuMemsetD2D32Async(hiPtr, 8, hiBits, 1, len, stream_));
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

ErrorOr<DeviceFunctionRef> CUDADeviceContext::loadFunction(
    StringRef moduleName, StringRef functionName,
    const DeviceFunction::SignatureBuilder &signature, StringRef code,
    int32_t maxDynamicSharedBytes, StringRef debugLevel,
    int32_t optimizationLevel) const {
  DeviceFunctionCacheEntryRef entry =
      device().findInFunctionCache(Device::FunctionCacheKey(
          moduleName.empty() ? code.str() : moduleName.str(), debugLevel.str(),
          optimizationLevel, maxDynamicSharedBytes));
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

  CUDA_CHECK_ERR(driver_.cuModuleLoadDataEx(
      &module, code.data(), optionKind.size(), optionKind.data(),
      reinterpret_cast<void **>(optionValues.data())));

  CUfunction function;
  CUDA_CHECK_ERR_MSG(
      driver_.cuModuleGetFunction(&function, module, functionName.data()),
      "function name: " + functionName.str());

  CUDA_CHECK_ERR(driver_.cuFuncSetAttribute(
      function,
      CUfunction_attribute::FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
      1));

  if (maxDynamicSharedBytes >= 0) {
    CUDA_CHECK_ERR(driver_.cuFuncSetAttribute(
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
  entry = device().insertInFunctionCache(
      Device::FunctionCacheKey(
          moduleName.empty() ? code.str() : moduleName.str(), debugLevel.str(),
          optimizationLevel, maxDynamicSharedBytes),
      *moduleFunc);
  result = CUDADeviceFunctionRef::create(
      *this, signature,
      CUDAModuleFunctionRef::take(CUDAModuleFunction::cast(entry.release())));
  return result;
}

ErrorOrSuccess CUDADeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args, const size_t *argSizes) const {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::enqueueFunctionExecDirect",
                       TRACY_COLOR_PURPLE);

  const auto &cudaFunc = CUDADeviceFunction::cast(func);

  CUDA_WITH_CTX_ERR(*this);
  return cudaFunc.cudaCallKernelDirect(
      gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
      reinterpret_cast<CUlaunchAttribute *>(attrs), numAttrs, args, stream_);
}

ErrorOr<DeviceStreamRef> CUDADeviceContext::stream() const {
  // Do not destroy the wrapped stream_ in DTOR as it is owned by this Context.
  return this->createExternalStream(reinterpret_cast<void *>(stream_));
}

ErrorOrSuccess
CUDADeviceContext::streamPriorityRange(int *leastPriority,
                                       int *greatestPriority) const {
  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(
      driver_.cuCtxGetStreamPriorityRange(leastPriority, greatestPriority));
  return success();
}

ErrorOr<DeviceStreamRef> CUDADeviceContext::createStream(int priority) const {
  // Default priority is 0 (lowest); compatible with CUDA specs.

  // CUDA blocking streams are a legacy feature; interpretation of "blocking"
  // is w.r.t. the default stream and does not mean "synchronous".
  // We want our streams to be independent of any other stream.
  unsigned int constexpr kNonBlocking = 0x1;

  CUstream stream;
  CUDA_CHECK_ERR(this->cudaDriver().cuStreamCreateWithPriority(
      &stream, kNonBlocking, priority));

  return CUDADeviceStreamRef::create(*this, stream,
                                     /*isOwner=*/true);
}

ErrorOr<DeviceStreamRef>
CUDADeviceContext::createExternalStream(void *externalStream) const {
  return CUDADeviceStreamRef::create(*this,
                                     reinterpret_cast<CUstream>(externalStream),
                                     /*isOwner=*/false);
}

ErrorOrSuccess CUDADeviceContext::synchronize() const {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::synchronize", TRACY_COLOR_PURPLE);

  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.cuStreamSynchronize(stream_));
  // Handle any pending events after synchronization.
  ERR_CHECK_ERR(checkPendingWork());
  return success();
}

ErrorOr<DeviceEventRef> CUDADeviceContext::enqueueEvent() const {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::enqueueEvent", TRACY_COLOR_PURPLE);

  CUDA_WITH_CTX_ERR(*this);
  DEFINE_OR_RAISE(CUevent, event, CUDADevice::cast(device()).eventCachePop());
  CUDA_CHECK_ERR(driver_.cuEventRecord(event, stream_));
  return CUDADeviceEventRef::create(CUDADevice::cast(this->device()), event);
}

ErrorOr<DeviceEventRef>
CUDADeviceContext::eventCreate(unsigned int flags) const {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::eventCreate", TRACY_COLOR_PURPLE);

  CUDA_WITH_CTX_ERR(*this);
  CUevent event;
  CUDA_CHECK_ERR(driver_.cuEventCreate(&event, flags));
  return CUDADeviceEventRef::create(CUDADevice::cast(this->device()), event);
}

ErrorOrSuccess
CUDADeviceContext::enqueueWaitFor(const DeviceEvent &event) const {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::enqueueWaitFor", TRACY_COLOR_PURPLE);

  const auto &cudaEvent = CUDADeviceEvent::cast(event);
  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.cuStreamWaitEvent(stream_, cudaEvent.cudaEvent(), 0));
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess
CUDADeviceContext::enqueueWaitFor(const DeviceContext &ctx) const {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::enqueueWaitFor", TRACY_COLOR_PURPLE);

  DEFINE_OR_RAISE(DeviceEventRef, event, ctx.enqueueEvent());
  ERR_CHECK_ERR(enqueueWaitFor(*event));
  return success();
}

ErrorOr<DeviceTimerRef> CUDADeviceContext::startTimer() const {
  CUevent start;
  CUevent stop;

  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.cuEventCreate(&start, 0));
  CUDA_CHECK_ERR(driver_.cuEventCreate(&stop, 0));

  auto timer = CUDADeviceTimerRef::create(*this, start, stop);

  CUDA_CHECK_ERR(driver_.cuEventRecord(timer->cudaStartEvent(), stream_));

  return timer;
}

ErrorOr<std::chrono::nanoseconds>
CUDADeviceContext::stopTimer(const DeviceTimer &timer) const {
  const auto &cudaTimer = CUDADeviceTimer::cast(timer);

  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.cuEventRecord(cudaTimer.cudaStopEvent(), stream_));
  CUDA_CHECK_ERR(driver_.cuEventSynchronize(cudaTimer.cudaStopEvent()));

  float millis;
  CUDA_CHECK_ERR(driver_.cuEventElapsedTime(&millis, cudaTimer.cudaStartEvent(),
                                            cudaTimer.cudaStopEvent()));

  constexpr double nanosPerMillisecond = 1000.0 * 1000.0;
  double nanos = static_cast<double>(millis) * nanosPerMillisecond;
  return std::chrono::nanoseconds(static_cast<int64_t>(nanos));
}

ErrorOrSuccess
CUDADeviceContext::enablePeerAccess(const DeviceContext &peer) const {
  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.cuCtxEnablePeerAccess(
      CUDADeviceContext::cast(peer).cudaContext(), 0));

  // Also "enable peer access" for stream-ordered allocations.
  // Do so by allowing this device to access the peer's current memory pool.
  // NOTE: this only enables access to allocations from the peer's current
  // memory pool, so other memory pools won't be accessible.
  CUmemoryPool pool;
  CUDA_CHECK_ERR(driver_.cuDeviceGetMemPool(
      &pool, CUDADevice::cast(peer.device()).cudaDevice()));

  CUmemAccessDesc desc{};
  desc.location.type = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = device().id();
  desc.flags = CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUDA_CHECK_ERR(driver_.cuMemPoolSetAccess(pool, &desc, /*count=*/1));

  return success();
}

ErrorOr<Device::MemInfo> CUDADeviceContext::getDeviceMemoryInfo() const {
  CUDA_WITH_CTX(*this);
  // To get more accurate memory stats, ensure that at least all pending memory
  // allocations and frees have completed for this context's stream.
  CUDA_CHECK_ERR(driver_.cuStreamSynchronize(stream_));

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
  CUDA_CHECK_ERR(driver_.cuTensorMapEncodeTiled(
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

ErrorOr<std::string> CUDADevice::getFirstArchName() {
  ErrorOr<const NVML &> nvml = NVML::get();
  if (nvml.isError())
    return nvml.takeError();

  nvmlDevice_t device;
  if (nvml->DeviceGetHandleByIndex(0, &device) != NVML::SUCCESS)
    RAISE_ERROR("NVML: Unable to get device index");

  int major, minor;
  if (nvml->DeviceGetCudaComputeCapability(device, &major, &minor) !=
      NVML::SUCCESS)
    RAISE_ERROR("NVML: Unable to get compute capability");

  return cudaCCToSM(major, minor);
}
