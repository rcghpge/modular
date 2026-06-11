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
#include "CUDADeviceGraphBuilder.h"
#include "CUDADriver.h"
#include "MLRT/Driver/DeviceContext/ErrorMacros.h"

#include "MLRT/Driver/DeviceContext/LaunchTraceRegistry.h"
#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "NVML.h"
#include "Support/HashUtils.h"
#include "Support/Profiling/Tracy.h"
#include "Support/Profiling/TracyCuda.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Process.h"
#include <memory>
#include <mutex>
#include <optional>
#include <unistd.h>
#include <vector>

using namespace M;
using namespace M::Driver;

static bool isStreamCapturing(const CUDADriver &driver, CUstream stream) {
  CUstreamCaptureStatus status =
      CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
  if (driver.cuStreamIsCapturing(stream, &status) != CUDADriver::SUCCESS)
    return false;

  return status != CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
}

namespace {

// Helper class to set and restore the CUcontext around interactions with the
// CUDA driver.
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

#define NVML_CHECK_ERR(nvml_call)                                              \
  do {                                                                         \
    NVML::Result nvml_result = (nvml_call);                                    \
    if (nvml_result != NVML::SUCCESS) {                                        \
      return formatNVMLError(__FILE__, __LINE__, __FUNCTION__, nvml_result);   \
    }                                                                          \
  } while (0)

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

#define CUDA_WITH_CTX_ERR(ctx)                                                 \
  RETURN_IF_PENDING_ERROR(ctx);                                                \
  std::optional<CUDAContextScope> cudaWithContext;                             \
  if (!(ctx).isBoundToCurrentThread()) {                                       \
    cudaWithContext.emplace((ctx).cudaContext(), (ctx).cudaDriver());          \
    CUDA_CHECK_ERR(cudaWithContext->status());                                 \
  }

// This MUST be called during all synchronous API calls like
// cuStreamSynchronize. To reduce overhead, we omit polling during asynchronous
// API calls like cuLaunchKernel. The call to cuEventQuery is not free!
#define POLL_PENDING_EVENTS(ctx) RETURN_IF_ERROR((ctx).pollPendingEvents());

#define CUDA_WITH_CTX(ctx)                                                     \
  std::optional<CUDAContextScope> cudaWithContext;                             \
  if (!(ctx).isBoundToCurrentThread()) {                                       \
    cudaWithContext.emplace((ctx).cudaContext(), (ctx).cudaDriver());          \
    CUDA_CHECK(cudaWithContext->status());                                     \
  }

#define CUDA_WITH_DEV_ERR(dev)                                                 \
  CUDAContextScope cudaWithContext(*(dev).cudaContext(), (dev).cudaDriver());  \
  CUDA_CHECK_ERR(cudaWithContext.status());

#define CUDA_WITH_DEV(dev)                                                     \
  CUDAContextScope cudaWithContext(*(dev).cudaContext(), (dev).cudaDriver());  \
  CUDA_CHECK(cudaWithContext.status());

#define CHECK_IS_CUDA_ERR(dev)                                                 \
  do {                                                                         \
    if ((dev).api() != Device::cudaAPI) {                                      \
      RETURN_ERROR("Not a CUDA device.");                                      \
    }                                                                          \
  } while (0)

#define MAYBE_SYNC(ctx)                                                        \
  do {                                                                         \
    const CUDADeviceContext &cuda_ctx = (ctx);                                 \
    if (cuda_ctx.syncMode() &&                                                 \
        !isStreamCapturing(cuda_ctx.cudaDriver(), cuda_ctx.cudaStream())) {    \
      CUDA_CHECK_ERR(                                                          \
          cuda_ctx.cudaDriver().cuStreamSynchronize(cuda_ctx.cudaStream()));   \
    }                                                                          \
  } while (0)

[[maybe_unused]]
static Error formatNVMLError(const char *file, int line, const char *func,
                             NVML::Result result, std::string extra_msg = "") {
  ErrorOr<const NVML &> errNVML = NVML::get();
  const char *errStr = nullptr;

  if (!errNVML) {
    errStr = errNVML->nvmlErrorString(result);
  }

  return DeviceContext::formatError(file, line, func,
                                    llvm::Twine("NVML call failed: ") + errStr +
                                        (extra_msg.empty() ? "" : " - ") +
                                        extra_msg);
}

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
CUDADevice::CUDADevice(const CUDADriver &driver, int id,
                       nvmlDevice_t nvmlDevice, CUdevice cudaDevice,
                       CUcontext context, std::string name,
                       int computeCapabilityMajor, int computeCapabilityMinor,
                       Device::MemInfo memInfo, size_t allocAlignment)
    : Device(api, std::move(name), id, memInfo, allocAlignment),
      driver_(driver), cudaDevice_(cudaDevice), nvmlDevice_(nvmlDevice),
      context_(context), computeCapability_(cudaCCMonolithic(
                             computeCapabilityMajor, computeCapabilityMinor)),
      targetArch_(cudaCCToSM(computeCapabilityMajor, computeCapabilityMinor)) {}

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

CUdevice CUDADevice::cudaDevice() const { return cudaDevice_; }

nvmlDevice_t *CUDADevice::getNvmlDevice() { return &nvmlDevice_; }

ErrorOr<CUcontext> CUDADevice::cudaContext() const { return context_; }

ErrorOr<int> CUDADevice::getApiVersion() const {
  int apiVersion = 0;
  CUDA_CHECK_ERR(driver_.cuDriverGetVersion(&apiVersion));
  return apiVersion;
}

ErrorOr<DriverVersion &> CUDADevice::getDriverVersion() const {
  // Initialize the driver version once on first call of this method
  // or remember the error.
  static ErrorOr<std::string> errOrVersion = []() -> ErrorOr<std::string> {
    DEFINE_OR_RETURN_ERROR(const NVML &, nvml, NVML::get());
    DEFINE_OR_RETURN_ERROR(std::string, version, nvml.GetDriverVersion());
    return version;
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
#if !MODULAR_LINUX
  return Error("NVIDIA GPUs are only supported on Linux based systems.");
#endif

  // When MODULAR_NVPTX_COMPILER_PATH is set, the user is explicitly opting
  // into a fallback compilation path for older drivers or hardware. Skip the
  // minimum version checks since the whole point of the fallback is to bypass
  // our bundled libnvptxcompiler's requirements.
  if (llvm::sys::Process::GetEnv("MODULAR_NVPTX_COMPILER_PATH"))
    return success();

  if (computeCapability_ < MIN_COMPUTE_CAPABILITY) {
    RETURN_ERROR(
        "Only NVIDIA Turing (sm_75) architectures or higher are supported. "
        "Your GPU has compute capability " +
        llvm::Twine(computeCapability_) +
        ". For older hardware, set MODULAR_NVPTX_COMPILER_PATH to use an "
        "external ptxas binary.\nFor more information, see "
        "https://modul.ar/gpu-requirements.");
  }

  int driverVersion = 0;
  CUDA_CHECK_ERR(driver_.cuDriverGetVersion(&driverVersion));
  if (driverVersion < MIN_CUDA_DRIVER_VERSION) {
    std::string detectedVersion = "CUDA " + apiVersionToString(driverVersion);
    auto errOrDriverVersion = getDriverVersion();
    if (!errOrDriverVersion.isError())
      detectedVersion =
          errOrDriverVersion->toString() + " (" + detectedVersion + ")";
    RETURN_ERROR(
        "Your current NVIDIA GPU driver version is not supported.\n"
        "  Required: driver version >= " +
        llvm::Twine(MIN_NVIDIA_DRIVER_VERSION) +
        " (CUDA >= " + apiVersionToString(MIN_CUDA_DRIVER_VERSION) +
        ")\n"
        "  Detected: " +
        detectedVersion +
        "\n\nTo run on an older driver, set the MODULAR_NVPTX_COMPILER_PATH "
        "environment variable to the path of a system `ptxas` binary. "
        "For more information, see https://modul.ar/gpu-requirements.");
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

ErrorOrSuccess CUDADevice::enablePeerAccessImpl(const Device &peer) {
  CUDA_WITH_DEV_ERR(*this);

  const CUDADevice &peerCudaDev = CUDADevice::cast(peer);
  DEFINE_OR_RETURN_ERROR(CUcontext, peerCtx, peerCudaDev.cudaContext());
  DEFINE_OR_RETURN_ERROR(bool, canAccessPeer, canAccess(peerCudaDev));
  if (!canAccessPeer) {
    RETURN_ERROR("Cannot enable peer access from GPU " + Twine(id()) +
                 " to GPU " + Twine(peer.id()) +
                 ": hardware does not support P2P access");
  }

  // Tolerate "already enabled" as success to keep idempotent.
  CUDADriver::Result res = driver_.cuCtxEnablePeerAccess(peerCtx, /*flags=*/0);
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

ErrorOr<std::string> CUDADevice::getPciBusId() const {
  DEFINE_OR_RETURN_ERROR(const NVML &, nvml, NVML::get());

  nvmlDevice_t device;
  if (nvml.nvmlDeviceGetHandleByIndex(id(), &device) != NVML::SUCCESS)
    RETURN_ERROR("NVML: Unable to get device index");

  nvmlPciInfo_t pciInfo;
  if (nvml.nvmlDeviceGetPciInfo_v3(device, &pciInfo) != NVML::SUCCESS)
    RETURN_ERROR("NVML: Unable to get PCI info");
  return std::string(pciInfo.busId);
}

ErrorOr<bool> CUDADevice::supportsMulticast() const {
  // Check if multicast memory is supported.
  int multicastSupported = 0;
  CUDA_CHECK_ERR(cudaDriver().cuDeviceGetAttribute(
      &multicastSupported, CUdevice_attribute::MULTICAST_SUPPORTED,
      cudaDevice()));
  return multicastSupported != 0;
}

ErrorOr<DeviceContextRef> CUDADevice::createContext() {
  DEFINE_OR_RETURN_ERROR(CUcontext, context, cudaContext());

  // For the remainder of this method we use the context scope to ensure that
  // the context is properly removed when an error happens or then at the
  // return.
  CUDAContextScope withCtx(context, this->cudaDriver());

  CUstream stream;
  CUDA_CHECK_ERR(this->cudaDriver().cuStreamCreate(&stream, 0));

  return CUDADeviceContextRef::create(*this, context, stream);
}

ErrorOr<int> CUDADevice::numberOfGPUs() {
  static ErrorOr<unsigned int> errOrCount = []() -> ErrorOr<int> {
    DEFINE_OR_RETURN_ERROR(const NVML &, nvml, NVML::get());

    unsigned int result;
    if (nvml.nvmlDeviceGetCount_v2(&result) != NVML::SUCCESS)
      RETURN_ERROR("NVML: Unable to get device count");
    return result;
  }();

  if (errOrCount.isError()) {
    return errOrCount.copy();
  }

  int numGPUs = static_cast<int>(*errOrCount);

  // Users assume we use CUDA under the hood
  std::optional<std::string> envOr =
      llvm::sys::Process::GetEnv("CUDA_VISIBLE_DEVICES");
  if (envOr) {
    // Assuming responsible users
    // https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/
    // environment-variables.html
    int const numCommas =
        envOr->size() == 0 ? -1 : StringRef(*envOr).count(',');
    numGPUs = std::min(numGPUs, numCommas + 1);
  }
  return numGPUs;
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
                                      bool isHost) {
  assert(byteSize != 0);
  CUDA_WITH_DEV_ERR(*this);
  if (isHost) {
    CUDA_CHECK_ERR(driver_.cuMemAllocHost_v2(ptr, byteSize));
  } else {
    CUDA_CHECK_ERR(driver_.cuMemAlloc_v2(ptr, byteSize));
  }
  return success();
}

ErrorOrSuccess CUDADevice::chunkFree(void *ptr, bool isHost) {
  CUDA_WITH_DEV_ERR(*this);
  if (isHost) {
    CUDA_CHECK_ERR(driver_.cuMemFreeHost(ptr));
  } else {
    CUDA_CHECK_ERR(driver_.cuMemFree_v2(ptr));
  }
  return success();
}

// ---- CUDA Virtual Memory Management (VMM) ----
//
// Builds the CUmemAllocationProp describing a pinned device granule on this
// device. Mirrors the multicast path in createMulticastBuffer.
static CUmemAllocationProp cudaVmmAllocProp(CUdevice cudaDevice) {
  CUmemAllocationProp memProp = {};
  memProp.type = CUmemAllocationType::PINNED;
  memProp.location.type = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
  memProp.location.id = cudaDevice;
  return memProp;
}

ErrorOr<size_t> CUDADevice::vmmGranularity() const {
  CUDA_WITH_DEV_ERR(*this);
  CUmemAllocationProp memProp = cudaVmmAllocProp(cudaDevice_);
  size_t granularity = 0;
  CUDA_CHECK_ERR(driver_.cuMemGetAllocationGranularity(
      &granularity, &memProp,
      CUmemAllocationGranularity_flags::GRANULARITY_MINIMUM));
  return granularity;
}

ErrorOr<void *> CUDADevice::vmmAddressReserve(size_t byteSize) {
  CUDA_WITH_DEV_ERR(*this);
  CUdeviceptr va = nullptr;
  CUDA_CHECK_ERR(driver_.cuMemAddressReserve(&va, byteSize, 0, nullptr, 0));
  return reinterpret_cast<void *>(va);
}

ErrorOrSuccess CUDADevice::vmmAddressFree(void *va, size_t byteSize) {
  CUDA_WITH_DEV_ERR(*this);
  CUDA_CHECK_ERR(
      driver_.cuMemAddressFree(reinterpret_cast<CUdeviceptr>(va), byteSize));
  return success();
}

ErrorOr<uint64_t> CUDADevice::vmmCreate(size_t byteSize) {
  CUDA_WITH_DEV_ERR(*this);
  CUmemAllocationProp memProp = cudaVmmAllocProp(cudaDevice_);
  CUmemGenericAllocationHandle handle = 0;
  CUDA_CHECK_ERR(driver_.cuMemCreate(&handle, byteSize, &memProp, 0));
  return static_cast<uint64_t>(handle);
}

ErrorOrSuccess CUDADevice::vmmRelease(uint64_t handle) {
  CUDA_WITH_DEV_ERR(*this);
  CUDA_CHECK_ERR(
      driver_.cuMemRelease(static_cast<CUmemGenericAllocationHandle>(handle)));
  return success();
}

const std::vector<CUmemAccessDesc> &CUDADevice::vmmAccessDescs() const {
  std::call_once(vmmAccessOnce_, [&]() {
    auto pushDesc = [&](CUdevice dev) {
      CUmemAccessDesc d = {};
      d.location.type = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
      d.location.id = dev;
      d.flags = CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      vmmAccessDescs_.push_back(d);
    };
    // The owning device always has access.
    pushDesc(cudaDevice_);

    // Grant access to every other GPU that can peer-access this device's memory
    // over a fast link. Use the same P2P + native-atomic (NVLink proxy) policy
    // as canAccess(), evaluated via raw ordinals so we do not eagerly
    // initialize peer device contexts.
    int count = 0;
    if (driver_.cuDeviceGetCount(&count) != CUDADriver::SUCCESS)
      return;
    for (int ord = 0; ord < count; ++ord) {
      CUdevice peerDev;
      if (driver_.cuDeviceGet(&peerDev, ord) != CUDADriver::SUCCESS)
        continue;
      if (peerDev == cudaDevice_)
        continue;
      // Can `peerDev` access memory physically resident on this device?
      int canAccessPeer = 0;
      if (driver_.cuDeviceCanAccessPeer(&canAccessPeer, peerDev, cudaDevice_) !=
              CUDADriver::SUCCESS ||
          !canAccessPeer)
        continue;
      int isNativeAtomicSupported = 0;
      if (driver_.cuDeviceGetP2PAttribute(
              &isNativeAtomicSupported,
              CUdevice_P2PAttribute::
                  CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED,
              peerDev, cudaDevice_) != CUDADriver::SUCCESS ||
          !isNativeAtomicSupported)
        continue;
      pushDesc(peerDev);
    }
  });
  return vmmAccessDescs_;
}

ErrorOrSuccess CUDADevice::vmmMapAndSetAccess(void *va, size_t byteSize,
                                              uint64_t handle) {
  CUDA_WITH_DEV_ERR(*this);
  CUdeviceptr ptr = reinterpret_cast<CUdeviceptr>(va);
  CUDA_CHECK_ERR(driver_.cuMemMap(
      ptr, byteSize, 0, static_cast<CUmemGenericAllocationHandle>(handle), 0));
  // Grant read/write access to the owning device and every P2P-capable peer so
  // multi-GPU (expert-parallel) kernels and collectives can read VMM-backed
  // buffers. VMM access does not inherit from cuCtxEnablePeerAccess.
  const std::vector<CUmemAccessDesc> &descs = vmmAccessDescs();
  CUDA_CHECK_ERR(
      driver_.cuMemSetAccess(ptr, byteSize, descs.data(), descs.size()));
  return success();
}

ErrorOrSuccess CUDADevice::vmmUnmap(void *va, size_t byteSize) {
  CUDA_WITH_DEV_ERR(*this);
  CUDA_CHECK_ERR(
      driver_.cuMemUnmap(reinterpret_cast<CUdeviceptr>(va), byteSize));
  return success();
}

ErrorOr<Device::MemInfo> CUDADevice::getMemoryInfo() const {
  Device::MemInfo memInfo;
  CUDA_WITH_DEV_ERR(*this);
  CUDA_CHECK_ERR(driver_.cuMemGetInfo_v2(&memInfo.free, &memInfo.total));
  return memInfo;
}

ErrorOr<Device::GraphMemInfo> CUDADevice::getGraphMemoryInfo() const {
  Device::GraphMemInfo info;
  CUDA_WITH_DEV_ERR(*this);
  uint64_t reserved = 0, used = 0;
  CUDA_CHECK_ERR(driver_.cuDeviceGetGraphMemAttribute(
      cudaDevice_, CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT,
      &reserved));
  CUDA_CHECK_ERR(driver_.cuDeviceGetGraphMemAttribute(
      cudaDevice_, CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT,
      &used));
  info.reserved = static_cast<size_t>(reserved);
  info.used = static_cast<size_t>(used);
  return info;
}

ErrorOr<void *> CUDADevice::getCurrentDriverContext() const {
  CUcontext result = nullptr;
  CUDA_CHECK_ERR(driver_.cuCtxGetCurrent(&result));
  return result;
}
ErrorOrSuccess CUDADevice::setCurrentDriverContext(void *ctx) {
  CUDA_CHECK_ERR(driver_.cuCtxSetCurrent(reinterpret_cast<CUcontext>(ctx)));
  return success();
}

const CUDADevice &CUDADevice::cast(const Device &device) {
  return static_cast<const CUDADevice &>(device);
}

CUDADevice &CUDADevice::cast(Device &device) {
  return static_cast<CUDADevice &>(device);
}

ErrorOr<CUDADevice &> CUDADevice::get(int id) {
  // On first call initialize devices and err. devices is only set if there was
  // no error during setup.
  static std::mutex mu;
  static DeviceList *devices = nullptr;
  static const ErrorOrSuccess err = [&]() -> ErrorOrSuccess {
    DEFINE_OR_RETURN_ERROR(int, count, numberOfGPUs());
    devices = new DeviceList(count, nullptr);
    return success();
  }();

  // If we did get an error during setup, skip accessing the device and forward
  // the error.
  if (err.isError()) {
    RETURN_ERROR(err.getError());
  }

  assert(devices != nullptr);
  assert((id >= 0) && (static_cast<size_t>(id) < devices->size()));

  // Get or initialize the Device object.
  std::lock_guard<std::mutex> lock(mu);
  CUDADevice *device = (*devices)[id];
  if (device == nullptr) {
    DEFINE_OR_RETURN_ERROR(const NVML &, nvml, NVML::get());

    nvmlDevice_t nvmlDevice;
    NVML_CHECK_ERR(nvml.nvmlDeviceGetHandleByIndex(id, &nvmlDevice));

    char name[256];
    NVML_CHECK_ERR(nvml.nvmlDeviceGetName(nvmlDevice, name, sizeof(name)));

    nvmlMemory_t nvmlMemInfo;
    auto nvmlMemResult = nvml.nvmlDeviceGetMemoryInfo(nvmlDevice, &nvmlMemInfo);

    Device::MemInfo memInfo;
    if (nvmlMemResult == NVML::SUCCESS) {
      memInfo.free = nvmlMemInfo.free;
      memInfo.total = nvmlMemInfo.total;
    } else if (nvmlMemResult == NVML::NOT_SUPPORTED) {
      // On unified memory systems (e.g. NVIDIA Grace Blackwell GB10), NVML
      // cannot report discrete GPU memory. Fall back to system memory info
      // since CPU and GPU share the same physical memory.
      long pages = sysconf(_SC_PHYS_PAGES);
      long pageSize = sysconf(_SC_PAGE_SIZE);
#if MODULAR_LINUX
      long availPages = sysconf(_SC_AVPHYS_PAGES);
#else
      long availPages = pages;
#endif
      if (pages < 0 || pageSize < 0 || availPages < 0)
        RETURN_ERROR("NVML: Unable to get memory info on unified memory "
                     "system: sysconf failed");
      memInfo.total =
          static_cast<size_t>(pages) * static_cast<size_t>(pageSize);
      memInfo.free =
          static_cast<size_t>(availPages) * static_cast<size_t>(pageSize);
    } else {
      RETURN_ERROR("NVML: Unable to get memory info");
    }

    // Some info is available only via CUDA. Ideally, we will use only NVML,
    // and not CUDA in CUDADevice.
    DEFINE_OR_RETURN_ERROR(const CUDADriver &, driver, CUDADriver::get());
    int alignment;
    CUdevice cuDevice;
    CUcontext cuContext;

    CUDA_CHECK_ERR(driver.cuDeviceGet(&cuDevice, id));
    // Alignment does not seem to be available through NVML.
    CUDA_CHECK_ERR(driver.cuDeviceGetAttribute(
        &alignment, CUdevice_attribute::TEXTURE_ALIGNMENT, cuDevice));

    // Use the primary context for this device, with the effect that we share
    // the same context with other users of the CUDA Runtime API.
    CUDA_CHECK_ERR(driver.cuDevicePrimaryCtxRetain(&cuContext, cuDevice));

    int computeCapabilityMajor, computeCapabilityMinor;
    NVML_CHECK_ERR(nvml.nvmlDeviceGetCudaComputeCapability(
        nvmlDevice, &computeCapabilityMajor, &computeCapabilityMinor));
    device = new CUDADevice(driver, id, nvmlDevice, cuDevice, cuContext, name,
                            computeCapabilityMajor, computeCapabilityMinor,
                            memInfo, alignment);

    // Once the Device has been created remember it in the devices list.
    (*devices)[id] = device;

    // Run healthcheck after device init — returns error on unhealthy GPU.
    RETURN_IF_ERROR(device->runHealthcheck());
  }
  return *device;
}

ErrorOrSuccess CUDADevice::runHealthcheck() const {
  if (DeviceContext::configAsBool("disable-healthcheck", false))
    return success();

  auto nvmlOr = NVML::get();
  if (nvmlOr.isError())
    return success(); // Gracefully skip if NVML unavailable.

  const NVML &nvml = *nvmlOr;

  GPUHealthData data;
  data.deviceId = id();
  data.deviceName = name();

  // Gather throttle reasons — map NVML-specific bitmask to platform-agnostic
  // GPUHealthData constants.
  unsigned long long nvmlThrottle = 0;
  if (nvml.nvmlDeviceGetCurrentClocksThrottleReasons(
          nvmlDevice_, &nvmlThrottle) == NVML::SUCCESS) {
    data.throttleQueried = true;
    if (nvmlThrottle & NVML_CLK_THROTTLE_REASON_HW_SLOWDOWN)
      data.throttleReasons |= GPUHealthData::THROTTLE_HW_SLOWDOWN;
    if (nvmlThrottle & NVML_CLK_THROTTLE_REASON_HW_THERMAL_SLOWDOWN)
      data.throttleReasons |= GPUHealthData::THROTTLE_HW_THERMAL_SLOWDOWN;
    if (nvmlThrottle & NVML_CLK_THROTTLE_REASON_HW_POWER_BRAKE_SLOWDOWN)
      data.throttleReasons |= GPUHealthData::THROTTLE_HW_POWER_BRAKE;
  }

  // Gather ECC errors.
  data.eccQueried = nvml.nvmlDeviceGetTotalEccErrors(
                        nvmlDevice_, NVML_MEMORY_ERROR_TYPE_UNCORRECTABLE,
                        NVML_VOLATILE_ECC, &data.eccErrors) == NVML::SUCCESS;

  return evaluateGPUHealth(data);
}

ErrorOr<const char *> CUDADevice::getArchName() const {
  return targetArch_.c_str();
}

void CUDADevice::eventCacheClear() const {
  std::lock_guard<std::mutex> lock(mu_);
  for (const auto &event : eventCache_) {
    // Ignore errors during destruction of device object.
    cudaDriver().cuEventDestroy(event);
  }
}

ErrorOr<CUevent> CUDADevice::eventCachePop(bool useCache) const {
  if (useCache) {
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
CUDADeviceMulticastBuffer::multicastBuffer(DeviceContext &ctx) const {
  return ctx.createBufferOwning(multicastPtr_, byteSize(),
                                /* owning = */ false);
}

ErrorOr<DeviceBufferRef>
CUDADeviceMulticastBuffer::unicastBufferFor(DeviceContext &ctx) const {
  CUdevice dev = CUDADevice::cast(ctx.device()).cudaDevice();
  for (const auto &it : unicastBuffers_) {
    if (it.device_ == dev) {
      return ctx.createBufferOwning(it.ptr_, byteSize(), /* owning = */ false);
    }
  }
  RETURN_ERROR("device is not part of this DeviceMulticastBuffer");
}

size_t CUDADeviceMulticastBuffer::numberOfDevices() const {
  return unicastBuffers_.size();
}

ErrorOr<DeviceMulticastBufferRef>
CUDADeviceMulticastBuffer::allocate(const std::vector<const Device *> &devices,
                                    size_t byteSize) {
  if (devices.size() < 2)
    RETURN_ERROR(
        "At least two DeviceContexts needed for device multicast allocation.");
  for (const auto &it : devices) {
    if (it->api() != CUDADevice::api)
      RETURN_ERROR("Only CUDADeviceContext supports device multicast access.");
  }
  const CUDADevice &dev0 = CUDADevice::cast(*devices[0]);
  const CUDADriver &driver = dev0.cudaDriver();

  // Verify that multicast memory is supported.
  int multicastSupported = 0;
  CUDA_CHECK_ERR(driver.cuDeviceGetAttribute(
      &multicastSupported, CUdevice_attribute::MULTICAST_SUPPORTED,
      dev0.cudaDevice()));
  if (multicastSupported == 0)
    RETURN_ERROR("Multicast buffer not supported.");

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
CUDADeviceEvent::CUDADeviceEvent(CUDADevice &device, CUevent event,
                                 bool recordedDuringCapture, bool enableTiming)
    : DeviceEvent(device), event_(event),
      recordedDuringCapture_(recordedDuringCapture),
      enableTiming_(enableTiming) {}

CUDADeviceEvent::~CUDADeviceEvent() {
  const auto &cudaDevice = CUDADevice::cast(device());
  CUDA_WITH_DEV(cudaDevice);
  if (recordedDuringCapture_ || enableTiming_) {
    // Timing-enabled events have different flags than cached events,
    // so they must not be returned to the non-timing event cache.
    // Events recorded during capture also bypass the cache.
    cudaDevice.cudaDriver().cuEventDestroy(event_);
  } else {
    cudaDevice.eventCachePush(event_);
  }
}

ErrorOrSuccess CUDADeviceEvent::synchronize() {
  const auto &cudaDevice = CUDADevice::cast(device());
  CUDA_WITH_DEV_ERR(cudaDevice);
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().cuEventSynchronize(event_));
  return success();
}

ErrorOr<bool> CUDADeviceEvent::isReady() const {
  if (recordedDuringCapture_) {
    // Events recorded during stream capture become graph nodes; polling them
    // directly is not allowed, so treat them as ready.
    return true;
  }
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

ErrorOr<float> CUDADeviceEvent::elapsedTime(const DeviceEvent &end) const {
  const auto &cudaEnd = CUDADeviceEvent::cast(end);
  const auto &cudaDevice = CUDADevice::cast(device());
  CUDA_WITH_DEV_ERR(cudaDevice);
  float millis;
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().cuEventElapsedTime(
      &millis, event_, cudaEnd.cudaEvent()));
  return millis;
}

CUevent CUDADeviceEvent::cudaEvent() const { return event_; }

const CUDADeviceEvent &CUDADeviceEvent::cast(const DeviceEvent &event) {
  return static_cast<const CUDADeviceEvent &>(event);
}

// ===== CUDADeviceGraph ===== //
CUDADeviceGraph::~CUDADeviceGraph() {
  const auto &cudaDevice = CUDADevice::cast(device());
  CUDA_WITH_DEV(cudaDevice);
  if (rawGraph_)
    CUDA_CHECK(cudaDevice.cudaDriver().cuGraphDestroy(rawGraph_));
  if (exec_)
    CUDA_CHECK(cudaDevice.cudaDriver().cuGraphExecDestroy(exec_));
}

static std::unique_lock<std::mutex> getCudaProfilerLock() {
  // CUPTI's graph-tracing hooks (active when any CUDA injection profiler such
  // as nsys or ncu is running) are not thread-safe for concurrent
  // cuGraphInstantiateWithParams calls, even on distinct CUgraph objects.
  // Serialize instantiation under a process-wide mutex when profiling is
  // detected via CUDA_INJECTION64_PATH.

  static const bool kUnderCudaProfiler =
      llvm::sys::Process::GetEnv("CUDA_INJECTION64_PATH").has_value();

  if (kUnderCudaProfiler) {
    static std::mutex sInstantiateMutex;
    std::unique_lock<std::mutex> profilingLock(sInstantiateMutex);
    return profilingLock;
  }

  return {};
}

ErrorOrSuccess CUDADeviceGraph::instantiate() {
  const auto &cudaDevice = CUDADevice::cast(device());
  CUDA_WITH_DEV_ERR(cudaDevice);

  // Prepare instantiation parameters with enhanced error reporting.
  CUDA_GRAPH_INSTANTIATE_PARAMS params = {
      .flags =
          M::to_underlying(CUgraphInstantiateFlags::
                               CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH),
      .hUploadStream = nullptr,
      .hErrNode_out = nullptr,
      .result_out = CUgraphInstantiateResult::CUDA_GRAPH_INSTANTIATE_SUCCESS};

  CUgraphExec exec = nullptr;
  CUDADriver::Result res = [&] {
    auto lock = getCudaProfilerLock();
    return cudaDevice.cudaDriver().cuGraphInstantiateWithParams(
        &exec, rawGraph_, &params);
  }();

  if (res != CUDADriver::SUCCESS) {
    std::string errorMsg = "failed to instantiate CUDA graph";
    if (params.result_out !=
        CUgraphInstantiateResult::CUDA_GRAPH_INSTANTIATE_SUCCESS) {
      errorMsg += ": ";
      errorMsg += toString(params.result_out);
    }
    if (params.hErrNode_out != nullptr) {
      errorMsg +=
          " at node " + llvm::formatv("{0}", (void *)params.hErrNode_out).str();
    }
    RETURN_ERROR(errorMsg);
  }

  // Destroy the raw graph now that instantiation succeeded; exec_ takes over.
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().cuGraphDestroy(rawGraph_));
  rawGraph_ = nullptr;
  exec_ = exec;
  return success();
}

ErrorOrSuccess CUDADeviceGraph::replay() const {
  if (!hasStream())
    RETURN_ERROR("cannot replay CUDA graph: no stream stored during capture");
  const auto *cudaStream = CUDADeviceStream::verified_cast(captureStream());
  if (!cudaStream)
    RETURN_ERROR(
        "cannot replay CUDA graph: stored stream is not a CUDA stream");
  const auto &cudaDevice = CUDADevice::cast(device());
  CUDA_WITH_DEV_ERR(cudaDevice);
  CUDA_CHECK_ERR(
      cudaDevice.cudaDriver().cuGraphLaunch(exec_, cudaStream->cudaStream()));
  return success();
}

// ===== CUDADeviceStream ===== //
CUDADeviceStream::CUDADeviceStream(CUDADeviceContext &ctx, CUstream stream,
                                   bool isOwner)
    : DeviceStream(ctx), stream_(stream), isOwner_(isOwner) {
  logGraphCapture_ = DeviceContext::configAsString("log_graph_capture", "");
}

CUDADeviceStream::~CUDADeviceStream() {
  if (!isOwner_) {
    return;
  }
  CUDA_WITH_CTX(CUDADeviceContext::cast(context_()));
  const auto &cudaDevice = CUDADevice::cast(this->device());
  CUDA_CHECK(cudaDevice.cudaDriver().cuStreamDestroy(stream_));
  isOwner_ = false;
}

ErrorOrSuccess CUDADeviceStream::synchronize() {
  const auto &cudaDevice = CUDADevice::cast(this->device());

  CUDA_WITH_DEV_ERR(cudaDevice);
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().cuStreamSynchronize(stream_));
  POLL_PENDING_EVENTS(this->context_());
  return success();
}

ErrorOr<DeviceEventRef> CUDADeviceStream::enqueueEvent() const {
  CUDADevice &cudaDevice = CUDADevice::cast(this->device());
  const bool recordedDuringCapture =
      ::isStreamCapturing(cudaDevice.cudaDriver(), stream_);

  DEFINE_OR_RETURN_ERROR(
      CUevent, event,
      cudaDevice.eventCachePop(/*useCache=*/!recordedDuringCapture));
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().cuEventRecord(event, stream_));
  return CUDADeviceEventRef::create(cudaDevice, event, recordedDuringCapture);
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

ErrorOrSuccess CUDADeviceStream::enqueueHostFunc(void (*fn)(void *),
                                                 void *userData) const {
  const auto &cudaDevice = CUDADevice::cast(this->device());
  CUDA_WITH_DEV_ERR(cudaDevice);
  CUDA_CHECK_ERR(
      cudaDevice.cudaDriver().cuLaunchHostFunc(stream_, fn, userData));
  return success();
}

ErrorOrSuccess
CUDADeviceStream::enqueueWaitOnHostValue(const CompletionFlag &flag,
                                         uint64_t value) const {
  const auto &cudaDevice = CUDADevice::cast(this->device());
  CUDA_WITH_DEV_ERR(cudaDevice);
  // Wait until *flag.devicePtr() == value. CU_STREAM_WAIT_VALUE_FLUSH is
  // intentionally omitted: it's gated by CAN_FLUSH_REMOTE_WRITES, which
  // is 0 on most current devices, and writes from a normal host thread
  // are visible via the device-mapped pinned alias without it.
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().cuStreamWaitValue64_v2(
      stream_, reinterpret_cast<CUdeviceptr>(flag.devicePtr()), value,
      static_cast<unsigned int>(
          CUstreamWaitValue_flags::CU_STREAM_WAIT_VALUE_EQ)));
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
    void *attrs, uint32_t numAttrs, void **args, uint32_t /*argCount*/,
    const size_t * /*argSizes*/) const {
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

CUstreamCaptureStatus CUDADeviceStream::getCaptureStatus() const {
  const auto &cudaDevice = CUDADevice::cast(this->device());
  CUstreamCaptureStatus status =
      CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
  cudaDevice.cudaDriver().cuStreamIsCapturing(stream_, &status);
  return status;
}

ErrorOrSuccess CUDADeviceStream::beginCapture() const {
  CUstreamCaptureStatus status = getCaptureStatus();
  if (status == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE)
    RETURN_ERROR("stream is already capturing");
  if (status == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED)
    RETURN_ERROR("stream capture was invalidated");

  const auto &cudaDevice = CUDADevice::cast(this->device());
  CUDA_WITH_DEV_ERR(cudaDevice);
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().cuStreamBeginCapture(
      stream_, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED));
  return success();
}

ErrorOr<DeviceGraphRef> CUDADeviceStream::endCapture() {
  CUstreamCaptureStatus status = getCaptureStatus();
  if (status == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE)
    RETURN_ERROR("stream is not capturing");
  if (status == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED)
    RETURN_ERROR("stream capture was invalidated");

  CUDADevice &cudaDevice = CUDADevice::cast(this->device());
  CUDA_WITH_DEV_ERR(cudaDevice);

  CUgraph graph = nullptr;
  CUDA_CHECK_ERR(cudaDevice.cudaDriver().cuStreamEndCapture(stream_, &graph));

  if (!graph)
    RETURN_ERROR("cuStreamEndCapture returned null graph");

  if (logGraphCapture_ != "") {
    static std::atomic<int32_t> graphCaptureCount = 0;

    int32_t currentId = graphCaptureCount.fetch_add(1);
    // Debug: print graph to DOT file for visualization.
    // This is only enabled at compile time for debugging purposes.
    // The DOT file can be visualized with Graphviz tools (e.g., dot -Tpng).
    // Set MODULAR_CUDA_GRAPH_DOT_PATH to customize the output path.
    std::string outputPath =
        llvm::formatv("{}/cuda_graph_{}.dot", logGraphCapture_, currentId)
            .str();
    cudaDevice.cudaDriver().cuGraphDebugDotPrint(
        graph, outputPath.c_str(),
        static_cast<unsigned>(CUgraphDotFlags::Verbose));
  }

  // Store the stream for replay/verification.
  return CUDADeviceGraphRef::create(
      cudaDevice, graph,
      DeviceStreamRef::copy(static_cast<DeviceStream *>(this)));
}

void CUDADeviceStream::beginLaunchTrace() const {
  auto &registry = launchTraceRegistry();
  registry.beginTraceForStream(static_cast<void *>(stream_));
}

std::vector<KernelLaunchMetadata> CUDADeviceStream::takeLaunchTrace() const {
  auto &registry = launchTraceRegistry();
  return registry.takeTraceForStream(static_cast<void *>(stream_));
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
CUfunction CUDADeviceFunction::cudaFunction() const {
  return moduleFunction_->cudaFunction();
}

CUmodule CUDAModuleFunction::cudaModule() const { return module_; }
CUfunction CUDAModuleFunction::cudaFunction() const { return function_; }

ArrayRef<size_t> CUDAModuleFunction::paramSizes() const {
  std::call_once(paramSizesOnce_, [&] {
    size_t paramIdx = 0;
    size_t offset = 0;
    size_t size = 0;
    // cuFuncGetParamInfo returns the offset/size of each parameter in the
    // device-side parameter layout for this CUfunction. The CUfunction is
    // created from a loaded module (PTX/cubin/fatbin), and the driver exposes
    // that layout metadata so we can size each argument for launch tracing.
    while (driver_.cuFuncGetParamInfo(function_, paramIdx, &offset, &size) ==
           CUDADriver::SUCCESS) {
      paramSizes_.push_back(size);
      ++paramIdx;
    }
  });
  return paramSizes_;
}

const CUDAModuleFunction *
CUDAModuleFunction::cast(const DeviceFunctionCacheEntry *entry) {
  return static_cast<const CUDAModuleFunction *>(entry);
}

// ===== CUDADeviceFunction ===== //
CUDADeviceFunction::CUDADeviceFunction(CUDADeviceContext &context,
                                       CUDAModuleFunctionRef modFunc)
    : DeviceFunction(context), moduleFunction_(std::move(modFunc)) {}
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

  // Record launch trace if enabled
  auto &registry = launchTraceRegistry();
  if (registry.isLaunchTraceEnabled(stream)) {
    // Convert CUDA-specific types to generic interface
    ArrayRef<uint8_t> attrsArray;
    if (attrs && numAttrs > 0) {
      attrsArray = ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(attrs),
                                     sizeof(CUlaunchAttribute) * numAttrs);
    }

    ArrayRef<size_t> paramSizes = moduleFunction_->paramSizes();
    ArrayRef<void *> argsArray = ArrayRef<void *>(args, paramSizes.size());

    registry.recordLaunchTrace(
        moduleFunction_->name(),
        reinterpret_cast<uintptr_t>(moduleFunction_->cudaFunction()),
        moduleFunction_->paramSizes(), gridX, gridY, gridZ, blockX, blockY,
        blockZ, sharedMemBytes, attrsArray, numAttrs, argsArray, stream);
  }

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
                                                       size_t byteSize) {
  TRACY_ZONE_SCOPED_NC("CUDADeviceFunction::copyToNamedConstant",
                       TRACY_COLOR_PURPLE);
  CUdeviceptr devicePtr;
  size_t deviceSize;
  const auto &cudaContext = CUDADeviceContext::cast(context());
  CUDA_CHECK_ERR(cudaContext.cudaDriver().cuModuleGetGlobal_v2(
      &devicePtr, &deviceSize, moduleFunction_->cudaModule(), name));

  if (deviceSize < byteSize) {
    RETURN_ERROR("byteSize too big");
  }

  CUDA_CHECK_ERR(cudaContext.cudaDriver().cuMemcpyHtoDAsync_v2(
      devicePtr, data, byteSize, cudaContext.cudaStream()));
  MAYBE_SYNC(CUDADeviceContext::cast(context()));
  return success();
}

ErrorOr<int32_t> CUDADeviceFunction::getAttribute(int32_t code) {
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

CUDADeviceFunction &CUDADeviceFunction::cast(DeviceFunction &func) {
  return static_cast<CUDADeviceFunction &>(func);
}

ErrorOrSuccess CUDADeviceFunction::occupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, int blockSize, size_t dynamicSharedMemSize) {
  auto &cudaContext = CUDADeviceContext::cast(context());
  CUDA_WITH_CTX_ERR(cudaContext);
  CUDA_CHECK_ERR(
      moduleFunction_->cudaDriver().cuOccupancyMaxActiveBlocksPerMultiprocessor(
          numBlocks, moduleFunction_->cudaFunction(), blockSize,
          dynamicSharedMemSize));
  return success();
}

// ===== CUDADeviceTimer ===== //
CUDADeviceTimer::CUDADeviceTimer(CUDADeviceContext &context, CUevent start,
                                 CUevent stop)
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
CUDADeviceContext::CUDADeviceContext(CUDADevice &device, CUcontext context,
                                     CUstream stream)
    : DeviceContext(device), driver_(device.cudaDriver()), context_(context),
      stream_(stream) {
  logLoadFunction_ = DeviceContext::configAsBool("log_load_function", false);
  logVerbose_ = DeviceContext::configAsBool("log_verbose", false);

  // Initialize Tracy CUDA profiling context if not already initialized.
  M::initTracyCuda();
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
                                           Device::AllocMode mode) {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::memAlloc", TRACY_COLOR_PURPLE);

  assert(byteSize != 0);
  CUDA_WITH_CTX_ERR(*this);
  switch (mode) {
  case Device::AllocMode::ASYNC: {
    // We used to support async allocations but prefer only sync ones now.
    return Error("Async allocation is not supported for CUDA devices");

    // Deprecated logic. Remove later.
    // CUDA_CHECK_ERR(driver_.cuMemAllocAsync(ptr, byteSize, stream_));
    // MAYBE_SYNC(*this);
    break;
  }
  case Device::AllocMode::HOST: {
    // We must poll here since cuMemAllocHost_v2 is synchronous.
    POLL_PENDING_EVENTS(*this);
    CUDA_CHECK_ERR(driver_.cuMemAllocHost_v2(ptr, byteSize));
    break;
  }
  case Device::AllocMode::SYNC: {
    POLL_PENDING_EVENTS(*this);
    CUDA_CHECK_ERR(driver_.cuMemAlloc_v2(ptr, byteSize));
    break;
  }
  }
  return success();
}

ErrorOr<CompletionFlagRef> CUDADeviceContext::allocateCompletionFlag() {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::allocateCompletionFlag",
                       TRACY_COLOR_PURPLE);
  CUDA_WITH_CTX_ERR(*this);
  POLL_PENDING_EVENTS(*this);
  void *hostPtr = nullptr;
  // CU_MEMHOSTALLOC_PORTABLE (0x1) | CU_MEMHOSTALLOC_DEVICEMAP (0x2).
  // DEVICEMAP is required so cuMemHostGetDevicePointer_v2 succeeds on
  // non-UVA systems. PORTABLE matches the existing cuMemAllocHost_v2
  // path and allows the same host allocation to be used from any
  // context on this process.
  constexpr unsigned int kFlags = 0x1 | 0x2;
  CUDA_CHECK_ERR(driver_.cuMemHostAlloc(&hostPtr, sizeof(uint64_t), kFlags));
  CUdeviceptr devPtr = nullptr;
  CUDADriver::Result r =
      driver_.cuMemHostGetDevicePointer_v2(&devPtr, hostPtr, 0);
  if (r != CUDADriver::SUCCESS) {
    (void)driver_.cuMemFreeHost(hostPtr);
    const char *errName = nullptr;
    (void)driver_.cuGetErrorName(r, &errName);
    return Error(
        std::string("cuMemHostGetDevicePointer_v2 failed for CompletionFlag: ")
            .append(errName ? errName : "(unknown)"));
  }
  // Initialize the flag to zero so the first wait observes a clean state.
  *static_cast<uint64_t *>(hostPtr) = 0;
  return CompletionFlagRef::take(
      new CompletionFlag(*this, hostPtr, reinterpret_cast<uint64_t>(devPtr)));
}

ErrorOrSuccess CUDADeviceContext::freeCompletionFlag(void *hostPtr) {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::freeCompletionFlag",
                       TRACY_COLOR_PURPLE);
  CUDA_WITH_CTX_ERR(*this);
  POLL_PENDING_EVENTS(*this);
  CUDA_CHECK_ERR(driver_.cuMemFreeHost(hostPtr));
  return success();
}

ErrorOrSuccess CUDADeviceContext::registerHostMemory(void *ptr,
                                                     size_t byteSize) {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::registerHostMemory",
                       TRACY_COLOR_PURPLE);
  CUDA_WITH_CTX_ERR(*this);
  // CU_MEMHOSTREGISTER_PORTABLE (0x1): the registration is visible from every
  // context in the process
  constexpr unsigned int kFlags = 0x1;
  CUDA_CHECK_ERR(driver_.cuMemHostRegister_v2(ptr, byteSize, kFlags));
  return success();
}

ErrorOrSuccess CUDADeviceContext::unregisterHostMemory(void *ptr) {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::unregisterHostMemory",
                       TRACY_COLOR_PURPLE);
  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.cuMemHostUnregister(ptr));
  return success();
}

ErrorOrSuccess CUDADeviceContext::memFree(void *ptr, bool isHost) {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::memFree", TRACY_COLOR_PURPLE);

  CUDA_WITH_CTX_ERR(*this);
  if (isHost) {
    POLL_PENDING_EVENTS(*this);
    CUDA_CHECK_ERR(driver_.cuMemFreeHost(ptr));
  } else {
    // Defer cuMemFree_v2 until after GPU stream work completes. This prevents
    // a use-after-free when a kernel (e.g. allreduce reduce-scatter) is still
    // reading memory via P2P on another GPU while the CPU drops its reference.
    // During CUDA graph capture the stream records ops but does not execute
    // them, so no P2P race is possible; free immediately to keep the memory
    // pool available for subsequent capture allocations.
    if (isStreamCapturing(driver_, stream_)) {
      CUDA_CHECK_ERR(driver_.cuMemFree_v2(ptr));
    } else {
      RETURN_IF_ERROR(enqueueEventHandler(
          [contextRef = DeviceContextRef::copy(this), ptr = ptr]() {
            auto &cudaCtx = static_cast<CUDADeviceContext &>(*contextRef);
            CUDA_WITH_CTX(cudaCtx);
            CUDADriver::Result result = cudaCtx.driver_.cuMemFree_v2(ptr);
            if (result != CUDADriver::SUCCESS) {
              cudaCtx.setPendingError(
                  formatCUDAError(__FILE__, __LINE__, __FUNCTION__, result));
            }
          }));
    }
  }
  return success();
}

void *CUDADeviceContext::driverContext() const { return cudaContext(); }

ErrorOrSuccess CUDADeviceContext::enqueueCopyToDevice(void *dst,
                                                      const void *src,
                                                      size_t byteSize) {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::enqueueCopyToDevice",
                       TRACY_COLOR_PURPLE);
  CUDA_WITH_CTX_ERR(*this);

  CUDA_CHECK_ERR(driver_.cuMemcpyHtoDAsync_v2(dst, src, byteSize, stream_));
  auto &registry = launchTraceRegistry();
  if (registry.isLaunchTraceEnabled(static_cast<void *>(stream_))) {
    KernelLaunchMetadata metadata(
        "cuMemcpyHtoD", hashMemcpyTrace(LaunchTraceKind::kMemcpyHtoD, byteSize),
        KernelLaunchMetadata::MemcpyKind::HtoD, byteSize);
    registry.appendLaunchTraceEntry(static_cast<void *>(stream_), metadata);
  }
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess CUDADeviceContext::enqueueCopyFromDevice(void *dst,
                                                        const void *src,
                                                        size_t byteSize) {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::enqueueCopyFromDevice",
                       TRACY_COLOR_PURPLE);
  CUDA_WITH_CTX_ERR(*this);

  CUDA_CHECK_ERR(driver_.cuMemcpyDtoHAsync_v2(dst, const_cast<void *>(src),
                                              byteSize, stream_));
  auto &registry = launchTraceRegistry();
  if (registry.isLaunchTraceEnabled(static_cast<void *>(stream_))) {
    KernelLaunchMetadata metadata(
        "cuMemcpyDtoH", hashMemcpyTrace(LaunchTraceKind::kMemcpyDtoH, byteSize),
        KernelLaunchMetadata::MemcpyKind::DtoH, byteSize);
    registry.appendLaunchTraceEntry(static_cast<void *>(stream_), metadata);
  }
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess CUDADeviceContext::memCopyDtoD(DeviceBuffer &dst,
                                              DeviceBuffer &src,
                                              size_t srcByteOffset,
                                              size_t srcByteSize) {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::memCopyDtoD", TRACY_COLOR_PURPLE);

  CHECK_IS_CUDA_ERR(src.context().device());
  CHECK_IS_CUDA_ERR(dst.context().device());
  CUDA_WITH_CTX_ERR(*this);

  auto &srcCtx = CUDADeviceContext::cast(src.context());
  auto &dstCtx = CUDADeviceContext::cast(dst.context());

  if (this->context_ != dstCtx.context_) {
    RETURN_ERROR("device context does not match context of dst");
  }

  CUdeviceptr srcPtr = (uint8_t *)src.ptr() + srcByteOffset;
  CUdeviceptr dstPtr = dst.ptr();
  size_t dstByteSize = dst.byteSize();
  assert(srcByteOffset + srcByteSize <= src.byteSize());
  assert(srcByteSize == dstByteSize);
  if (srcCtx.cudaStream() != dstCtx.cudaStream()) {
    // Ensure that we are waiting for the source data to be ready.
    DEFINE_OR_RETURN_ERROR(DeviceEventRef, srcEvent, srcCtx.enqueueEvent());
    RETURN_IF_ERROR(dstCtx.enqueueWaitForEvent(*srcEvent));

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
    DEFINE_OR_RETURN_ERROR(DeviceEventRef, dstEvent, dstCtx.enqueueEvent());
    RETURN_IF_ERROR(srcCtx.enqueueWaitForEvent(*dstEvent));
  } else {
    CUDA_CHECK_ERR(
        driver_.cuMemcpyDtoDAsync_v2(dstPtr, srcPtr, dstByteSize, stream_));
  }
  auto &registry = launchTraceRegistry();
  if (registry.isLaunchTraceEnabled(static_cast<void *>(stream_))) {
    KernelLaunchMetadata metadata(
        "cuMemcpyDtoD",
        hashMemcpyTrace(LaunchTraceKind::kMemcpyDtoD, dstByteSize),
        KernelLaunchMetadata::MemcpyKind::DtoD, dstByteSize);
    registry.appendLaunchTraceEntry(static_cast<void *>(stream_), metadata);
  }

  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess CUDADeviceContext::enqueueSetMemory(void *dst, size_t byteSize,
                                                   uint64_t val,
                                                   size_t valSize) {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::enqueueSetMemory",
                       TRACY_COLOR_PURPLE);

  CUDA_WITH_CTX_ERR(*this);
  if (valSize == 1) {
    CUDA_CHECK_ERR(driver_.cuMemsetD8Async(dst, static_cast<uint8_t>(val),
                                           byteSize, stream_));
  } else if (valSize == 2) {
    if ((byteSize & 1) != 0) {
      return Error("byteSize of buffer is misaligned");
    }
    size_t len = byteSize >> 1;
    CUDA_CHECK_ERR(driver_.cuMemsetD16Async(dst, static_cast<uint16_t>(val),
                                            len, stream_));
  } else if (valSize == 4) {
    if ((byteSize & 3) != 0) {
      return Error("byteSize of buffer is misaligned");
    }
    size_t len = byteSize >> 2;
    CUDA_CHECK_ERR(driver_.cuMemsetD32Async(dst, static_cast<uint32_t>(val),
                                            len, stream_));
  } else {
    if (valSize != 8) {
      return Error("unsupported valSize");
    }
    if ((byteSize & 7) != 0) {
      return Error("byteSize of buffer is misaligned");
    }
    uint32_t hiBits = static_cast<uint32_t>(val >> 32);
    uint32_t loBits = static_cast<uint32_t>(val & 0xfffffffful);
    if (hiBits == loBits) {
      size_t len = byteSize >> 2; // Divide by 4 as we are using 32-bit memset.
      CUDA_CHECK_ERR(driver_.cuMemsetD32Async(dst, loBits, len, stream_));
    } else {
      size_t len = byteSize >> 3;
      void *hiPtr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(dst) +
                                             sizeof(uint32_t));
      CUDA_CHECK_ERR(
          driver_.cuMemsetD2D32Async(dst, 8, loBits, 1, len, stream_));
      CUDA_CHECK_ERR(
          driver_.cuMemsetD2D32Async(hiPtr, 8, hiBits, 1, len, stream_));
    }
  }
  auto &registry = launchTraceRegistry();
  if (registry.isLaunchTraceEnabled(static_cast<void *>(stream_))) {
    KernelLaunchMetadata metadata("cuMemset",
                                  hashMemsetTrace(byteSize, val, valSize),
                                  byteSize, val, valSize);
    registry.appendLaunchTraceEntry(static_cast<void *>(stream_), metadata);
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
      RETURN_ERROR((("debugLevel of \"full\" requires optimizationLevel of "
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

ErrorOr<DeviceFunctionRef>
CUDADeviceContext::loadFunction(StringRef moduleName, StringRef functionName,
                                StringRef code, int32_t maxDynamicSharedBytes,
                                StringRef debugLevel,
                                int32_t optimizationLevel) {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::loadFunction", TRACY_COLOR_PURPLE);
  DeviceFunctionCacheEntryRef entry =
      device().findInFunctionCache(Device::FunctionCacheKey(
          moduleName.empty() ? code.str() : moduleName.str(), debugLevel.str(),
          optimizationLevel, maxDynamicSharedBytes));
  if (entry.getPointer() != nullptr) {
    CUDAModuleFunctionRef moduleFunc =
        CUDAModuleFunctionRef::take(CUDAModuleFunction::cast(entry.release()));
    return CUDADeviceFunctionRef::create(*this, std::move(moduleFunc));
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
  POLL_PENDING_EVENTS(*this);

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
      *this,
      CUDAModuleFunctionRef::take(CUDAModuleFunction::cast(entry.release())));
  return result;
}

ErrorOrSuccess CUDADeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args, uint32_t argCount,
    const size_t *argSizes) {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::enqueueFunctionExecDirect",
                       TRACY_COLOR_PURPLE);

  const auto &cudaFunc = CUDADeviceFunction::cast(func);

  CUDA_WITH_CTX_ERR(*this);
  return cudaFunc.cudaCallKernelDirect(
      gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
      reinterpret_cast<CUlaunchAttribute *>(attrs), numAttrs, args, stream_);
}

std::optional<intptr_t> CUDADeviceContext::getDLPackStream() const {
  ErrorOr<DeviceStreamRef> errStream = stream();
  if (errStream.isError())
    return std::nullopt;
  return reinterpret_cast<intptr_t>((*errStream)->ptr());
}

ErrorOr<DeviceStreamRef> CUDADeviceContext::stream() const {
  // This function is const only for the duration of its execution;
  // `this` is non-const downstream so cast in order to provide
  // mutable reference to the new stream object.
  return CUDADeviceStreamRef::create(const_cast<CUDADeviceContext &>(*this),
                                     reinterpret_cast<CUstream>(stream_),
                                     /*isOwner=*/false);
}

ErrorOrSuccess CUDADeviceContext::streamPriorityRange(int *leastPriority,
                                                      int *greatestPriority) {
  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(
      driver_.cuCtxGetStreamPriorityRange(leastPriority, greatestPriority));
  return success();
}

ErrorOr<DeviceGraphBuilderRef> CUDADeviceContext::createGraphBuilder() {
  return CUDADeviceGraphBuilder::create(*this);
}

ErrorOr<DeviceStreamRef> CUDADeviceContext::createStream(int priority) {
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
CUDADeviceContext::createExternalStream(void *externalStream) {
  return CUDADeviceStreamRef::create(*this,
                                     reinterpret_cast<CUstream>(externalStream),
                                     /*isOwner=*/false);
}

ErrorOrSuccess CUDADeviceContext::synchronize() {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::synchronize", TRACY_COLOR_PURPLE);

  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.cuStreamSynchronize(stream_));
  // Check for any errors and events after synchronization.
  RETURN_IF_PENDING_ERROR(*this);
  POLL_PENDING_EVENTS(*this);
  return success();
}

ErrorOr<DeviceEventRef> CUDADeviceContext::enqueueEvent() {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::enqueueEvent", TRACY_COLOR_PURPLE);

  CUDA_WITH_CTX_ERR(*this);
  const bool recordedDuringCapture = isStreamCapturing(driver_, stream_);

  DEFINE_OR_RETURN_ERROR(CUevent, event,
                         CUDADevice::cast(device()).eventCachePop(
                             /*useCache=*/!recordedDuringCapture));
  CUDA_CHECK_ERR(driver_.cuEventRecord(event, stream_));
  return CUDADeviceEventRef::create(CUDADevice::cast(this->device()), event,
                                    recordedDuringCapture);
}

ErrorOrSuccess CUDADeviceContext::enqueueEvent(const DeviceEvent &event) {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::enqueueEvent", TRACY_COLOR_PURPLE);

  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(
      driver_.cuEventRecord(CUDADeviceEvent::cast(event).cudaEvent(), stream_));
  return success();
}

ErrorOr<DeviceEventRef> CUDADeviceContext::eventCreate(bool enableTiming) {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::eventCreate", TRACY_COLOR_PURPLE);

  const bool recordedDuringCapture = isStreamCapturing(driver_, stream_);
  unsigned int flags =
      enableTiming
          ? static_cast<unsigned int>(CUevent_flags::CU_EVENT_DEFAULT)
          : static_cast<unsigned int>(CUevent_flags::CU_EVENT_DISABLE_TIMING);

  CUDA_WITH_CTX_ERR(*this);
  POLL_PENDING_EVENTS(*this);
  CUevent event;
  CUDA_CHECK_ERR(driver_.cuEventCreate(&event, flags));
  return CUDADeviceEventRef::create(
      CUDADevice::cast(this->device()), event,
      /*recordedDuringCapture=*/recordedDuringCapture,
      /*enableTiming=*/enableTiming);
}

ErrorOrSuccess CUDADeviceContext::enqueueWaitFor(DeviceEvent &event) {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::enqueueWaitFor", TRACY_COLOR_PURPLE);

  const auto &cudaEvent = CUDADeviceEvent::cast(event);
  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.cuStreamWaitEvent(stream_, cudaEvent.cudaEvent(), 0));
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess CUDADeviceContext::enqueueWaitFor(DeviceContext &ctx) {
  TRACY_ZONE_SCOPED_NC("CUDADeviceContext::enqueueWaitFor", TRACY_COLOR_PURPLE);

  DEFINE_OR_RETURN_ERROR(DeviceEventRef, event, ctx.enqueueEvent());
  RETURN_IF_ERROR(enqueueWaitFor(*event));
  return success();
}

ErrorOr<DeviceTimerRef> CUDADeviceContext::startTimer() {
  CUevent start;
  CUevent stop;

  CUDA_WITH_CTX_ERR(*this);
  POLL_PENDING_EVENTS(*this);
  CUDA_CHECK_ERR(driver_.cuEventCreate(&start, 0));
  CUDA_CHECK_ERR(driver_.cuEventCreate(&stop, 0));

  auto timer = CUDADeviceTimerRef::create(*this, start, stop);

  CUDA_CHECK_ERR(driver_.cuEventRecord(timer->cudaStartEvent(), stream_));

  return timer;
}

ErrorOr<std::chrono::nanoseconds>
CUDADeviceContext::stopTimer(const DeviceTimer &timer) {
  const auto &cudaTimer = CUDADeviceTimer::cast(timer);

  CUDA_WITH_CTX_ERR(*this);
  POLL_PENDING_EVENTS(*this);
  CUDA_CHECK_ERR(driver_.cuEventRecord(cudaTimer.cudaStopEvent(), stream_));
  CUDA_CHECK_ERR(driver_.cuEventSynchronize(cudaTimer.cudaStopEvent()));

  float millis;
  CUDA_CHECK_ERR(driver_.cuEventElapsedTime(&millis, cudaTimer.cudaStartEvent(),
                                            cudaTimer.cudaStopEvent()));

  constexpr double nanosPerMillisecond = 1000.0 * 1000.0;
  double nanos = static_cast<double>(millis) * nanosPerMillisecond;
  return std::chrono::nanoseconds(static_cast<int64_t>(nanos));
}

ErrorOr<Device::MemInfo> CUDADeviceContext::getDeviceMemoryInfo() const {
  CUDA_WITH_CTX(*this);
  // To get more accurate memory stats, ensure that at least all pending memory
  // allocations and frees have completed for this context's stream.
  CUDA_CHECK_ERR(driver_.cuStreamSynchronize(stream_));

  return device().getMemoryInfo();
}

ErrorOrSuccess CUDADeviceContext::occupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, DeviceFunction &func, int blockSize,
    size_t dynamicSharedMemSize) {
  auto &cudaFunc = CUDADeviceFunction::cast(func);
  CUDA_WITH_CTX_ERR(*this);
  return cudaFunc.occupancyMaxActiveBlocksPerMultiprocessor(
      numBlocks, blockSize, dynamicSharedMemSize);
}

bool CUDADeviceContext::isGraphCapturing() const {
  return isStreamCapturing(cudaDriver(), cudaStream());
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
    CUtensorMapFloatOOBfill oobFill) {
  CHECK_IS_CUDA_ERR(globalAddress.context().device());
  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.cuTensorMapEncodeTiled(
      tensorMap, tensorDataType, tensorRank, globalAddress.ptr(), globalDim,
      globalStrides, boxDim, elementStrides, interleave, swizzle, l2Promotion,
      oobFill));
  return success();
}

ErrorOrSuccess CUDADeviceContext::cudaTensorMapEncodeIm2col(
    void *tensorMap, CUtensorMapDataType tensorDataType, uint32_t tensorRank,
    const DeviceBuffer &globalAddress, const uint64_t *globalDim,
    const uint64_t *globalStrides, const int *pixelBoxLowerCorner,
    const int *pixelBoxUpperCorner, uint32_t channelsPerPixel,
    uint32_t pixelsPerColumn, const uint32_t *elementStrides,
    CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
  CHECK_IS_CUDA_ERR(globalAddress.context().device());
  CUDA_WITH_CTX_ERR(*this);
  CUDA_CHECK_ERR(driver_.cuTensorMapEncodeIm2col(
      tensorMap, tensorDataType, tensorRank, globalAddress.ptr(), globalDim,
      globalStrides, pixelBoxLowerCorner, pixelBoxUpperCorner, channelsPerPixel,
      pixelsPerColumn, elementStrides, interleave, swizzle, l2Promotion,
      oobFill));
  return success();
}

const CUDADeviceContext &CUDADeviceContext::cast(const DeviceContext &ctx) {
  return static_cast<const CUDADeviceContext &>(ctx);
}

CUDADeviceContext &CUDADeviceContext::cast(DeviceContext &ctx) {
  return static_cast<CUDADeviceContext &>(ctx);
}

const CUDADeviceContext *
CUDADeviceContext::verified_cast(const DeviceContext *ctx) {
  if (ctx->device().api() != CUDADevice::api) {
    return nullptr;
  }
  return static_cast<const CUDADeviceContext *>(ctx);
}

ErrorOr<std::string> CUDADevice::getFirstArchName() {
  DEFINE_OR_RETURN_ERROR(const NVML &, nvml, NVML::get());

  nvmlDevice_t device;
  if (nvml.nvmlDeviceGetHandleByIndex(0, &device) != NVML::SUCCESS)
    RETURN_ERROR("NVML: Unable to get device index");

  int major, minor;
  if (nvml.nvmlDeviceGetCudaComputeCapability(device, &major, &minor) !=
      NVML::SUCCESS)
    RETURN_ERROR("NVML: Unable to get compute capability");

  return cudaCCToSM(major, minor);
}
