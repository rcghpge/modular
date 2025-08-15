//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the HIP device context.
//
//===----------------------------------------------------------------------===//

#include "HIPDeviceContext.h"

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "AsyncRT/lib/DeviceContext/BufferCache.h"
#include "HIPDriver.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

// Helpers for logging.
#define TWINE(val) (llvm::Twine(val))

#define HIP_CHECK_ERR(hip_call)                                                \
  do {                                                                         \
    HIPDriver::Result hipResult = hip_call;                                    \
    if (hipResult != HIPDriver::SUCCESS) {                                     \
      return formatHipError(__FILE__, __LINE__, __FUNCTION__, hipResult);      \
    }                                                                          \
  } while (0)

#define HIP_CHECK(hip_call)                                                    \
  do {                                                                         \
    HIPDriver::Result hipResult = hip_call;                                    \
    if (hipResult != HIPDriver::SUCCESS) {                                     \
      M::Error err =                                                           \
          formatHipError(__FILE__, __LINE__, __FUNCTION__, hipResult);         \
      /* We are not in a context where we can return an Error here. So we   */ \
      /* print the message and abort().                                     */ \
      std::cerr << err << "\n";                                                \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define RAISE_ERROR(msg)                                                       \
  do {                                                                         \
    return DeviceContext::formatError(__FILE__, __LINE__, __FUNCTION__, msg);  \
  } while (0)

#define CHECK_IS_HIP_ERR(dev)                                                  \
  do {                                                                         \
    if ((dev).api() != Device::hipAPI) {                                       \
      RAISE_ERROR("Not a HIP device.");                                        \
    }                                                                          \
  } while (0)

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

#define HIP_WITH_CTX_ERR(ctx)                                                  \
  ErrorOrSuccess __pendingErr = (ctx).checkPendingWork();                      \
  if (__pendingErr.isError()) {                                                \
    return __pendingErr.takeError();                                           \
  }                                                                            \
  HIPDeviceScope __hipWithContext((ctx).hipDevice(), (ctx).hipDriver());       \
  HIP_CHECK_ERR(__hipWithContext.status());

#define HIP_WITH_CTX(ctx)                                                      \
  HIPDeviceScope __hipWithContext((ctx).hipDevice(), (ctx).hipDriver());       \
  HIP_CHECK(__hipWithContext.status());

#define HIP_WITH_DEV_ERR(dev)                                                  \
  HIPDeviceScope __hipWithContext((dev).hipDevice(), (dev).hipDriver());       \
  HIP_CHECK_ERR(__hipWithContext.status());

#define HIP_WITH_DEV(dev)                                                      \
  HIPDeviceScope __hipWithContext((dev).hipDevice(), (dev).hipDriver());       \
  HIP_CHECK(__hipWithContext.status());

#define MAYBE_SYNC(ctx)                                                        \
  do {                                                                         \
    const HIPDeviceContext &hip_ctx = (ctx);                                   \
    if (hip_ctx.syncMode()) {                                                  \
      HIP_CHECK_ERR(                                                           \
          hip_ctx.hipDriver().StreamSynchronize(hip_ctx.hipStream()));         \
    }                                                                          \
  } while (0)

namespace M::AsyncRT {

// Avoid using nullptr for the "device context pointer". We add/subtract the
// following constant when handling the hipDevice_t as a pointer.
constexpr intptr_t devicePtrOffset =
    0x51413; // First 5 digits of pi in reverse.

static void *hipDeviceToPtr(hipDevice_t id) {
  return reinterpret_cast<void *>(static_cast<intptr_t>(id) + devicePtrOffset);
}

static hipDevice_t ptrToHipDevice(void *ptr) {
  return static_cast<hipDevice_t>(reinterpret_cast<intptr_t>(ptr) -
                                  devicePtrOffset);
}

// The helper class HIPDeviceScope is used to set and restore the hipDevice_t
// around interactions with the HIP driver.
class HIPDeviceScope {
public:
  HIPDeviceScope(hipDevice_t device, const HIPDriver &driver)
      : driver_(driver), device_(device), prevDevice_(-1),
        status_(HIPDriver::SUCCESS) {
    driver_.GetDevice(&prevDevice_);
    if (prevDevice_ == device_)
      return;
    status_ = driver_.SetDevice(device_);
  }

  ~HIPDeviceScope() {
    if ((prevDevice_ == device_) || (status_ != HIPDriver::SUCCESS))
      return;

    // Only pop the context if push was necessary and successful.
    hipDevice_t device = -1;
    driver_.GetDevice(&device);
    driver_.SetDevice(prevDevice_);
    assert(device == device_);
  }

  HIPDriver::Result status() { return status_; }

private:
  const HIPDriver &driver_;
  const hipDevice_t device_;
  hipDevice_t prevDevice_;
  HIPDriver::Result status_;
};

static Error formatHipError(const char *file, int line, const char *func,
                            HIPDriver::Result result) {
  ErrorOr<const HIPDriver &> errDriver = HIPDriver::get();
  const char *errName = nullptr;
  const char *errStr = nullptr;

  if (!errDriver) {
    // Ignore errors and use nullptr of char* instead.
    errDriver->GetErrorName(result, &errName);
    errDriver->GetErrorString(result, &errStr);
  }

  if ((errName != nullptr) && (errStr != nullptr)) {
    return DeviceContext::formatError(file, line, func,
                                      llvm::Twine("HIP call failed: ") +
                                          errName + " (" + errStr + ")");
  } else if ((errName != nullptr) || (errStr != nullptr)) {
    return DeviceContext::formatError(
        file, line, func,
        llvm::Twine("HIP call failed: ") +
            (errName != nullptr ? errName : errStr));
  } else {
    return DeviceContext::formatError(file, line, func,
                                      llvm::Twine("HIP call failed: ") +
                                          llvm::Twine(result));
  }
}

// ===== HIPDevice ===== //
// Plan:
// DeviceContext::Attribute will reuse the CUDA enum mapping
// CUDADeviceContext::getAttribute can thus use the enum values directly
// HIPDeviceContext will map the DeviceContext attributes to their HIP
// equivalents
static ErrorOr<hipDeviceAttribute_t>
gethipDeviceAttribute(Device::Attribute attr) {
#define ENTRY(X, Y)                                                            \
  case Device::Attribute::X:                                                   \
    return hipDeviceAttribute_t::hipDeviceAttribute##Y

#define RAISE_ENTRY(X)                                                         \
  case Device::Attribute::X:                                                   \
    RAISE_ERROR("Attribute \"" #X "\" not supported for AMD GPU.")

  switch (attr) {
    ENTRY(MAX_THREADS_PER_BLOCK, MaxThreadsPerBlock);
    ENTRY(MAX_BLOCK_DIM_X, MaxBlockDimX);
    ENTRY(MAX_BLOCK_DIM_Y, MaxBlockDimY);
    ENTRY(MAX_BLOCK_DIM_Z, MaxBlockDimZ);
    ENTRY(MAX_GRID_DIM_X, MaxGridDimX);
    ENTRY(MAX_GRID_DIM_Y, MaxGridDimY);
    ENTRY(MAX_GRID_DIM_Z, MaxGridDimZ);
    ENTRY(MAX_SHARED_MEMORY_PER_BLOCK, MaxSharedMemoryPerBlock);
    ENTRY(WARP_SIZE, WarpSize);
    ENTRY(MAX_REGISTERS_PER_BLOCK, MaxRegistersPerBlock);
    ENTRY(CLOCK_RATE, ClockRate);
    ENTRY(MULTIPROCESSOR_COUNT, MultiprocessorCount);
    ENTRY(MAX_THREADS_PER_MULTIPROCESSOR, MaxThreadsPerMultiProcessor);
    ENTRY(COMPUTE_CAPABILITY_MAJOR, ComputeCapabilityMajor);
    ENTRY(COMPUTE_CAPABILITY_MINOR, ComputeCapabilityMinor);
    ENTRY(MAX_REGISTERS_PER_MULTIPROCESSOR, MaxRegistersPerMultiprocessor);
    RAISE_ENTRY(MAX_ACCESS_POLICY_WINDOW_SIZE);
    // No `default:` to ensure that all possible enum values are handled.
  }
#undef RAISE_ENTRY
#undef ENTRY
  // If a unsupported value is passed as a numerical value, fail with a generic
  // message here. This can happen if for example we create an unsupported value
  // from within Mojo.
  RAISE_ERROR("Attribute not defined for AMD GPU");
}

HIPDevice::HIPDevice(const HIPDriver &driver, int id, hipDevice_t device,
                     std::string name, MemInfo memInfo, size_t allocAlignment)
    : Device(api, name, id, memInfo, allocAlignment), driver_(driver),
      device_(device) {}

HIPDevice::~HIPDevice() {
  // Destroy the function map before destroying other device structures.
  clearFunctionCache();
  bufferCache().clear();
}

const HIPDriver &HIPDevice::hipDriver() const { return driver_; }

hipDevice_t HIPDevice::hipDevice() const { return device_; }

ErrorOr<int> HIPDevice::getApiVersion() const {
  int driverVersion = 0;
  HIP_CHECK_ERR(driver_.DriverGetVersion(&driverVersion));
  return driverVersion;
}

ErrorOr<DriverVersion &> HIPDevice::getDriverVersion() const {
  static DriverVersion version("0.0.0");
  return version;
}

ErrorOrSuccess HIPDevice::isCompatible() const {
#ifndef MODULAR_LINUX
  return Error("MAX with GPUs is only supported on Linux based systems.");
#endif

  ErrorOr<int> compute = computeCapability();
  if (compute.isError())
    return compute.takeError();
  if (*compute < MIN_COMPUTE_CAPABILITY) {
    RAISE_ERROR("MAX only supports HIP architectures " +
                TWINE(MIN_COMPUTE_CAPABILITY) + " or higher");
  }

  int driverVersionCoded = 0;
  HIP_CHECK_ERR(driver_.DriverGetVersion(&driverVersionCoded));

  // Deconstruct the driverVersionCoded into major, minor and patch.
  constexpr int driverVersionMinorFactor = 100000;
  constexpr int driverVersionMajorFactor = 10000000;

  int driverVersionPatch = driverVersionCoded % driverVersionMinorFactor;
  int driverVersionMinor =
      (driverVersionCoded / driverVersionMinorFactor) %
      (driverVersionMajorFactor / driverVersionMinorFactor);
  int driverVersionMajor = driverVersionCoded / driverVersionMajorFactor;

  if (driverVersionMajor > MIN_HIP_DRIVER_VERSION_MAJOR) {
    return success();
  } else if (driverVersionMajor == MIN_HIP_DRIVER_VERSION_MAJOR) {
    if (driverVersionMinor > MIN_HIP_DRIVER_VERSION_MINOR) {
      return success();
    } else if (driverVersionMinor == MIN_HIP_DRIVER_VERSION_MINOR) {
      if (driverVersionPatch >= MIN_HIP_DRIVER_VERSION_PATCH) {
        return success();
      }
    }
  }

  RAISE_ERROR(
      "MAX doesn't support your current AMD GPU driver. MAX requires a "
      "minimum driver version of " +
      TWINE(MIN_HIP_DRIVER_VERSION_MAJOR) + "." +
      TWINE(MIN_HIP_DRIVER_VERSION_MINOR) + "." +
      TWINE(MIN_HIP_DRIVER_VERSION_PATCH) +
      ". The driver version MAX detected is: " + TWINE(driverVersionMajor) +
      "." + TWINE(driverVersionMinor) + "." + TWINE(driverVersionPatch) +
      ".\nFor information about how to update your driver, see "
      "https://modul.ar/gpu-requirements.");
}

ErrorOr<int> HIPDevice::computeCapability() const {
  int computeMajor = 0, computeMinor = 0;
  HIP_CHECK_ERR(driver_.DeviceGetAttribute(
      &computeMajor,
      hipDeviceAttribute_t::hipDeviceAttributeComputeCapabilityMajor,
      hipDevice()));
  HIP_CHECK_ERR(driver_.DeviceGetAttribute(
      &computeMinor,
      hipDeviceAttribute_t::hipDeviceAttributeComputeCapabilityMinor,
      hipDevice()));
  return 10 * computeMajor + computeMinor;
}

ErrorOr<bool> HIPDevice::canAccess(const Device &peer) const {
  if (id() == peer.id()) {
    // Always disable self access (same behaviour as hipDeviceCanAccessPeer).
    return false;
  }

  const HIPDevice *peerDevice = HIPDevice::verified_cast(&peer);
  if (peerDevice == nullptr) {
    RAISE_ERROR("Peer is not a HIP device");
  }

  int canAccess = 0;
  HIP_CHECK_ERR(driver_.DeviceCanAccessPeer(&canAccess, hipDevice(),
                                            peerDevice->hipDevice()));
  return canAccess == 1;
}

ErrorOr<int> HIPDevice::getAttribute(Attribute attr) const {
  HIP_WITH_DEV_ERR(*this);
  ErrorOr<hipDeviceAttribute_t> errorOrAttr = gethipDeviceAttribute(attr);
  if (errorOrAttr.isError())
    return errorOrAttr.takeError();
  int value = 0;
  HIP_CHECK_ERR(driver_.DeviceGetAttribute(&value, *errorOrAttr, hipDevice()));
  return value;
}

ErrorOr<std::string> HIPDevice::getArchName() const {
  // We do need the full AMD GPU architecture name in order to compie for the
  // right LLVM target. Parsing via KFD, as used by amd-smi and rocm-smi seems
  // to be one of the two reliable ways to get the full architecture name. The
  // other way, which we use here, is through the HSA API and some PCIe device
  // matching with the HIP API. The HIP API directly does provide the
  // architecture name in the `hipDeviceProp_t::gcnArchName` field, but it is
  // not always set. Likewise, `hipDeviceAttributeGcnArchName` is now
  // deprecated. Major & minor versions are available via
  // `hipDeviceAttributeComputeCapabilityMajor` and
  // `hipDeviceAttributeComputeCapabilityMinor`, but lack the last digit.

  // Initialize HSA runtime if available.
  if (driver_.HsaInit() != HSA_STATUS_SUCCESS)
    return Error("Failed to initialize HSA runtime");

  // Get HIP device PCIe info via device attributes
  int pciBusId = 0, pciDeviceId = 0, pciDomainId = 0;
  HIP_CHECK_ERR(driver_.DeviceGetAttribute(
      &pciBusId, hipDeviceAttribute_t::hipDeviceAttributePciBusId,
      hipDevice()));
  HIP_CHECK_ERR(driver_.DeviceGetAttribute(
      &pciDeviceId, hipDeviceAttribute_t::hipDeviceAttributePciDeviceId,
      hipDevice()));
  HIP_CHECK_ERR(driver_.DeviceGetAttribute(
      &pciDomainId, hipDeviceAttribute_t::hipDeviceAttributePciDomainID,
      hipDevice()));

  // Data structure to pass to HSA agent and ISA iteration callbacks.
  struct SearchData {
    const HIPDriver *driver;
    uint32_t targetPciBusID;
    uint32_t targetPciDeviceID;
    uint32_t targetPciDomainID;
    std::string archName;
  };

  SearchData search = {&driver_, static_cast<uint32_t>(pciBusId),
                       static_cast<uint32_t>(pciDeviceId),
                       static_cast<uint32_t>(pciDomainId), ""};

  // Callback function for HSA agent iteration.
  auto agentCallback = [](hsaAgent_t agent, void *data) -> hsaStatus_t {
    auto *search = static_cast<SearchData *>(data);

    // Match by PCIe address of HIP and HSA device.
    uint32_t hsaDomain = 0;
    uint32_t hsaBdfId = 0;

    // Get PCIe domain. On failure, continue to next agent.
    if (search->driver->HsaAgentGetInfo(
            agent, hsaAgentInfo_t::HSA_AMD_AGENT_INFO_DOMAIN, &hsaDomain) !=
            HSA_STATUS_SUCCESS ||
        search->driver->HsaAgentGetInfo(
            agent, hsaAgentInfo_t::HSA_AMD_AGENT_INFO_BDFID, &hsaBdfId) !=
            HSA_STATUS_SUCCESS)
      return HSA_STATUS_SUCCESS;

    // BDFID is typically encoded as: (bus << 8) | (device << 3) | function
    uint32_t hsaBus = (hsaBdfId >> 8) & 0xFF;
    uint32_t hsaDevice = (hsaBdfId >> 3) & 0x1F;
    // uint32_t hsaFunction = hsaBdfId & 0x7;

    // Compare with HIP device PCIe info. On failure, continue to next agent.
    if (hsaDomain != search->targetPciDomainID ||
        hsaBus != search->targetPciBusID ||
        hsaDevice != search->targetPciDeviceID)
      return HSA_STATUS_SUCCESS;

    // Callback to iterate through ISAs for this agent.
    auto isaCallback = [](hsaIsa_t isa, void *data) -> hsaStatus_t {
      auto *search = static_cast<SearchData *>(data);

      // First, get the length of the ISA name.
      uint32_t nameLen = 0;
      if (search->driver->HsaIsaGetInfo(isa,
                                        hsaIsaInfo_t::HSA_ISA_INFO_NAME_LENGTH,
                                        &nameLen) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;

      // Continue if the ISA name length is invalid.
      if (nameLen == 0 || nameLen > 256)
        return HSA_STATUS_SUCCESS;

      // Allocate memory for the ISA name
      std::vector<char> isaName(nameLen);

      // Get the actual ISA name. Continue to next ISA if it fails.
      if (search->driver->HsaIsaGetInfo(isa, hsaIsaInfo_t::HSA_ISA_INFO_NAME,
                                        isaName.data()) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;

      // Check if this ISA name contains "gfx".
      std::string isaStr(isaName.data(), nameLen > 0
                                             ? nameLen - 1
                                             : 0); // Remove null terminator.
      size_t gfxPos = isaStr.find("gfx");
      if (gfxPos != std::string::npos) {
        // Find the end of the digits after "gfx".
        size_t start = gfxPos;
        // Start after "gfx".
        size_t end = start + 3;

        // Continue until we find a non-digit character.
        while (end < isaStr.length() && std::isdigit(isaStr[end]))
          ++end;

        // Extract the gfx* part.
        std::string gfxArch = isaStr.substr(start, end - start);
        search->archName = gfxArch;
        // Stop ISA iteration.
        return HSA_STATUS_INFO_BREAK;
      }

      // Continue ISA iteration.
      return HSA_STATUS_SUCCESS;
    };

    // Iterate through ISAs for this agent. Continue agent iteration if it
    // fails.
    if (search->driver->HsaAgentIterateIsas(agent, isaCallback, search) !=
        HSA_STATUS_SUCCESS)
      return HSA_STATUS_SUCCESS;

    // If we found an architecture, stop agent iteration.
    if (search->archName.size())
      return HSA_STATUS_INFO_BREAK;

    // Continue agent iteration.
    return HSA_STATUS_SUCCESS;
  };

  // Iterate through all HSA agents to find our device.
  if (driver_.HsaIterateAgents(agentCallback, &search) != HSA_STATUS_SUCCESS) {
    driver_.HsaShutDown();
    return Error("Failed to iterate HSA agents");
  }

  // Clean up HSA runtime.
  driver_.HsaShutDown();

  if (search.archName.empty())
    return Error("Could not determine architecture name for HIP device");

  return search.archName;
}

ErrorOr<DeviceContextRef> HIPDevice::createContext() const {
  // Make sure that we associate the stream with the correct device.
  HIP_WITH_DEV_ERR(*this);

  hipStream_t stream;
  HIP_CHECK_ERR(this->hipDriver().StreamCreate(&stream, 0));

  return HIPDeviceContextRef::create(*this, stream);
}

ErrorOr<DeviceStreamRef>
HIPDevice::createExternalStream(void *externalStream) const {
  HIP_WITH_DEV_ERR(*this);
  return HIPDeviceStreamRef::create(
      *this, reinterpret_cast<hipStream_t>(externalStream));
}

ErrorOr<int> HIPDevice::numberOfGPUs() {
  static ErrorOr<int> errOrCount = []() -> ErrorOr<int> {
    ErrorOr<const HIPDriver &> errDriver = HIPDriver::get();
    if (errDriver.isError())
      return errDriver.takeError();

    int result;
    HIP_CHECK_ERR((*errDriver).DeviceGetCount(&result));
    return result;
  }();

  if (errOrCount.isError()) {
    return errOrCount.copy();
  }
  return *errOrCount;
}

int HIPDevice::deviceCount() const {
  ErrorOr<int> errOrCount = numberOfGPUs();
  if (errOrCount.isError()) {
    // Do not propagate an error.
    return 0;
  }
  return *errOrCount;
}

ErrorOrSuccess HIPDevice::chunkAlloc(void **ptr, size_t byteSize,
                                     bool isHost) const {
  assert(byteSize != 0);
  HIP_WITH_DEV_ERR(*this);
  if (isHost) {
    HIP_CHECK_ERR(driver_.MemAllocHost(ptr, byteSize));
  } else {
    HIP_CHECK_ERR(driver_.MemAlloc(ptr, byteSize));
  }
  return success();
}

ErrorOrSuccess HIPDevice::chunkFree(void *ptr, bool isHost) const {
  HIP_WITH_DEV_ERR(*this);
  if (isHost) {
    HIP_CHECK_ERR(driver_.MemFreeHost(ptr));
  } else {
    HIP_CHECK_ERR(driver_.MemFree(ptr));
  }
  return success();
}

ErrorOr<Device::MemInfo> HIPDevice::getMemoryInfo() const {
  Device::MemInfo memInfo;
  HIP_WITH_DEV_ERR(*this);
  HIP_CHECK_ERR(driver_.GetMemInfo(&memInfo.free, &memInfo.total));
  return memInfo;
}

ErrorOr<void *> HIPDevice::getCurrentDriverContext() const {
  hipDevice_t curDev = 0;
  HIP_CHECK_ERR(driver_.GetDevice(&curDev));
  return hipDeviceToPtr(curDev);
}
ErrorOrSuccess HIPDevice::setCurrentDriverContext(void *ctx) const {
  HIP_CHECK_ERR(driver_.SetDevice(ptrToHipDevice(ctx)));
  return success();
}

const HIPDevice &HIPDevice::cast(const Device &device) {
  return static_cast<const HIPDevice &>(device);
}

const HIPDevice *HIPDevice::verified_cast(const Device *device) {
  if (device->api() != HIPDevice::api) {
    return nullptr;
  }
  return static_cast<const HIPDevice *>(device);
}

ErrorOr<const HIPDevice &> HIPDevice::get(int id) {
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
  HIPDevice *device = (*devices)[id];
  if (device == nullptr) {
    ErrorOr<const HIPDriver &> errDriver = HIPDriver::get();
    if (errDriver) {
      RAISE_ERROR(llvm::Twine("HIP Driver initialization failed: ") +
                  errDriver.getError());
    }
    const HIPDriver &driver = *errDriver;
    char name[256];
    int alignment;

    hipDevice_t hipDevice = static_cast<hipDevice_t>(id);
    HIP_CHECK_ERR(driver.DeviceGetName(name, sizeof(name), hipDevice));
    HIP_CHECK_ERR(driver.DeviceGetAttribute(
        &alignment, hipDeviceAttribute_t::hipDeviceAttributeTextureAlignment,
        hipDevice));

    HIPDeviceScope withDev(hipDevice, driver);
    Device::MemInfo memInfo;
    HIP_CHECK_ERR(driver.GetMemInfo(&memInfo.free, &memInfo.total));

    device = new HIPDevice(driver, id, hipDevice, name, memInfo, alignment);

    // Once the Device has been created remember it in the devices list.
    (*devices)[id] = device;
  }
  return *device;
}

// ===== HIPDeviceEvent ===== //
HIPDeviceEvent::HIPDeviceEvent(const HIPDevice &device, hipEvent_t event)
    : DeviceEvent(device), event_(event) {}

HIPDeviceEvent::~HIPDeviceEvent() {
  const auto &hipDevice = HIPDevice::cast(device());

  HIP_WITH_DEV(hipDevice);
  HIP_CHECK(hipDevice.hipDriver().EventDestroy(event_));
}

ErrorOrSuccess HIPDeviceEvent::synchronize() const {
  const auto &hipDevice = HIPDevice::cast(device());
  HIP_WITH_DEV_ERR(hipDevice);
  HIP_CHECK_ERR(hipDevice.hipDriver().EventSynchronize(event_));
  return success();
}

ErrorOr<bool> HIPDeviceEvent::isReady() const {
  const auto &hipDevice = HIPDevice::cast(device());
  HIP_WITH_DEV_ERR(hipDevice);
  HIPDriver::Result res = hipDevice.hipDriver().EventQuery(event_);
  if (res == HIPDriver::SUCCESS)
    return true;
  else if (res == HIPDriver::NOT_READY) {
    return false;
  }
  HIP_CHECK_ERR(res); // All other errors get raised here.
  // Not reachable because there is an error status in res, but compiler
  // complains if not returning a value.
  llvm_unreachable("Not Reachable");
  return Error("Not Reachable.");
}

hipEvent_t HIPDeviceEvent::hipEvent() const { return event_; }

const HIPDeviceEvent &HIPDeviceEvent::cast(const DeviceEvent &event) {
  return static_cast<const HIPDeviceEvent &>(event);
}

// ===== HIPDeviceStream ===== //
HIPDeviceStream::HIPDeviceStream(const HIPDevice &device, hipStream_t stream)
    : DeviceStream(device), stream_(stream) {}

ErrorOrSuccess HIPDeviceStream::synchronize() const {
  const auto &hipDevice = HIPDevice::cast(device());

  HIP_WITH_DEV_ERR(hipDevice);
  HIP_CHECK_ERR(hipDevice.hipDriver().StreamSynchronize(stream_));
  return success();
}

ErrorOr<DeviceEventRef> HIPDeviceStream::enqueueEvent() const {
  const auto &hipDevice = HIPDevice::cast(device());

  HIP_WITH_DEV_ERR(hipDevice);
  hipEvent_t event;
  HIP_CHECK_ERR(hipDevice.hipDriver().EventCreate(&event, 0));
  HIP_CHECK_ERR(hipDevice.hipDriver().EventRecord(event, stream_));
  return HIPDeviceEventRef::create(HIPDevice::cast(this->device()), event);
}

ErrorOrSuccess
HIPDeviceStream::enqueueWaitForEvent(const DeviceEvent &event) const {
  const auto &hipEvent = HIPDeviceEvent::cast(event);
  const auto &hipDevice = HIPDevice::cast(device());

  HIP_WITH_DEV_ERR(hipDevice);
  HIP_CHECK_ERR(
      hipDevice.hipDriver().StreamWaitEvent(stream_, hipEvent.hipEvent(), 0));
  return success();
}

ErrorOrSuccess HIPDeviceStream::eventRecord(const DeviceEvent &event) const {
  const auto &hipEvent = HIPDeviceEvent::cast(event);
  const auto &hipDevice = HIPDevice::cast(device());
  HIP_WITH_DEV_ERR(hipDevice);
  HIP_CHECK_ERR(
      hipDevice.hipDriver().EventRecord(hipEvent.hipEvent(), stream_));
  return success();
}

ErrorOrSuccess HIPDeviceStream::waitEvent(const DeviceEvent &event,
                                          unsigned int flags) const {
  const auto &hipEvent = HIPDeviceEvent::cast(event);
  const auto &hipDevice = HIPDevice::cast(device());
  HIP_WITH_DEV_ERR(hipDevice);
  HIP_CHECK_ERR(hipDevice.hipDriver().StreamWaitEvent(
      stream_, hipEvent.hipEvent(), flags));
  return success();
}

ErrorOrSuccess HIPDeviceStream::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args) const {
  const auto &hipFunc = HIPDeviceFunction::cast(func);
  const auto &hipDevice = HIPDevice::cast(device());
  HIP_WITH_DEV_ERR(hipDevice);
  if (numAttrs > 0)
    RAISE_ERROR("Function launch attributes not supported;");
  return hipFunc.hipCallKernelDirect(gridX, gridY, gridZ, blockX, blockY,
                                     blockZ, sharedMemBytes, args, stream_);
}

void *HIPDeviceStream::ptr() const { return hipStream(); }

hipStream_t HIPDeviceStream::hipStream() const { return stream_; }

const HIPDeviceStream *
HIPDeviceStream::verified_cast(const DeviceStream *stream) {
  if (stream->device().api() != HIPDevice::api) {
    return nullptr;
  }
  return static_cast<const HIPDeviceStream *>(stream);
}

// ===== HIPDeviceFunctionArguments ===== //
HIPDeviceFunctionArguments::HIPDeviceFunctionArguments(
    const HIPDeviceFunction &fn)
    : DeviceFunction::Arguments(fn) {}

void HIPDeviceFunctionArguments::pushArg(int index, const uint64_t &val) {
  argValues_.push_back(reinterpret_cast<void *>(val));
}

void HIPDeviceFunctionArguments::pushArg(int index, const DeviceBuffer &val) {
  // Ensure buffer is preserved by increasing the reference count through
  // copy().
  assert(val.context().device().api() == Device::hipAPI);
  refs_.emplace_back(DeviceBufferRef::copy(&val));
  argValues_.push_back(val.ptr());
}

std::vector<void *> HIPDeviceFunctionArguments::hipArgs() const {
  std::vector<void *> launchArgs;
  launchArgs.reserve(argValues_.size());
  for (size_t i = 0; i < argValues_.size(); i++) {
    launchArgs.push_back(const_cast<void **>(&argValues_[i]));
  }
  return launchArgs;
}

const HIPDeviceFunctionArguments &
HIPDeviceFunctionArguments::cast(const DeviceFunction::Arguments &args) {
  return static_cast<const HIPDeviceFunctionArguments &>(args);
}

// ===== HIPModuleFunction ===== //
HIPModuleFunction::HIPModuleFunction(const std::string name,
                                     const HIPDriver &driver,
                                     hipModule_t module, hipFunction_t function)
    : DeviceFunctionCacheEntry(std::move(name)), driver_(driver),
      module_(module), function_(function) {}

HIPModuleFunction::~HIPModuleFunction() {
  HIP_CHECK(hipDriver().ModuleUnload(module_));
}

const HIPDriver &HIPModuleFunction::hipDriver() const { return driver_; }
hipModule_t HIPModuleFunction::hipModule() const { return module_; }
hipFunction_t HIPModuleFunction::hipFunction() const { return function_; }

const HIPModuleFunction *
HIPModuleFunction::cast(const DeviceFunctionCacheEntry *entry) {
  return static_cast<const HIPModuleFunction *>(entry);
}

// ===== HIPDeviceFunction ===== //
HIPDeviceFunction::HIPDeviceFunction(
    const HIPDeviceContext &context,
    const DeviceFunction::SignatureBuilder &signature,
    HIPModuleFunctionRef modFunc)
    : DeviceFunction(context, signature), moduleFunction_(std::move(modFunc)) {}

HIPDeviceFunction::~HIPDeviceFunction() {}

DeviceFunction::Arguments *
HIPDeviceFunction::allocateArguments(const DeviceFunction &fn) const {
  return new HIPDeviceFunctionArguments(HIPDeviceFunction::cast(fn));
}

ErrorOrSuccess HIPDeviceFunction::hipCallKernel(
    uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
    uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    const HIPDeviceFunctionArguments &args, hipStream_t stream) const {
  HIP_CHECK_ERR(moduleFunction_->hipDriver().ModuleLaunchKernel(
      moduleFunction_->hipFunction(), gridX, gridY, gridZ, blockX, blockY,
      blockZ, sharedMemBytes, stream, args.hipArgs().data(), nullptr));
  return success();
}

ErrorOrSuccess HIPDeviceFunction::hipCallKernelDirect(
    uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
    uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes, void **args,
    hipStream_t stream) const {
  HIP_CHECK_ERR(moduleFunction_->hipDriver().ModuleLaunchKernel(
      moduleFunction_->hipFunction(), gridX, gridY, gridZ, blockX, blockY,
      blockZ, sharedMemBytes, stream, args, nullptr));
  return success();
}

const std::string &HIPDeviceFunction::name() const {
  return moduleFunction_->name();
}

ErrorOrSuccess HIPDeviceFunction::copyToNamedConstant(const char *name,
                                                      const void *data,
                                                      size_t byteSize) const {
  hipDeviceptr_t devicePtr;
  size_t deviceSize;
  const auto &hipContext = HIPDeviceContext::cast(context());

  HIP_WITH_CTX_ERR(hipContext);
  HIP_CHECK_ERR(hipContext.hipDriver().ModuleGetGlobal(
      &devicePtr, &deviceSize, moduleFunction_->hipModule(), name));

  if (deviceSize < byteSize) {
    RAISE_ERROR("byteSize too big");
  }

  HIP_CHECK_ERR(hipContext.hipDriver().MemcpyHtoDAsync(
      devicePtr, data, byteSize, hipContext.hipStream()));
  MAYBE_SYNC(HIPDeviceContext::cast(context()));
  return success();
}

ErrorOr<int32_t> HIPDeviceFunction::getAttribute(int32_t code) const {
  int32_t result;
  const auto &hipContext = HIPDeviceContext::cast(context());

  HIP_WITH_CTX_ERR(hipContext);
  HIP_CHECK_ERR(hipContext.hipDriver().FuncGetAttribute(
      &result, static_cast<hipFunction_attribute>(code),
      moduleFunction_->hipFunction()));
  return result;
}

ErrorOrSuccess HIPDeviceFunction::occupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, int blockSize, size_t dynamicSharedMemSize) const {
  const auto &hipContext = HIPDeviceContext::cast(context());

  HIP_WITH_CTX_ERR(hipContext);
  HIP_CHECK_ERR(
      hipContext.hipDriver().ModuleOccupancyMaxActiveBlocksPerMultiprocessor(
          numBlocks, moduleFunction_->hipFunction(), blockSize,
          dynamicSharedMemSize));

  return success();
}

const HIPDeviceFunction &HIPDeviceFunction::cast(const DeviceFunction &func) {
  return static_cast<const HIPDeviceFunction &>(func);
}

// ===== HIPDeviceTimer ===== //
HIPDeviceTimer::HIPDeviceTimer(const HIPDeviceContext &context,
                               hipEvent_t start, hipEvent_t stop)
    : DeviceTimer(context), start_(start), stop_(stop) {}

HIPDeviceTimer::~HIPDeviceTimer() {
  const auto &hipContext = HIPDeviceContext::cast(context());

  HIP_WITH_CTX(hipContext);
  HIP_CHECK(hipContext.hipDriver().EventDestroy(start_));
  HIP_CHECK(hipContext.hipDriver().EventDestroy(stop_));
}

hipEvent_t HIPDeviceTimer::hipStartEvent() const { return start_; }
hipEvent_t HIPDeviceTimer::hipStopEvent() const { return stop_; }

const HIPDeviceTimer &HIPDeviceTimer::cast(const DeviceTimer &timer) {
  return static_cast<const HIPDeviceTimer &>(timer);
}

// ===== HIPDeviceContext ===== //
HIPDeviceContext::HIPDeviceContext(const HIPDevice &device, hipStream_t stream)
    : DeviceContext(device), driver_(device.hipDriver()),
      device_(device.hipDevice()), stream_(stream) {}

HIPDeviceContext::~HIPDeviceContext() {
  HIP_WITH_CTX(*this);
  HIP_CHECK(driver_.StreamSynchronize(stream_));
  HIP_CHECK(driver_.StreamDestroy(stream_));
}

ErrorOrSuccess HIPDeviceContext::memAlloc(void **ptr, size_t byteSize,
                                          Device::AllocMode mode) const {
  assert(byteSize != 0);
  HIP_WITH_CTX_ERR(*this);
  switch (mode) {
  case Device::AllocMode::ASYNC: {
    HIP_CHECK_ERR(driver_.MemAllocAsync(ptr, byteSize, stream_));
    MAYBE_SYNC(*this);
    break;
  }
  case Device::AllocMode::HOST: {
    HIP_CHECK_ERR(driver_.MemAllocHost(ptr, byteSize));
    break;
  }
  case Device::AllocMode::SYNC: {
    HIP_CHECK_ERR(driver_.MemAlloc(ptr, byteSize));
    break;
  }
  }
  return success();
}

ErrorOrSuccess HIPDeviceContext::memFree(void *ptr, bool isHost) const {
  HIP_WITH_CTX_ERR(*this);
  if (isHost) {
    HIP_CHECK_ERR(driver_.MemFreeHost(ptr));
  } else {
    HIP_CHECK_ERR(driver_.MemFreeAsync(ptr, stream_));
    MAYBE_SYNC(*this);
  }
  return success();
}

void *HIPDeviceContext::driverContext() const {
  return hipDeviceToPtr(hipDevice());
}

ErrorOrSuccess HIPDeviceContext::enqueueCopyToDevice(const DeviceBuffer &dst,
                                                     const void *src) const {
  CHECK_IS_HIP_ERR(dst.context().device());
  HIP_WITH_CTX_ERR(*this);

  hipDeviceptr_t dstPtr = dst.ptr();
  size_t byteSize = dst.byteSize();
  HIP_CHECK_ERR(driver_.MemcpyHtoDAsync(dstPtr, src, byteSize, stream_));
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess
HIPDeviceContext::enqueueCopyFromDevice(void *dst,
                                        const DeviceBuffer &src) const {
  CHECK_IS_HIP_ERR(src.context().device());
  HIP_WITH_CTX_ERR(*this);

  hipDeviceptr_t srcPtr = src.ptr();
  size_t byteSize = src.byteSize();
  HIP_CHECK_ERR(driver_.MemcpyDtoHAsync(dst, srcPtr, byteSize, stream_));
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess
HIPDeviceContext::enqueueCopyToDeviceSized(void *dst, void *src,
                                           size_t byteSize) const {
  HIP_WITH_CTX_ERR(*this);

  HIP_CHECK_ERR(driver_.MemcpyHtoDAsync(dst, src, byteSize, stream_));
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess
HIPDeviceContext::enqueueCopyFromDeviceSized(void *dst, void *src,
                                             size_t byteSize) const {
  HIP_WITH_CTX_ERR(*this);

  hipDeviceptr_t srcPtr = src;
  HIP_CHECK_ERR(driver_.MemcpyDtoHAsync(dst, srcPtr, byteSize, stream_));
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess HIPDeviceContext::memCopyDtoD(const DeviceBuffer &dst,
                                             const DeviceBuffer &src) const {
  CHECK_IS_HIP_ERR(src.context().device());
  CHECK_IS_HIP_ERR(dst.context().device());

  const auto &srcCtx = HIPDeviceContext::cast(src.context());
  const auto &dstCtx = HIPDeviceContext::cast(dst.context());

  if (this->device_ != dstCtx.device_) {
    RAISE_ERROR("device context does not match context of dst");
  }

  // It is recommended to set the src device to be the current device according
  // to documentation:
  // https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv49hipMemcpyPvPKv6size_t13hipMemcpyKind
  HIP_WITH_CTX_ERR(srcCtx);

  hipDeviceptr_t srcPtr = src.ptr();
  hipDeviceptr_t dstPtr = dst.ptr();
  size_t dstByteSize = dst.byteSize();
  assert(src.byteSize() >= dstByteSize);
  if (srcCtx.hipStream() != dstCtx.hipStream()) {
    // Ensure that we are waiting for the source data to be ready.
    DEFINE_OR_RAISE(DeviceEventRef, srcEvent, srcCtx.enqueueEvent());
    ERR_CHECK_ERR(dstCtx.enqueueWaitForEvent(*srcEvent));

    HIP_CHECK_ERR(
        driver_.MemcpyDtoDAsync(dstPtr, srcPtr, dstByteSize, stream_));

    // Wait for the copy to complete before proceeding with the source stream
    // (e.g. freeing or overwriting memory).
    DEFINE_OR_RAISE(DeviceEventRef, dstEvent, dstCtx.enqueueEvent());
    ERR_CHECK_ERR(srcCtx.enqueueWaitForEvent(*dstEvent));
  } else {
    HIP_CHECK_ERR(
        driver_.MemcpyDtoDAsync(dstPtr, srcPtr, dstByteSize, stream_));
  }
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess HIPDeviceContext::enqueueSetMemory(const DeviceBuffer &dst,
                                                  uint64_t val,
                                                  size_t valSize) const {
  CHECK_IS_HIP_ERR(dst.context().device());
  void *ptr = dst.ptr();
  size_t byteSize = dst.byteSize();
  if (valSize == 1) {
    HIP_CHECK_ERR(driver_.MemsetD8Async(ptr, static_cast<uint8_t>(val),
                                        byteSize, stream_));
  } else if (valSize == 2) {
    if ((byteSize & 1) != 0) {
      return Error("byteSize of buffer is misaliged");
    }
    size_t len = byteSize >> 1;
    HIP_CHECK_ERR(
        driver_.MemsetD16Async(ptr, static_cast<uint16_t>(val), len, stream_));
  } else if (valSize == 4) {
    if ((byteSize & 3) != 0) {
      return Error("byteSize of buffer is misaliged");
    }
    size_t len = byteSize >> 2;
    HIP_CHECK_ERR(
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
      HIP_CHECK_ERR(driver_.MemsetD32Async(ptr, loBits, len, stream_));
    } else {
      size_t len = byteSize >> 3;
      // For each 64-bit element, set the low and high 32-bit parts separately
      for (size_t i = 0; i < len; i++) {
        void *loPtr =
            reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ptr) + i * 8);
        void *hiPtr = reinterpret_cast<void *>(
            reinterpret_cast<uintptr_t>(ptr) + i * 8 + 4);
        HIP_CHECK_ERR(driver_.MemsetD32Async(loPtr, loBits, 1, stream_));
        HIP_CHECK_ERR(driver_.MemsetD32Async(hiPtr, hiBits, 1, stream_));
      }
    }
  }
  MAYBE_SYNC(*this);
  return success();
}

DeviceContext::MMapFileImplResult
HIPDeviceContext::memmapFileImpl(const std::filesystem::path &filename) const {
  return Error("Unimplemented for HIP devices");
}

ErrorOr<DeviceFunctionRef> HIPDeviceContext::loadFunction(
    StringRef moduleName, StringRef functionName,
    const DeviceFunction::SignatureBuilder &signature, StringRef code,
    int32_t maxDynamicSharedBytes, StringRef debugLevel,
    int32_t optimizationLevel) const {
  DeviceFunctionCacheEntryRef entry =
      device().findInFunctionCache(moduleName.empty() ? code : moduleName);
  if (entry.getPointer() != nullptr) {
    HIPModuleFunctionRef moduleFunc =
        HIPModuleFunctionRef::take(HIPModuleFunction::cast(entry.release()));
    return HIPDeviceFunctionRef::create(*this, signature,
                                        std::move(moduleFunc));
  }

  // TODO(iposva): Remove once not needed for debugging.
  if (false) {
    std::cerr << "HIPDeviceContext::loadFunction(" << functionName.str()
              << "):\n"
              << code.data() << "\n";
  }
  hipModule_t module;

  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.ModuleLoadData(&module, code.data()));

  hipFunction_t function;
  HIP_CHECK_ERR(
      driver_.ModuleGetFunction(&function, module, functionName.data()));

  // Allocate func outside of the lock region to reduce the time spent under
  // lock. If this functionName has been registered concurrently, then we will
  // release this ref outside of the lock region before returning.
  auto moduleFunc = HIPModuleFunctionRef::create(
      std::string(functionName), this->hipDriver(), module, function);
  HIPDeviceFunctionRef result;
  entry = device().insertInFunctionCache(moduleName.empty() ? code : moduleName,
                                         *moduleFunc);
  result = HIPDeviceFunctionRef::create(
      *this, signature,
      HIPModuleFunctionRef::take(HIPModuleFunction::cast(entry.release())));
  return result;
}

ErrorOrSuccess HIPDeviceContext::enqueueFunctionExec(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    const DeviceFunction::Arguments &args) const {
  if (args.hasError()) {
    RAISE_ERROR(args.getError().getError());
  }

  const auto &hipFunc = HIPDeviceFunction::cast(func);

  HIP_WITH_CTX_ERR(*this);
  return hipFunc.hipCallKernel(gridX, gridY, gridZ, blockX, blockY, blockZ,
                               sharedMemBytes,
                               HIPDeviceFunctionArguments::cast(args), stream_);
}

ErrorOrSuccess HIPDeviceContext::occupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, const DeviceFunction &func, int blockSize,
    size_t dynamicSharedMemSize) const {
  const auto &hipFunc = HIPDeviceFunction::cast(func);
  HIP_WITH_CTX_ERR(*this);
  return (hipFunc.occupancyMaxActiveBlocksPerMultiprocessor(
      numBlocks, blockSize, dynamicSharedMemSize));
}

ErrorOrSuccess HIPDeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args) const {
  const auto &hipFunc = HIPDeviceFunction::cast(func);
  if (numAttrs > 0)
    RAISE_ERROR("Function launch attributes not supported;");

  HIP_WITH_CTX_ERR(*this);
  return hipFunc.hipCallKernelDirect(gridX, gridY, gridZ, blockX, blockY,
                                     blockZ, sharedMemBytes, args, stream_);
}

ErrorOr<DeviceStreamRef> HIPDeviceContext::stream() const {
  return HIPDeviceStreamRef::create(HIPDevice::cast(this->device()), stream_);
}

ErrorOr<DeviceStreamRef>
HIPDeviceContext::streamCreate(unsigned int flags) const {
  hipStream_t stream;
  HIP_CHECK_ERR(this->hipDriver().StreamCreate(&stream, flags));
  return HIPDeviceStreamRef::create(HIPDevice::cast(this->device()), stream);
}

ErrorOr<DeviceStreamRef>
HIPDeviceContext::streamCreateWithPriority(unsigned int flags,
                                           int priority) const {
  hipStream_t stream;
  HIP_CHECK_ERR(
      this->hipDriver().StreamCreateWithPriority(&stream, flags, priority));
  return HIPDeviceStreamRef::create(HIPDevice::cast(this->device()), stream);
}

ErrorOrSuccess
HIPDeviceContext::streamPriorityRange(int *leastPriority,
                                      int *greatestPriority) const {
  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.StreamPriorityRange(leastPriority, greatestPriority));
  return success();
}

ErrorOrSuccess HIPDeviceContext::synchronize() const {
  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.StreamSynchronize(stream_));
  return success();
}

ErrorOr<DeviceEventRef> HIPDeviceContext::enqueueEvent() const {
  hipEvent_t event;
  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.EventCreate(&event, 0));
  HIP_CHECK_ERR(driver_.EventRecord(event, stream_));
  return HIPDeviceEventRef::create(HIPDevice::cast(this->device()), event);
}

ErrorOr<DeviceEventRef>
HIPDeviceContext::eventCreate(unsigned int flags) const {
  hipEvent_t event;
  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.EventCreate(&event, flags));
  return HIPDeviceEventRef::create(HIPDevice::cast(this->device()), event);
}

ErrorOrSuccess
HIPDeviceContext::enqueueWaitFor(const DeviceEvent &event) const {
  const auto &hipEvent = HIPDeviceEvent::cast(event);
  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.StreamWaitEvent(stream_, hipEvent.hipEvent(), 0));
  return success();
}

ErrorOrSuccess
HIPDeviceContext::enqueueWaitFor(const DeviceContext &ctx) const {
  DEFINE_OR_RAISE(DeviceEventRef, event, ctx.enqueueEvent());
  ERR_CHECK_ERR(enqueueWaitFor(*event));
  return success();
}

ErrorOr<DeviceTimerRef> HIPDeviceContext::startTimer() const {
  hipEvent_t start;
  hipEvent_t stop;

  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.EventCreate(&start, 0));
  HIP_CHECK_ERR(driver_.EventCreate(&stop, 0));

  auto timer = HIPDeviceTimerRef::create(*this, start, stop);

  HIP_CHECK_ERR(driver_.EventRecord(timer->hipStartEvent(), stream_));

  return timer;
}

ErrorOr<std::chrono::nanoseconds>
HIPDeviceContext::stopTimer(const DeviceTimer &timer) const {
  const auto &hipTimer = HIPDeviceTimer::cast(timer);

  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.EventRecord(hipTimer.hipStopEvent(), stream_));
  HIP_CHECK_ERR(driver_.EventSynchronize(hipTimer.hipStopEvent()));

  float millis;
  HIP_CHECK_ERR(driver_.EventElapsedTime(&millis, hipTimer.hipStartEvent(),
                                         hipTimer.hipStopEvent()));

  constexpr double nanosPerMillisecond = 1000.0 * 1000.0;
  double nanos = static_cast<double>(millis) * nanosPerMillisecond;
  return std::chrono::nanoseconds(static_cast<int64_t>(nanos));
}

ErrorOrSuccess
HIPDeviceContext::enablePeerAccess(const DeviceContext &peer) const {
  if (device().id() == peer.device().id()) {
    // Enabling peer access to self is not allowed.
    RAISE_ERROR("Cannot enable peer access to self");
  }

  const HIPDevice *peerDevice = HIPDevice::verified_cast(&peer.device());
  if (peerDevice == nullptr) {
    RAISE_ERROR("Peer is not a HIP device");
  }

  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.DeviceEnablePeerAccess(peerDevice->hipDevice(), 0));
  return success();
}

ErrorOr<Device::MemInfo> HIPDeviceContext::getMemoryInfo() const {
  HIP_WITH_CTX(*this);
  // To get more accurate memory stats, ensure that at least all pending memory
  // allocations and frees have completed for this context's stream.
  HIP_CHECK_ERR(driver_.StreamSynchronize(stream_));

  return device().getMemoryInfo();
}

const HIPDriver &HIPDeviceContext::hipDriver() const { return driver_; }

hipDevice_t HIPDeviceContext::hipDevice() const { return device_; }

hipStream_t HIPDeviceContext::hipStream() const { return stream_; }

const HIPDeviceContext &HIPDeviceContext::cast(const DeviceContext &ctx) {
  return static_cast<const HIPDeviceContext &>(ctx);
}

const HIPDeviceContext *
HIPDeviceContext::verified_cast(const DeviceContext *ctx) {
  if (ctx->device().api() != HIPDevice::api) {
    return nullptr;
  }
  return static_cast<const HIPDeviceContext *>(ctx);
}

} // namespace M::AsyncRT
