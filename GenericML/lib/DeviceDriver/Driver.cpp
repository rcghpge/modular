//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/DeviceDriver/Driver.h"
#include "GenericML/DeviceDriver/Utils.h"
#include "MLRT/AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/AssertStream.h"
#include "Support/MArchTarget/MArchTargetMinimal.h"

using namespace M::Driver;
using namespace M;

// Evaluate EXPR. If the returned value is an error, return it.
// Otherwise, assign to a new variable named VAR.
#define ASSIGN_OR_RETURN(VAR, EXPR)                                            \
  auto VAR##OrErr = EXPR;                                                      \
  if (VAR##OrErr.isError())                                                    \
    return VAR##OrErr.takeError();                                             \
  auto VAR = VAR##OrErr.takeValue()

MODULAR_DRIVER_EXPORT DeviceMemory::~DeviceMemory() {
  if (buf->ptr() != nullptr && destructorFn) {
    (*destructorFn)(buf->ptr(), getSpec());
    buf->releasePtr();
  }
}

MODULAR_DRIVER_EXPORT ErrorOrSuccess
DeviceMemory::copyTo(DeviceMemory &dst) const {
  return this->buf->copyTo(*dst.buf);
}

MODULAR_DRIVER_EXPORT ErrorOr<DeviceMemory *>
DeviceMemory::createViewOn(Device &device) const {
  ASSIGN_OR_RETURN(newBuf, buf->createSubBuffer(0, buf->byteSize()));
  ErrorOrSuccess err =
      newBuf->reassignOwnershipTo(*device.tmpGetDeviceContext());
  if (err.isError())
    return err.takeError();
  return new DeviceMemory(M::copyRCRef(&device), newBuf, spec);
}

MODULAR_DRIVER_EXPORT ErrorOrSuccess DeviceMemory::memset(int value) {
  return buf->context().enqueueSetMemory(*buf, value, 1);
}

MODULAR_DRIVER_EXPORT Device *M_copyDevice(Device *devPtr) {
  auto newRef = M::copyRCRef(devPtr);
  return newRef.release();
}

MODULAR_DRIVER_EXPORT void M_freeDeviceData(Device *ptr, void *data,
                                            M::Driver::Status *status) {
  if (auto err = ptr->freeData(data))
    status->errorOr = err.takeError();
}

MODULAR_DRIVER_EXPORT void M_destroyDevice(Device *devPtr) {
  (void)M::takeRCRef(devPtr);
}

MODULAR_DRIVER_EXPORT const char *M_getDeviceDesc(const Device *devPtr) {
  return devPtr->getString().data();
}

MODULAR_DRIVER_EXPORT DeviceMemory *
M_createDeviceMemory(const M::TensorSpec *spec, Device *devPtr,
                     M::Driver::Status *status) {
  // CAUTION: this assumes that TensorSpec is bitwise identical in mojo and cpp
  auto deviceMemoryOr = devPtr->createDeviceMemory(*spec);
  if (deviceMemoryOr.isError()) {
    status->errorOr = deviceMemoryOr.takeError();
    return nullptr;
  }
  return *deviceMemoryOr;
}

MODULAR_DRIVER_EXPORT DeviceMemory *
M_createDeviceMemoryFrom(void *data, const M::TensorSpec *spec,
                         Device *devPtr) {
  // CAUTION: this assumes that TensorSpec is bitwise identical in mojo and cpp
  return devPtr->createDeviceMemory(data, *spec);
}

MODULAR_DRIVER_EXPORT Device *
M_getDeviceFromDeviceMemory(DeviceMemory *tensorPtr) {
  return tensorPtr->getDevice().getPointer();
}

MODULAR_DRIVER_EXPORT void *
M_takeDataFromDeviceMemory(DeviceMemory *tensorPtr) {
  return tensorPtr->release();
}

MODULAR_DRIVER_EXPORT void M_destroyDeviceMemory(DeviceMemory *tensorPtr) {
  delete tensorPtr;
}

MODULAR_DRIVER_EXPORT void M_copyDeviceMemory(DeviceMemory *dst,
                                              const DeviceMemory *src,
                                              M::Driver::Status *status) {
  if (auto err = src->copyTo(*dst))
    status->errorOr = err.takeError();
}

MODULAR_DRIVER_EXPORT void *M_getData(DeviceMemory *deviceMem) {
  return deviceMem->data();
}

MODULAR_DRIVER_EXPORT std::string Device::toString() const {
  std::string str;
  llvm::raw_string_ostream oss(str);
  oss << "Device(type=" << deviceSpecification.ref.label << ",";
  oss << "id=" << deviceSpecification.ref.id << ")";
  return str;
}

MODULAR_DRIVER_EXPORT ErrorOr<std::string> Device::getStats() {
  ASSIGN_OR_RETURN(info, ctx->getMemoryInfo());

  // Crafting JSON string
  std::string s;
  llvm::raw_string_ostream os(s);
  llvm::json::OStream j(os);

  j.object([&] {
    j.attribute("free_memory", info.free);
    j.attribute("total_memory", info.total);
  });

  os.flush();
  return s;
}

MODULAR_DRIVER_EXPORT ErrorOr<std::string> Device::getApi() {
  return ctx->device().api().str();
}

MODULAR_DRIVER_EXPORT ErrorOr<std::string> Device::getArchName() {
  return ctx->device().getArchName();
}

MODULAR_DRIVER_EXPORT ErrorOrSuccess Device::isCompatible() const {
  return ctx->device().isCompatible();
}

MODULAR_DRIVER_EXPORT
ErrorOr<std::pair<AsyncRT::DeviceBufferRef,
                  std::optional<llvm::sys::fs::mapped_file_region>>>
Device::memmapFile(const std::filesystem::path &filename) {
  return ctx->memmapFile(filename);
}

MODULAR_DRIVER_EXPORT ErrorOr<void *> Device::allocateData(size_t bytes) {
  ASSIGN_OR_RETURN(buf, ctx->enqueueCreateBuffer(bytes));
  // TODO change return type to DeviceBufferRef instead of returning naked void
  // pointer.
  return buf->releasePtr();
}

MODULAR_DRIVER_EXPORT ErrorOrSuccess Device::freeData(void *ptr) {
  // TODO change argument type to DeviceBufferRef.
  // Since we return a released pointer in allocateData(), we need to
  // re-create the DeviceBuffer to free it here. We have lost the size info,
  // but it will not be needed to free the buffer.
  ctx->createBufferOwning(ptr, 1, /*owning=*/true);
  return success();
}

MODULAR_DRIVER_EXPORT ErrorOr<DeviceMemory *>
Device::createDeviceMemory(const TensorSpec &spec, bool hostAlloc) {
  ASSIGN_OR_RETURN(buf, hostAlloc
                            ? ctx->createHostBuffer(spec.getSizeInBytes())
                            : ctx->enqueueCreateBuffer(spec.getSizeInBytes()));
  return new DeviceMemory(M::copyRCRef(this), buf, spec);
}

MODULAR_DRIVER_EXPORT DeviceMemory *
Device::createDeviceMemory(void *data, const TensorSpec &spec) {
  auto buf =
      ctx->createBufferOwning(data, spec.getSizeInBytes(), /*owning=*/true);
  return new DeviceMemory(M::copyRCRef(this), buf, spec);
}

MODULAR_DRIVER_EXPORT DeviceMemory *Device::createDeviceMemory(
    void *data, const TensorSpec &spec,
    llvm::unique_function<void(void *, const TensorSpec &)> destructor) {
  auto deviceMemory = createDeviceMemory(data, spec);
  deviceMemory->setDestructorFn(std::move(destructor));
  return deviceMemory;
}

MODULAR_VISIBILITY_EXPORT ErrorOr<llvm::SmallVector<AsyncRT::AnyAsyncValueRef>>
M::Driver::mapDeviceRefsToDevices(const ArrayRef<DeviceRef> deviceRefs,
                                  const Driver::DeviceMap &deviceMap,
                                  Runtime &runtime) {

  // We need to explicitly synchronize the device here until MGP shares the same
  // DeviceContext as the driver. Otherwise there may be an async alloc/copy
  // pending on the driver when the first kernel tries to access the resource.
  for (auto &dev : deviceMap) {
    if (dev.second) {
      if (auto err = dev.second->synchronize())
        return err.takeError();
    }
  }

  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> deviceContexts;
  for (const auto &devRef : deviceRefs) {
    // CPU device not explicitly set so deviceMap["cpu"] may be null
    if (devRef.label == kCPULabel) {
      ASSIGN_OR_RETURN(ctx, AsyncRT::DeviceContext::create("cpu", 0));
      deviceContexts.push_back(
          AnyAsyncValueRef::createReady<AsyncRT::DeviceContextRef>(runtime,
                                                                   ctx));
    } else {
      if (auto it = deviceMap.find(devRef);
          it != deviceMap.end() && it->second) {
        const RCRef<Device> &dev = it->second;
        deviceContexts.push_back(
            AnyAsyncValueRef::createReady<AsyncRT::DeviceContextRef>(
                runtime, dev->tmpGetDeviceContext()));
      } else {
        ASSERT_STREAM(devRef.label == kGPULabel,
                      "Only accelerator device context supported");
        // In this case we use the device ref id as the device id. THIS IS NOT
        // THE CASE WHEN THE DEVICE IS PASSED IN TO THE GRAPH.
        // We should eventually push the device selection up the stack into the
        // mt command and remove all of the code in this else branch.
        size_t numPhysicalDevices =
            AsyncRT::Device::numberOfDevices(AsyncRT::Device::gpuAPI);
        if (devRef.id >= numPhysicalDevices)
          return Error(devRef.toString() +
                       " id exceeds number of available accelerator devices.");
        ASSIGN_OR_RETURN(ctx, AsyncRT::DeviceContext::create(
                                  AsyncRT::Device::gpuAPI, devRef.id));
        deviceContexts.push_back(
            AnyAsyncValueRef::createReady<AsyncRT::DeviceContextRef>(runtime,
                                                                     ctx));
      }
    }
  }
  return deviceContexts;
}

static ErrorOr<Device *> createCPUDevice(ssize_t numaId) {
  ASSIGN_OR_RETURN(targetInfo, getHostTargetInfo());
  DeviceSpec spec;
  spec.ref.id = 0; // TODO: this was stored in CPUDevice but never used???
  spec.ref.label = kCPULabel;
  spec.target = std::move(targetInfo);
  ASSIGN_OR_RETURN(ctx, AsyncRT::DeviceContext::create("cpu"));
  auto device = RCRef<Device>::create(spec, ctx);

  ErrorOrSuccess compatibleOr = device->isCompatible();
  if (compatibleOr.isError())
    return compatibleOr.takeError();

  return device.release();
}

MODULAR_DRIVER_EXPORT Device *M_createCPUDevice(ssize_t numaId,
                                                M::Driver::Status *status) {
  auto deviceOrError = createCPUDevice(numaId);
  if (deviceOrError.isError()) {
    status->errorOr = deviceOrError.takeError();
    return nullptr;
  }
  return deviceOrError.takeValue();
}

ErrorOr<std::unique_ptr<Device, decltype(&M_destroyDevice)>>
M::Driver::makeCPUDevice() {
  M::Driver::Status status;
  M::Driver::Device *device = M_createCPUDevice(0, &status);
  if (status.errorOr.isError())
    return status.errorOr.takeError();
  return std::unique_ptr<Device, decltype(&M_destroyDevice)>{device,
                                                             &M_destroyDevice};
}

static ErrorOr<Device *> createAccelerator(ssize_t gpuID,
                                           int64_t deviceMemoryLimit = -1) {
  size_t id = gpuID >= 0 ? gpuID : 0;

  ASSIGN_OR_RETURN(ctx, AsyncRT::DeviceContext::getOrCreate(
                            AsyncRT::Device::gpuAPI, id, deviceMemoryLimit));
  DeviceSpec spec;
  spec.ref.label = kGPULabel;
  spec.ref.id = gpuID;
  auto device = RCRef<Device>::create(spec, ctx);

  ErrorOrSuccess compatibleOr = device->isCompatible();
  if (compatibleOr.isError())
    return compatibleOr.takeError();

  return device.release();
}

MODULAR_DRIVER_EXPORT Device *
M_createAcceleratorDevice(ssize_t gpuID, int64_t deviceMemoryLimit,
                          M::Driver::Status *status) {
  auto deviceOrError = createAccelerator(gpuID, deviceMemoryLimit);
  if (deviceOrError.isError()) {
    status->errorOr = deviceOrError.takeError();
    return nullptr;
  }
  return deviceOrError.takeValue();
}

MODULAR_DRIVER_EXPORT void *M_getDeviceContext(Device *device) {
  return (void *)device->tmpGetDeviceContext().getPointer();
}

MODULAR_DRIVER_EXPORT int M_getAcceleratorCount() {
  return AsyncRT::Device::numberOfDevices(AsyncRT::Device::gpuAPI);
}
