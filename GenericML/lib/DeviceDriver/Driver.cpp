//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/DeviceDriver/Driver.h"
#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "GenericML/DeviceDriver/Utils.h"
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

MODULAR_API_EXPORT DeviceMemory::~DeviceMemory() {
  if (buf->ptr() != nullptr && destructorFn) {
    (*destructorFn)(buf->ptr(), getSpec());
    buf->releasePtr();
  }
}

MODULAR_API_EXPORT ErrorOrSuccess
DeviceMemory::copyTo(DeviceMemory &dst) const {
  // Here we ensure that the context iniating the copy is a GPU context, since
  // DeviceContext::copyDeviceToDevice() only works when both are the same kind.
  // TODO: Handle this within AsyncRT:DeviceContext.
  const auto &src = *this;
  if (src.getKind() == DeviceKind::kCPU)
    return dst.buf->context().enqueueCopyToDevice(*dst.buf, src.buf->ptr());
  if (src.getKind() == DeviceKind::kACCELERATOR) {
    if (dst.getKind() == DeviceKind::kCPU)
      return src.buf->context().enqueueCopyFromDevice(dst.buf->ptr(), *src.buf);
    if (dst.getKind() == DeviceKind::kACCELERATOR)
      return dst.buf->context().enqueueCopyDeviceToDevice(*dst.buf, *src.buf);
  }
  return Error("Unhandled device memory type");
}

MODULAR_API_EXPORT ErrorOrSuccess DeviceMemory::memset(int value) {
  return buf->context().setMemory(*buf, value, 1,
                                  AsyncRT::DeviceContext::SyncMode::ASYNC);
}

MODULAR_API_EXPORT Device *M_copyDevice(Device *devPtr) {
  auto newRef = M::copyRCRef(devPtr);
  return newRef.release();
}

MODULAR_API_EXPORT void M_freeDeviceData(Device *ptr, void *data,
                                         M::Driver::Status *status) {
  if (auto err = ptr->freeData(data))
    status->errorOr = err.takeError();
}

MODULAR_API_EXPORT void M_destroyDevice(Device *devPtr) {
  (void)M::takeRCRef(devPtr);
}

MODULAR_API_EXPORT const char *M_getDeviceDesc(const Device *devPtr) {
  return devPtr->getString().data();
}

MODULAR_API_EXPORT DeviceMemory *
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

MODULAR_API_EXPORT DeviceMemory *
M_createDeviceMemoryFrom(void *data, const M::TensorSpec *spec,
                         Device *devPtr) {
  // CAUTION: this assumes that TensorSpec is bitwise identical in mojo and cpp
  return devPtr->createDeviceMemory(data, *spec);
}

MODULAR_API_EXPORT Device *
M_getDeviceFromDeviceMemory(DeviceMemory *tensorPtr) {
  return tensorPtr->getDevice().getPointer();
}

MODULAR_API_EXPORT void *M_takeDataFromDeviceMemory(DeviceMemory *tensorPtr) {
  return tensorPtr->release();
}

MODULAR_API_EXPORT void M_destroyDeviceMemory(DeviceMemory *tensorPtr) {
  delete tensorPtr;
}

MODULAR_API_EXPORT void M_copyDeviceMemory(DeviceMemory *dst,
                                           const DeviceMemory *src,
                                           M::Driver::Status *status) {
  if (auto err = src->copyTo(*dst))
    status->errorOr = err.takeError();
}

MODULAR_API_EXPORT void *M_getData(DeviceMemory *deviceMem) {
  return deviceMem->data();
}

MODULAR_API_EXPORT std::string Device::toString() const {
  std::string str;
  llvm::raw_string_ostream oss(str);
  oss << "Device(type=" << deviceSpecification.ref.label << ",";
  if (getKind() != DeviceKind::kCPU)
    oss << "gpu_id=" << deviceSpecification.ref.id << ",";
  oss << formatTarget(deviceSpecification);
  return str;
}

MODULAR_API_EXPORT ErrorOr<std::string> Device::getStats() {
  ASSIGN_OR_RETURN(info, ctx->device().getMemoryInfo());

  // Crafting JSON string
  std::string s;
  llvm::raw_string_ostream os(s);
  llvm::json::OStream J(os);

  J.object([&] {
    J.attribute("free_memory", info.free);
    J.attribute("total_memory", info.total);
  });

  os.flush();
  return s;
}

MODULAR_API_EXPORT ErrorOr<void *> Device::allocateData(size_t bytes) {
  ASSIGN_OR_RETURN(buf, ctx->enqueueCreateBuffer(bytes));
  // TODO change return type to DeviceBufferRef instead of returning naked void
  // pointer.
  return buf->releasePtr();
}

MODULAR_API_EXPORT ErrorOrSuccess Device::freeData(void *ptr) {
  // TODO change argument type to DeviceBufferRef.
  // Since we return a released pointer in allocateData(), we need to
  // re-create the DeviceBuffer to free it here. We have lost the size info,
  // but it will not be needed to free the buffer.
  ctx->createBufferOwning(ptr, 1, /*owning=*/true);
  return success();
}

MODULAR_API_EXPORT ErrorOr<DeviceMemory *>
Device::createDeviceMemory(const TensorSpec &spec) {
  auto byteSize = spec.getNumElements() * spec.getEltType().getSizeInBytes();
  ASSIGN_OR_RETURN(buf, ctx->enqueueCreateBuffer(byteSize));
  return new DeviceMemory(M::copyRCRef(this), buf, spec, getKind());
}

MODULAR_API_EXPORT DeviceMemory *
Device::createDeviceMemory(void *data, const TensorSpec &spec) {
  auto byteSize = spec.getNumElements() * spec.getEltType().getSizeInBytes();
  auto buf = ctx->createBufferOwning(data, byteSize, /*owning=*/true);
  return new DeviceMemory(M::copyRCRef(this), buf, spec, getKind());
}

MODULAR_API_EXPORT DeviceMemory *Device::createDeviceMemory(
    void *data, const TensorSpec &spec,
    llvm::unique_function<void(void *, const TensorSpec &)> destructor) {
  auto deviceMemory = createDeviceMemory(data, spec);
  deviceMemory->setDestructorFn(std::move(destructor));
  return deviceMemory;
}

MODULAR_VISIBILITY_EXPORT ErrorOr<llvm::SmallVector<AnyAsyncValueRef>>
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

  llvm::SmallVector<AnyAsyncValueRef> deviceContexts;
  for (const auto &devRef : deviceRefs) {
    // CPU device not explicity yet so deviceMap["cpu"] may be null
    if (devRef.label == kCPULabel) {
      ASSIGN_OR_RETURN(ctx, AsyncRT::DeviceContext::create("cpu", 0));
      deviceContexts.push_back(
          AnyAsyncValueRef::createReady<AsyncRT::DeviceContextRef>(runtime,
                                                                   ctx));
    } else {
      if (deviceMap.contains(devRef) && deviceMap.at(devRef)) {
        const auto &dev = deviceMap.at(devRef);
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
        if (numPhysicalDevices == 0)
          // No GPU device found: Return an error.
          return Error("No GPU device available.");
        auto envPtr = std::getenv("MODULAR_NUM_VIRTUAL_CUDA_DEVICES");
        size_t numVirtualDevices =
            envPtr ? std::stoi(envPtr) : numPhysicalDevices;
        if (devRef.id >= numVirtualDevices)
          return Error(devRef.toString() +
                       " id exceeds number of available devices. Set "
                       "MODULAR_NUM_VIRTUAL_CUDA_DEVICES to allow the creation "
                       "of virtual devices.");
        auto virtualDeviceId = devRef.id % numPhysicalDevices;
        ASSIGN_OR_RETURN(ctx, AsyncRT::DeviceContext::create(
                                  AsyncRT::Device::gpuAPI, virtualDeviceId));
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
  auto device = RCRef<Device>::create(spec, DeviceKind::kCPU, ctx);
  return device.release();
}

MODULAR_API_EXPORT Device *M_createCPUDevice(ssize_t numaId,
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

// TODO: Rename to createAccelerator.
static ErrorOr<Device *> createCUDADevice(ssize_t gpuID) {
  size_t id = gpuID >= 0 ? gpuID : 0;

  ASSIGN_OR_RETURN(ctx,
                   AsyncRT::DeviceContext::create(AsyncRT::Device::gpuAPI, id));
  DeviceSpec spec;
  spec.ref.label = kGPULabel;
  spec.ref.id = gpuID;
  // Since the ctx creation above succeeded, we can assume we have cuda or hip.
  // Match the target triple from gpu/host/info.mojo.
  // TODO: where is this actually used?
  spec.target.triple = ctx->device().api() == "cuda"
                           ? llvm::Triple("nvptx64-nvidia-cuda")
                           : llvm::Triple("amdgcn-amd-amdhsa");
  spec.target.arch = ctx->device().getAcceleratorArchOrEmpty();
  auto device = RCRef<Device>::create(spec, DeviceKind::kACCELERATOR, ctx);

  return device.release();
}

MODULAR_API_EXPORT Device *M_createCUDADevice(ssize_t gpuID,
                                              M::Driver::Status *status) {
  auto deviceOrError = createCUDADevice(gpuID);
  if (deviceOrError.isError()) {
    status->errorOr = deviceOrError.takeError();
    return nullptr;
  }
  return deviceOrError.takeValue();
}

MODULAR_API_EXPORT void *M_getDeviceContext(Device *device) {
  return (void *)device->tmpGetDeviceContext().getPointer();
}
