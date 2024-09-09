//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/DeviceDriver/Driver.h"
#include "GenericML/DeviceDriver/Utils.h"
#include "Kernels/DeviceContext.h"
#include "Runtime/MojoValue.h"
#include "Support/AssertStream.h"

using namespace M::Driver;
using namespace M;

DeviceMemory::~DeviceMemory() {
  if (ptr) {
    if (destructorFn)
      (*destructorFn)(ptr, getSpec());
    else if (auto err = devicePtr->freeData(ptr))
      llvm::report_fatal_error(err.getError());
  }
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
  return devPtr->getString().c_str();
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
  return tensorPtr->devicePtr.getPointer();
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

static void destroyDeviceContextQuiet(DeviceContext *ptr) {
  [[maybe_unused]] auto err = destroyDeviceContext(ptr);
  if (err.isError())
    ASSERT_STREAM(err.isError(), "Destroying device context caused error: "
                                     << err.takeError());
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
      deviceContexts.push_back(AnyAsyncValueRef::createReady<MojoValue>(
          runtime, MojoValue::createRef(nullptr)));
    } else {
      if (deviceMap.contains(devRef) && deviceMap.at(devRef)) {
        const auto &dev = deviceMap.at(devRef);
        // ==== CAUTION! ====
        // Need to ensure that dev (Device) outlives this AsyncValue
        // because Device will free the DeviceContext in its destructor.
        // This should be fixed by either:
        // 1. the graph compiler taking a Device* not a DeviceContext*.
        // 2. have MGP take an AnyAsyncValueRef<RCRef<DeviceContext>>.
        deviceContexts.push_back(AnyAsyncValueRef::createReady<MojoValue>(
            runtime, MojoValue::createRef(dev->tmpGetDeviceContext())));
      } else {
        ASSERT_STREAM(devRef.label == kCUDALabel,
                      "Only cuda device context supported");
        // In this case we use the device ref id as the device id. THIS IS NOT
        // THE CASE WHEN THE DEVICE IS PASSED IN TO THE GRAPH.
        auto dev = createDeviceContext(devRef.id);
        if (dev.isError())
          return dev.takeError();
        deviceContexts.push_back(AnyAsyncValueRef::createReady<MojoValue>(
            runtime, MojoValue::createSelfFree(dev.takeValue(),
                                               destroyDeviceContextQuiet)));
      }
    }
  }
  return deviceContexts;
}
