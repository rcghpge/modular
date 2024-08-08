//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/DeviceDriver/Driver.h"
#include "Runtime/MojoValue.h"

using namespace M::Driver;
using namespace M;

DeviceMemory::~DeviceMemory() {
  if (ptr) {
    if (destructorFn)
      (*destructorFn)(ptr, getSpec());
    else
      devicePtr->freeData(ptr);
  }
}

MODULAR_API_EXPORT Device *M_copyDevice(Device *devPtr) {
  auto newRef = M::copyRCRef(devPtr);
  return newRef.release();
}

MODULAR_API_EXPORT void M_freeDeviceData(Device *ptr, void *data) {
  ptr->freeData(data);
}

MODULAR_API_EXPORT void M_destroyDevice(Device *devPtr) {
  (void)M::takeRCRef(devPtr);
}

MODULAR_API_EXPORT const char *M_getDeviceDesc(const Device *devPtr) {
  return devPtr->getString().c_str();
}

MODULAR_API_EXPORT DeviceMemory *M_createDeviceMemory(const M::TensorSpec *spec,
                                                      Device *devPtr) {
  // CAUTION: this assumes that TensorSpec is bitwise identical in mojo and cpp
  return devPtr->createDeviceMemory(*spec);
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
                                           const DeviceMemory *src) {
  auto errOr = src->copy_to(*dst);
  assert(!errOr.isError());
}

MODULAR_API_EXPORT void *M_getData(DeviceMemory *deviceMem) {
  return deviceMem->data();
}

MODULAR_VISIBILITY_EXPORT ErrorOr<llvm::SmallVector<AnyAsyncValueRef>>
M::Driver::mapDeviceRefsToDevices(const ArrayRef<DeviceRef> deviceRefs,
                                  const Driver::DeviceMap &deviceMap,
                                  Runtime &runtime) {

  // We need to explicitly synchronize the device here until MGP shares the same
  // DeviceContext as the driver. Otherwise there may be an async alloc/copy
  // pending on the driver when the first kernel tries to access the resource.
  for (auto &dev : deviceMap)
    if (dev.second)
      dev.second->synchronize();

  llvm::SmallVector<AnyAsyncValueRef> deviceContexts;
  for (const auto &devRef : deviceRefs) {
    // CPU device not explicity yet so deviceMap["cpu"] may be null
    if (devRef.label == kCPULabel) {
      deviceContexts.push_back(AnyAsyncValueRef::createReady<MojoValue>(
          runtime, nullptr, [](void *) {}));
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
        // This should be an error, but for now we'll default construct in MGP
        deviceContexts.push_back(AnyAsyncValueRef::createReady<MojoValue>(
            runtime, nullptr, [](void *) {}));
      }
    }
  }
  return deviceContexts;
}
