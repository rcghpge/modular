//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "GenericML/API/max/c/symbol_export.h"

using namespace M::Driver;

DeviceMemory::~DeviceMemory() {
  if (ptr) {
    devicePtr->freeData(ptr);
  }
};

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
