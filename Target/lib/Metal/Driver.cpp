//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "Common.h"
#include "HAL/M_driver_driver.h"

using namespace M::Target::Metal;

Driver *Driver::create() {
  auto autoreleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  auto driver = std::make_unique<Driver>();
  if (!driver)
    return nullptr;

  NS::SharedPtr<NS::Array> mDevices = NS::TransferPtr(MTL::CopyAllDevices());
  if (!mDevices)
    return nullptr;

  driver->devices.reserve(mDevices->count());
  for (NS::UInteger i = 0; i < mDevices->count(); i++) {
    driver->devices.emplace_back(
        static_cast<MTL::Device *>(mDevices->object(i)));
  }

  return driver.release();
}

void Driver::destroy(Driver *driver) { delete driver; }

extern "C" M_driver_status_t M_driver_create(M_driver_driver_t *outDriver) {
  Driver *driver = Driver::create();
  if (!driver)
    return M_DRIVER_STATUS_NO_DEVICES;
  *outDriver = driver->cast();
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t M_driver_destroy(M_driver_driver_t handle) {
  Driver::destroy(Driver::from(handle));
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t M_driver_property(const char *property_name,
                                               void *value) {
  (void)property_name;
  (void)value;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
