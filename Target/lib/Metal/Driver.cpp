//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "Common.h"
#include "HAL/M_driver_driver.h"

using namespace M::Target::Metal;

std::unique_ptr<Driver> &Driver::get() {
  static std::unique_ptr<Driver> driver{};
  return driver;
}

extern "C" M_driver_status_t M_driver_initialize(void) {
  auto autoreleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  NS::SharedPtr<NS::Array> mDevices = NS::TransferPtr(MTL::CopyAllDevices());
  if (!mDevices) {
    return M_DRIVER_STATUS_NO_DEVICES;
  }

  auto &driver = Driver::get();
  if (!driver) {
    driver = std::make_unique<Driver>();
  }

  driver->devices.reserve(mDevices->count());
  for (NS::UInteger i = 0; i < mDevices->count(); i++) {
    driver->devices.emplace_back(
        static_cast<MTL::Device *>(mDevices->object(i)));
  }

  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t M_driver_finalize(void) {
  Driver::get().reset(nullptr);
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t M_driver_property(const char *property_name,
                                               void *value) {
  (void)property_name;
  (void)value;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
