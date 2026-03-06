//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "Common.h"
#include "HAL/M_driver_driver.h"
#include "llvm/ADT/StringRef.h"

M_driver_driver_t M_driver_driver::create() {
  auto autoreleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  auto driver = std::make_unique<M_driver_driver>();
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

void M_driver_driver::destroy(M_driver_driver *driver) { delete driver; }

extern "C" M_driver_status_t M_driver_create(M_driver_driver_t *outDriver) {
  M_driver_driver *driver = M_driver_driver::create();
  if (!driver)
    return M_DRIVER_STATUS_NO_DEVICES;
  *outDriver = driver;
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t M_driver_destroy(M_driver_driver_t handle) {
  M_driver_driver::destroy(handle);
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t
M_driver_property([[maybe_unused]] M_driver_driver_t handle,
                  const char *property_name, void *value) {
  StringRef name(property_name);
  if (name == M_DRIVER_PROPERTY_API)
    *static_cast<const char **>(value) = "Metal";
  else
    return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
  return M_DRIVER_STATUS_SUCCESS;
}
