//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#include "Driver.h"
#include "Error.h"
#include "HAL/M_driver_api.h"

#include "cuda.h"

#include <memory>

Driver *Driver::create() {
  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS)
    return nullptr;
  auto driver = std::make_unique<Driver>();
  return driver.release();
}

M_driver_status_t Driver::destroy(Driver *driver) {
  M_driver_status_t result = M_DRIVER_STATUS_SUCCESS;
  M_driver_driver_t handle = driver->cast();
  for (size_t i = 0; i < driver->cudaContexts.size(); i++) {
    if (driver->cudaContextsValid[i]) {
      M_driver_status_t status =
          M_driver_context_destroy(handle, static_cast<M_driver_context_t>(i));
      if (status != M_DRIVER_STATUS_SUCCESS)
        result = status;
    }
  }
  delete driver;
  return result;
}

M_driver_status_t M_driver_create(M_driver_driver_t *outDriver) {
  Driver *driver = Driver::create();
  if (!driver)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;
  *outDriver = driver->cast();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_destroy(M_driver_driver_t handle) {
  M_driver_status_t status = Driver::destroy(Driver::from(handle));
  return status;
}

M_driver_status_t M_driver_property(const char *property_name, void *value) {
  (void)property_name;
  (void)value;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
