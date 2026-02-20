//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#include "HAL/M_driver_api.h"
#include "error.h"

#include "cuda.h"

#include <string.h>

M_driver_status_t initialize_devices(void);
M_driver_status_t finalize_devices(void);

M_driver_status_t M_driver_initialize(void) {
  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    return M_DRIVER_STATUS_UNKNOWN_ERROR;
  }
  M_driver_status_t status = initialize_devices();
  return status;
}

M_driver_status_t M_driver_finalize(void) {
  M_driver_status_t status = finalize_devices();
  return status;
}

M_driver_status_t M_driver_property(const char *property_name, void *value) {
  (void)property_name;
  (void)value;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
