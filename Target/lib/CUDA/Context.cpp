//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#include "Context.h"
#include "Driver.h"
#include "Error.h"
#include "HAL/M_driver_api.h"

M_driver_status_t M_driver_context_create(M_driver_driver_t driver,
                                          M_driver_device_t device,
                                          M_driver_context_t *context) {
  CUcontext primaryCtx;
  CUDA_CHECK(
      driver->cuda().cuDevicePrimaryCtxRetain(&primaryCtx, device->deviceId),
      M_DRIVER_STATUS_BAD_DEVICE);

  *context = new M_driver_context{primaryCtx};

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context_destroy(M_driver_driver_t driver,
                                           M_driver_context_t context) {
  CUcontext primaryCtx = context->primaryCtx;

  CUdevice device;
  CUDA_CHECK(driver->cuda().cuCtxGetDevice_v2(&device, primaryCtx),
             M_DRIVER_STATUS_BAD_DEVICE);
  CUDA_CHECK(driver->cuda().cuDevicePrimaryCtxRelease(device),
             M_DRIVER_STATUS_BAD_CONTEXT);

  delete context;
  return M_DRIVER_STATUS_SUCCESS;
}
