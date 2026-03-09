//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#include "Context.h"
#include "Device.h"

DriverResult<M_driver_context_t>
M_driver_context::create(M_driver_device &dev) {
  CUcontext primaryCtx;
  CUDA_CHECK_DR(
      dev.getDriver().cuda().cuDevicePrimaryCtxRetain(&primaryCtx, dev.id()),
      M_DRIVER_STATUS_BAD_DEVICE);

  return new M_driver_context(dev, primaryCtx);
}

M_driver_status_t M_driver_context::destroy(M_driver_context_t ctx) {
  CUcontext primaryCtx = ctx->primaryCtx;

  CUdevice cudaDevice;
  CUDA_CHECK(
      ctx->device.getDriver().cuda().cuCtxGetDevice_v2(&cudaDevice, primaryCtx),
      M_DRIVER_STATUS_BAD_DEVICE);
  CUDA_CHECK(
      ctx->device.getDriver().cuda().cuDevicePrimaryCtxRelease(cudaDevice),
      M_DRIVER_STATUS_BAD_CONTEXT);

  delete ctx;
  return M_DRIVER_STATUS_SUCCESS;
}
