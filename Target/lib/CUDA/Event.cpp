//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Event.h"
#include "Context.h"
#include "Device.h"

DriverResult<M_driver_event_t> M_driver_event::create(M_driver_context &ctx) {
  CUDALoader &cuda = ctx.getDevice().getDriver().cuda();

  CUDA_CHECK_DR(cuda.cuCtxSetCurrent(ctx.getPrimaryCtx()),
                M_DRIVER_STATUS_BAD_CONTEXT);

  CUevent event;
  CUDA_CHECK_DR(cuda.cuEventCreate(&event, CU_EVENT_DISABLE_TIMING),
                M_DRIVER_STATUS_UNKNOWN_ERROR);

  return new M_driver_event(event);
}

M_driver_status_t M_driver_event::destroy(M_driver_context &ctx,
                                          M_driver_event_t event) {
  CUDALoader &cuda = ctx.getDevice().getDriver().cuda();

  CUDA_CHECK(cuda.cuCtxSetCurrent(ctx.getPrimaryCtx()),
             M_DRIVER_STATUS_BAD_CONTEXT);

  CUDA_CHECK(cuda.cuEventDestroy(event->getEvent()),
             M_DRIVER_STATUS_UNKNOWN_ERROR);

  delete event;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event::synchronize(M_driver_context &ctx,
                                              M_driver_event_t event) {
  CUDALoader &cuda = ctx.getDevice().getDriver().cuda();

  CUDA_CHECK(cuda.cuCtxSetCurrent(ctx.getPrimaryCtx()),
             M_DRIVER_STATUS_BAD_CONTEXT);

  CUDA_CHECK(cuda.cuEventSynchronize(event->getEvent()),
             M_DRIVER_STATUS_UNKNOWN_ERROR);

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event::isReady(M_driver_context &ctx,
                                          M_driver_event_t event,
                                          bool *is_ready) {
  CUDALoader &cuda = ctx.getDevice().getDriver().cuda();

  CUDA_CHECK(cuda.cuCtxSetCurrent(ctx.getPrimaryCtx()),
             M_DRIVER_STATUS_BAD_CONTEXT);

  CUresult result = cuda.cuEventQuery(event->getEvent());
  if (result == CUDA_SUCCESS) {
    *is_ready = true;
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (result == CUDA_ERROR_NOT_READY) {
    *is_ready = false;
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_ERROR;
}
