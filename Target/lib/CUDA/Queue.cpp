//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Queue.h"
#include "Context.h"
#include "Device.h"
#include "Target/lib/CUDA/Error.h"

DriverResult<M_driver_queue_t> M_driver_queue::create(M_driver_context &ctx) {
  RETURN_IF_ERROR(ctx.ensureContextBound(), tl::unexpected(status));
  CUDALoader &cuda = ctx.device.driver.rawCuda;

  // Default priority is 0 (lowest); compatible with CUDA specs.
  int constexpr priority = 0;

  // CUDA blocking streams are a legacy feature; interpretation of "blocking"
  // is w.r.t. the default stream and does not mean "synchronous".
  // We want our streams to be independent of any other stream.
  unsigned int constexpr kNonBlocking = 0x1;

  CUstream stream;
  CUDA_CHECK_DR(
      cuda.cuStreamCreateWithPriority(&stream, kNonBlocking, priority),
      M_DRIVER_STATUS_UNKNOWN_ERROR);

  return new M_driver_queue(stream);
}

M_driver_status_t M_driver_queue::destroy(M_driver_context &ctx,
                                          M_driver_queue_t queue) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_HANDLE;

  QUICK_RETURN_IF_ERROR(ctx.ensureContextBound());
  CUDALoader &cuda = ctx.device.driver.rawCuda;
  CUDA_CHECK(cuda.cuStreamDestroy(queue->stream),
             M_DRIVER_STATUS_UNKNOWN_ERROR);

  delete queue;
  return M_DRIVER_STATUS_SUCCESS;
}
