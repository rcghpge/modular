//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CUDA_QUEUE_H
#define TARGET_LIB_CUDA_QUEUE_H

#include "Error.h"

#include <cuda.h>

struct M_driver_context;

struct M_driver_queue {
  CUstream stream;

  [[nodiscard]] static DriverResult<M_driver_queue_t>
  create(M_driver_context &ctx);

  static M_driver_status_t destroy(M_driver_context &ctx,
                                   M_driver_queue_t queue);

  explicit M_driver_queue(CUstream stream) : stream(stream) {}
};

#endif // TARGET_LIB_CUDA_QUEUE_H
