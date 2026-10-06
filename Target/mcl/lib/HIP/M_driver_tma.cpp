//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// HIP has no TMA engine, so the descriptor encoders report
// M_DRIVER_STATUS_INCOMPATIBLE. The symbols must still exist because plugin
// entry points are resolved eagerly at load time.

#include "HAL/M_driver_api.h"

M_driver_status_t
M_driver_tensor_map_encode_tiled(M_driver_context_t, void *, int32_t, uint32_t,
                                 uint64_t, const uint64_t *, const uint64_t *,
                                 const uint32_t *, const uint32_t *, int32_t,
                                 int32_t, int32_t, int32_t) {
  return M_DRIVER_STATUS_INCOMPATIBLE;
}

M_driver_status_t M_driver_tensor_map_encode_im2col(
    M_driver_context_t, void *, int32_t, uint32_t, uint64_t, const uint64_t *,
    const uint64_t *, const int32_t *, const int32_t *, int32_t, int32_t,
    const uint32_t *, int32_t, int32_t, int32_t, int32_t) {
  return M_DRIVER_STATUS_INCOMPATIBLE;
}
