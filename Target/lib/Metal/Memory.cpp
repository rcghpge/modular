//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "HAL/M_driver_api.h"

M_driver_status_t
M_driver_memory_alloc([[maybe_unused]] M_driver_context_t context,
                      [[maybe_unused]] M_driver_alloc_mode_t mode,
                      [[maybe_unused]] uint64_t size,
                      [[maybe_unused]] void **memory) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t
M_driver_memory_free([[maybe_unused]] M_driver_context_t context,
                     [[maybe_unused]] void *memory) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
