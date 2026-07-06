//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_HIP_CONTEXT_H
#define TARGET_LIB_HIP_CONTEXT_H

#include "Error.h"
#include "HAL/M_driver_memory.h"
#include "llvm/ADT/DenseMap.h"

struct M_driver_device;

struct M_driver_context {
private:
  M_driver_device &device;

public:
  [[nodiscard]] static DriverResult<M_driver_context_t>
  create(M_driver_device &dev);

  // TODO: should we implicitly also dealloc during destroy?
  static M_driver_status_t destroy(M_driver_context_t ctx) {
    delete ctx;
    return M_DRIVER_STATUS_SUCCESS;
  }

  explicit M_driver_context(M_driver_device &device) : device(device) {}

  M_driver_device &getDevice() { return device; }

  DriverResult<M_driver_memory_t> allocPinned(uint64_t size);
  DriverResult<M_driver_memory_t> allocSync(uint64_t size);
  DriverResult<M_driver_memory_t> wrapSync(uint64_t address, uint64_t size,
                                           bool owning);
  M_driver_status_t freePinned(M_driver_memory_t memory);
  M_driver_status_t freeSync(M_driver_memory_t memory);
};

#endif // TARGET_LIB_HIP_CONTEXT_H
