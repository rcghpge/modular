//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CUDA_EVENT_H
#define TARGET_LIB_CUDA_EVENT_H

#include "Error.h"

#include <cuda.h>

struct M_driver_context;

struct M_driver_event {
private:
  CUevent event;

public:
  explicit M_driver_event(CUevent event) : event(event) {}

  [[nodiscard]] static DriverResult<M_driver_event_t>
  create(M_driver_context &ctx);

  static M_driver_status_t destroy(M_driver_context &ctx,
                                   M_driver_event_t event);

  static M_driver_status_t synchronize(M_driver_context &ctx,
                                       M_driver_event_t event);

  static M_driver_status_t isReady(M_driver_context &ctx,
                                   M_driver_event_t event, bool *is_ready);

  CUevent getEvent() const { return event; }
};

#endif // TARGET_LIB_CUDA_EVENT_H
