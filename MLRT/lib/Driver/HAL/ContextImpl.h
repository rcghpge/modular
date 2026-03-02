//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "HAL/M_driver_api.h"

namespace M::Driver::HAL {

struct ContextImpl {
  ContextImpl() { reset(); }
  ~ContextImpl() { reset(); }

  void reset() { handle = 0xDEADBEEFFEEDFACEuLL; }

  M_driver_context_t handle;
};

} // namespace M::Driver::HAL
