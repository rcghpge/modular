//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides C++ bindings over the Hardware Abstraction Layer (HAL) C
// interface in SDK/Driver.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_HAL_FUNCTION_H
#define MLRT_DRIVER_HAL_FUNCTION_H

#include "HAL/M_driver_common.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"

namespace M::Driver::HAL {

class Context;

class Function : public ReferenceCounted<Function> {
public:
  ~Function();

private:
  explicit Function(Context *context, M_driver_function_t handle)
      : context(context), handle(handle) {}

  Context *context;
  M_driver_function_t handle;

  friend class Context;
};

using FunctionRef = RCRef<Function>;

} // namespace M::Driver::HAL

#endif // MLRT_DRIVER_HAL_FUNCTION_H
