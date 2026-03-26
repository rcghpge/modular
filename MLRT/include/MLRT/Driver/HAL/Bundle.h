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

#ifndef MLRT_DRIVER_HAL_BUNDLE_H
#define MLRT_DRIVER_HAL_BUNDLE_H

#include "HAL/M_driver_bundle.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"

namespace M::Driver::HAL {

class Context;

/// Roughly corresponds to a module or library.
/// May contain any number of functions.
/// Holds backing data for all functions within the module,
/// so is collectively owned by all FunctionRef that live within this module.
class Bundle : public ReferenceCounted<Bundle> {
public:
  ~Bundle();

private:
  explicit Bundle(Context *context, M_driver_runtime_bundle_t handle)
      : context(context), handle(handle) {}

  Context *context;
  M_driver_runtime_bundle_t handle;

  friend class Context;
};

using BundleRef = RCRef<Bundle>;

} // namespace M::Driver::HAL

#endif // MLRT_DRIVER_HAL_BUNDLE_H
