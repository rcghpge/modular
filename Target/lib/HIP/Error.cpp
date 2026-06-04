//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "Driver.h"

#include "llvm/ADT/Twine.h"

#include <cstdint>

namespace M::Target::HIP {

M_driver_status_t hipErrorStatus(M_driver_driver *driver, hipError_t result,
                                 llvm::Twine const &context) {
  HIPApi &hip = driver->hip();
  const char *name = hip.hipGetErrorName(result);
  const char *desc = hip.hipGetErrorString(result);

  if (name != nullptr && desc != nullptr)
    return driver->status.create(context + ": " + name + " (" + desc + ")");
  if (name != nullptr)
    return driver->status.create(context + ": " + name);
  if (desc != nullptr)
    return driver->status.create(context + ": " + desc);
  return driver->status.create(context + ": hipError_t " +
                               llvm::Twine(static_cast<int64_t>(result)));
}

} // namespace M::Target::HIP
