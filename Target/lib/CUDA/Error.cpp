//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "Driver.h"

#include "llvm/ADT/Twine.h"

#include <cstdint>

namespace M::Target::CUDA {

M_driver_status_t cudaErrorStatus(M_driver_driver *driver, CUresult result,
                                  llvm::Twine const &context) {
  const char *name = nullptr;
  const char *desc = nullptr;
  (void)driver->rawCuda.cuGetErrorName(result, &name);
  (void)driver->rawCuda.cuGetErrorString(result, &desc);

  if (name != nullptr && desc != nullptr)
    return driver->status.create(context + ": " + name + " (" + desc + ")");
  if (name != nullptr)
    return driver->status.create(context + ": " + name);
  if (desc != nullptr)
    return driver->status.create(context + ": " + desc);
  return driver->status.create(context + ": CUresult " +
                               llvm::Twine(static_cast<int64_t>(result)));
}

M_driver_status_t nvmlErrorStatus(M_driver_driver *driver, nvmlReturn_t result,
                                  llvm::Twine const &context) {
  const char *desc = driver->rawNvml.nvmlErrorString(result);
  if (desc != nullptr)
    return driver->status.create(context + ": " + desc);
  return driver->status.create(context + ": nvmlReturn_t " +
                               llvm::Twine(static_cast<int64_t>(result)));
}

} // namespace M::Target::CUDA
