//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "_core/Driver/CompletionFlag.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"

#include <stdexcept>
#include <string>

using namespace M::Driver;

Python::CompletionFlag::CompletionFlag(const Device &device) {
  M::Driver::DeviceContextRef ctx = device.ctx;
  // Non-CUDA backends return Error("allocateCompletionFlag is not
  // supported on this device") from the MLRT virtual default; propagate
  // that message verbatim to the Python caller.
  M::ErrorOr<M::Driver::CompletionFlagRef> flagOr =
      ctx->allocateCompletionFlag();
  if (flagOr.isError()) {
    throw std::runtime_error(std::string("CompletionFlag allocation failed: ") +
                             std::string(flagOr.getError()));
  }
  ref_ = std::move(*flagOr);
}
