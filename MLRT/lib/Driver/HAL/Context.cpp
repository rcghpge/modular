//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Context.h"
#include "ContextImpl.h"

namespace M::Driver::HAL {

Context::~Context() { delete static_cast<ContextImpl *>(impl_); }

} // namespace M::Driver::HAL
