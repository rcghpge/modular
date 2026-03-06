//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Buffer.h"
#include "BufferImpl.h"

namespace M::Driver::HAL {

Buffer::~Buffer() { delete impl; }

} // namespace M::Driver::HAL
