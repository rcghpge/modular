//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the base DeviceGraphBuilder interface.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceGraphBuilder.h"

namespace M::Driver {

DeviceGraphBuilder::~DeviceGraphBuilder() = default;

void DeviceGraphBuilder::addOutput(AsyncRT::AnyAsyncValueRef output) {
  outputs.push_back(std::move(output));
}

} // namespace M::Driver
