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

#ifndef MLRT_DRIVER_HAL_CONTEXT_H
#define MLRT_DRIVER_HAL_CONTEXT_H

#include "Device.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ReferenceCounted.h"

namespace M::Driver::HAL {

class Context : public ReferenceCounted<Context> {
public:
  ~Context();

  /// Retrieve the device for this context.
  DeviceRef device() const;

private:
  friend class Device;
  explicit Context(void *impl) : impl_(impl) {}

  void *impl_;
};

using ContextRef = RCRef<Context>;

} // namespace M::Driver::HAL

#endif // MLRT_DRIVER_HAL_CONTEXT_H
