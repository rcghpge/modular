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

#include "Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"

namespace M::Driver::HAL {

class Device;
using DeviceRef = RCRef<Device>;

struct ContextImpl;

class Context : public ReferenceCounted<Context> {
public:
  ~Context();

  /// Retrieve the device for this context.
  DeviceRef device() const;

  ErrorOr<BufferRef> allocBuffer(size_t byteSize, Buffer::AllocMode mode);

private:
  explicit Context(ContextImpl *impl) : impl(impl) {}

  ContextImpl *impl;

  friend class Device;
};

using ContextRef = RCRef<Context>;

} // namespace M::Driver::HAL

#endif // MLRT_DRIVER_HAL_CONTEXT_H
