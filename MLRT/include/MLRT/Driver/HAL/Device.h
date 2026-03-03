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

#ifndef MLRT_DRIVER_HAL_DEVICE_H
#define MLRT_DRIVER_HAL_DEVICE_H

#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"

namespace M::Driver::HAL {

class Context;
using ContextRef = RCRef<Context>;

struct DeviceImpl;

class Device : public ReferenceCounted<Device> {
public:
  ~Device();

  ErrorOr<int64_t> id() const;
  ErrorOr<std::string const &> name() const;
  ErrorOr<ContextRef> createContext() const;

private:
  explicit Device(DeviceImpl *impl) : impl(impl) {}

  DeviceImpl *impl;

  friend class Driver;
};

using DeviceRef = RCRef<Device>;

} // namespace M::Driver::HAL

#endif // MLRT_DRIVER_HAL_DEVICE_H
