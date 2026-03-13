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

#include "HAL/M_driver_api.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"

namespace M::Driver::HAL {

class Context;
using ContextRef = RCRef<Context>;

class Driver;

class Device : public ReferenceCounted<Device> {
public:
  ~Device();

  ErrorOr<int64_t> getId() const;
  ErrorOr<StringRef> getName() const;
  ErrorOr<ContextRef> createContext();

private:
  explicit Device(Driver *driver, M_driver_device_t handle, int64_t id,
                  std::string name)
      : driver(driver), handle(handle), id(id), name(name) {}

  Driver *driver;
  M_driver_device_t handle;

  int64_t id;
  std::string name;

  friend class Driver;
  friend class Context;
  friend class Buffer;
  friend class Queue;
  friend class Event;
  friend class Function;
};

using DeviceRef = RCRef<Device>;

} // namespace M::Driver::HAL

#endif // MLRT_DRIVER_HAL_DEVICE_H
