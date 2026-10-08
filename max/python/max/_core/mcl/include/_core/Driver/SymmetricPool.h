//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef MAX_CORE_DRIVER_SYMMETRICPOOL_H
#define MAX_CORE_DRIVER_SYMMETRICPOOL_H
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "_core/Driver/Buffer.h"
#include "_core/Driver/Device.h"

namespace M::Driver::Python {
/// Python handle to a driver-owned symmetric multicast pool.
///
/// The HAL device context has no multicast support: there, `isSupported` is
/// false, `lookup` finds nothing, and `allocate` throws.
///
/// The pool lives in the driver's process-wide registry and outlives this
/// handle: dropping the Python object does not free device memory. The handle
/// only hands out non-owning `Buffer` views.
struct SymmetricPool {
  /// Creates and registers the pool. Throws if a pool already exists for this
  /// device set or a device lacks multicast support.
  static SymmetricPool allocate(const std::vector<PyDeviceTy> &devices,
                                size_t byteSize);
  /// Returns the pool registered for `devices`, or nullopt.
  static std::optional<SymmetricPool>
  lookup(const std::vector<PyDeviceTy> &devices);

  /// Whether every device can back a multicast pool. A device that cannot
  /// report its capability counts as unsupported.
  static bool isSupported(const std::vector<PyDeviceTy> &devices);

  /// Usable bytes; the views handed out are exactly this long.
  size_t getByteSize() const;

  /// The device's own view of the pool, as a flat uint8 buffer.
  Buffer unicastView(const PyDeviceTy &device) const;

  /// The multicast alias the switch reduces through. Only meaningful to a
  /// kernel issuing multimem instructions.
  Buffer multicastView(const PyDeviceTy &device) const;

#if !MODULAR_HAL_DEVICE_CONTEXT
  M::Driver::DeviceMulticastBufferRef pool;
#endif
  std::vector<PyDeviceTy> devices;
};
} // namespace M::Driver::Python

#endif // MAX_CORE_DRIVER_SYMMETRICPOOL_H
