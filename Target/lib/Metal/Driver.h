//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_DRIVER_H
#define TARGET_LIB_METAL_DRIVER_H

#include "Device.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

namespace M::Target::Metal {
struct Device;

struct Driver {
  [[nodiscard]] static Driver *create();
  static void destroy(Driver *driver);

  M_driver_driver_t cast() { return reinterpret_cast<M_driver_driver_t>(this); }
  static Driver *from(M_driver_driver_t handle) {
    return reinterpret_cast<Driver *>(handle);
  }

  static void ensureThreadlocalAutoreleasePoolExists() {
    thread_local NS::SharedPtr<NS::AutoreleasePool> autoreleasePool;
    if (!autoreleasePool) {
      autoreleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    }
  }

  SmallVector<Device, 1> devices;
};
} // namespace M::Target::Metal

#endif // TARGET_LIB_METAL_DRIVER_H
