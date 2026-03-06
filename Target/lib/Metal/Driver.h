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

struct M_driver_driver {
  [[nodiscard]] static M_driver_driver_t create();
  static void destroy(M_driver_driver_t driver);

  static void ensureThreadlocalAutoreleasePoolExists() {
    thread_local NS::SharedPtr<NS::AutoreleasePool> autoreleasePool;
    if (!autoreleasePool) {
      autoreleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    }
  }

  SmallVector<M_driver_device, 1> devices;
};

#endif // TARGET_LIB_METAL_DRIVER_H
