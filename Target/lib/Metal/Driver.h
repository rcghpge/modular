//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_DRIVER_H
#define TARGET_LIB_METAL_DRIVER_H

#include "Common.h"
#include "Device.h"
#include "Target/lib/Support/Utils.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;
using M::Target::CustomStatus;

struct M_driver_driver {
  M_driver_driver() {
    status.messages[M_DRIVER_STATUS_UNIMPLEMENTED] = "Unimplemented";
    status.messages[M_DRIVER_STATUS_UNINITIALIZED] = "Uninitialized";
  }

  [[nodiscard]] static M_driver_driver_t create();
  static void destroy(M_driver_driver_t driver);

  static void ensureThreadlocalAutoreleasePoolExists() {
    thread_local NS::SharedPtr<NS::AutoreleasePool> autoreleasePool;
    if (!autoreleasePool) {
      autoreleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    }
  }

  SmallVector<M_driver_device, 1> devices;
  CustomStatus status{M_DRIVER_NUM_CUSTOM_STATUS_CODES};
};

#endif // TARGET_LIB_METAL_DRIVER_H
