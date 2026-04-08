//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_DRIVER_H
#define TARGET_LIB_METAL_DRIVER_H

#include "Common.h"
#include "Device.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

using namespace llvm;

struct M_driver_driver {
  M_driver_driver() {
    customStatusMessages[M_DRIVER_STATUS_UNIMPLEMENTED] = "Unimplemented";
    customStatusMessages[M_DRIVER_STATUS_UNINITIALIZED] = "Uninitialized";
  }

  [[nodiscard]] static M_driver_driver_t create();
  static void destroy(M_driver_driver_t driver);

  static void ensureThreadlocalAutoreleasePoolExists() {
    thread_local NS::SharedPtr<NS::AutoreleasePool> autoreleasePool;
    if (!autoreleasePool) {
      autoreleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    }
  }

  M_driver_status_t createStatus(Twine message) {
    M_driver_status_t status = customStatusCodeNext++;
    SmallString<64> &storage = customStatusMessages[status];
    message.toVector(storage);
    return status;
  }

  SmallVector<M_driver_device, 1> devices;
  SmallDenseMap<M_driver_status_t, SmallString<64>> customStatusMessages;
  M_driver_status_t customStatusCodeNext =
      M_DRIVER_STATUS_CUSTOM_CODES_BASE - M_DRIVER_NUM_CUSTOM_STATUS_CODES;
};

#endif // TARGET_LIB_METAL_DRIVER_H
