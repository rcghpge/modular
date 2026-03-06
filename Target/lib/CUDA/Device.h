//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CUDA_DEVICE_H
#define TARGET_LIB_CUDA_DEVICE_H

#include "HAL/M_driver_common.h"

#include <cuda.h>

struct M_driver_device {
  CUdevice deviceId;
};

#endif // TARGET_LIB_CUDA_DEVICE_H
