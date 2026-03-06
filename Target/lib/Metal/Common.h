//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_COMMON_H
#define TARGET_LIB_METAL_COMMON_H

#include "HAL/M_driver_common.h"

#define M_DRIVER_STATUS_UNIMPLEMENTED ((M_driver_status_t)(-9))
#define M_DRIVER_STATUS_UNINITIALIZED ((M_driver_status_t)(-10))
#define M_DRIVER_NUM_ERRORS 11

extern const char *status_messages[M_DRIVER_NUM_ERRORS];

#endif // TARGET_LIB_METAL_COMMON_H
