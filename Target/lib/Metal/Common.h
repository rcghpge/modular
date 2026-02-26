//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_COMMON_H
#define TARGET_LIB_METAL_COMMON_H

#include "HAL/M_driver_common.h"

#define M_DRIVER_STATUS_UNIMPLEMENTED ((M_driver_status_t)(-7))
#define M_DRIVER_STATUS_UNINITIALIZED ((M_driver_status_t)(-8))
#define M_DRIVER_NUM_ERRORS 9

extern const char *status_messages[M_DRIVER_NUM_ERRORS];

#endif // TARGET_LIB_METAL_COMMON_H
