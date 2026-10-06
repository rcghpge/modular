//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_COMMON_H
#define TARGET_LIB_METAL_COMMON_H

#include "HAL/M_driver_common.h"
#include "tl/expected.hpp"

#define M_DRIVER_STATUS_UNIMPLEMENTED (M_DRIVER_STATUS_CUSTOM_CODES_BASE - 0)
#define M_DRIVER_STATUS_UNINITIALIZED (M_DRIVER_STATUS_CUSTOM_CODES_BASE - 1)
#define M_DRIVER_NUM_CUSTOM_STATUS_CODES 2

template <class T>
using DriverResult = tl::expected<T, M_driver_status_t>;

#endif // TARGET_LIB_METAL_COMMON_H
