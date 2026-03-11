//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_HIP_ERROR_H
#define TARGET_LIB_HIP_ERROR_H

#include "HAL/M_driver_common.h"
#include <tl/expected.hpp>

#define M_DRIVER_STATUS_UNIMPLEMENTED (M_DRIVER_STATUS_CUSTOM_CODES_BASE - 0)
#define M_DRIVER_STATUS_BAD_HANDLE (M_DRIVER_STATUS_CUSTOM_CODES_BASE - 1)
#define M_DRIVER_NUM_CUSTOM_STATUS_CODES 2

extern const char *customStatusMessages[M_DRIVER_NUM_CUSTOM_STATUS_CODES];

template <typename T>
using DriverResult = tl::expected<T, M_driver_status_t>;

#endif // TARGET_LIB_HIP_ERROR_H
