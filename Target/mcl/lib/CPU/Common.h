//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CPU_COMMON_H
#define TARGET_LIB_CPU_COMMON_H

#include "HAL/M_driver_common.h"
#include "tl/expected.hpp"

// Plugin-specific (custom) status codes. These map to human-readable strings
// registered on the driver's CustomStatus at construction time.
#define M_DRIVER_STATUS_UNIMPLEMENTED (M_DRIVER_STATUS_CUSTOM_CODES_BASE - 0)
#define M_DRIVER_NUM_CUSTOM_STATUS_CODES 1

template <class T>
using DriverResult = tl::expected<T, M_driver_status_t>;

#endif // TARGET_LIB_CPU_COMMON_H
