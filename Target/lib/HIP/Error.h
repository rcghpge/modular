//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ERROR_H
#define ERROR_H

#include "HAL/M_driver_common.h"
#include <tl/expected.hpp>

#define M_DRIVER_STATUS_UNIMPLEMENTED ((M_driver_status_t)(-9))
#define M_DRIVER_STATUS_BAD_HANDLE ((M_driver_status_t)(-10))
#define M_DRIVER_NUM_ERRORS 11

extern const char *statusMessages[M_DRIVER_NUM_ERRORS];

template <typename T>
using DriverResult = tl::expected<T, M_driver_status_t>;

#endif /* ERROR_H */
