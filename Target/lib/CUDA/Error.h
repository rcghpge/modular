//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#ifndef TARGET_LIB_CUDA_ERROR_H
#define TARGET_LIB_CUDA_ERROR_H

#include "HAL/M_driver_common.h"

#include <cuda.h>
#include <tl/expected.hpp>

#define M_DRIVER_STATUS_UNIMPLEMENTED (M_DRIVER_STATUS_CUSTOM_CODES_BASE - 0)
#define M_DRIVER_NUM_CUSTOM_STATUS_CODES 1

#define CUDA_CHECK(stmt, errorCode)                                            \
  do {                                                                         \
    CUresult result = (stmt);                                                  \
    if (result != CUDA_SUCCESS)                                                \
      return errorCode;                                                        \
  } while (false)

#define CUDA_CHECK_DR(stmt, errorCode)                                         \
  do {                                                                         \
    CUresult result = (stmt);                                                  \
    if (result != CUDA_SUCCESS)                                                \
      return tl::unexpected(errorCode);                                        \
  } while (false)

#define RETURN_IF_ERROR(stmt, error)                                           \
  do {                                                                         \
    M_driver_status_t status = (stmt);                                         \
    if (status != M_DRIVER_STATUS_SUCCESS)                                     \
      return error;                                                            \
  } while (0)

#define QUICK_RETURN_IF_ERROR(stmt)                                            \
  do {                                                                         \
    M_driver_status_t status = (stmt);                                         \
    if (status != M_DRIVER_STATUS_SUCCESS)                                     \
      return status;                                                           \
  } while (0)

template <typename T>
using DriverResult = tl::expected<T, M_driver_status_t>;

#endif // TARGET_LIB_CUDA_ERROR_H
