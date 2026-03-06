//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#ifndef ERROR_H
#define ERROR_H

#include "HAL/M_driver_common.h"

#include <cuda.h>
#include <stdint.h>

#define M_DRIVER_STATUS_UNIMPLEMENTED ((M_driver_status_t)(-9))
#define M_DRIVER_NUM_ERRORS 10

#define CUDA_CHECK(stmt, errorCode)                                            \
  do {                                                                         \
    CUresult result = (stmt);                                                  \
    if (result != CUDA_SUCCESS)                                                \
      return errorCode;                                                        \
  } while (false)

extern const char *statusMessages[M_DRIVER_NUM_ERRORS];

#endif /* ERROR_H */
