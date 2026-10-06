//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#ifndef TARGET_LIB_CUDA_ERROR_H
#define TARGET_LIB_CUDA_ERROR_H

#include "HAL/M_driver_common.h"

#include "llvm/ADT/Twine.h"

#include <cuda.h>
#include <nvml.h>
#include <tl/expected.hpp>

#define M_DRIVER_STATUS_UNIMPLEMENTED (M_DRIVER_STATUS_CUSTOM_CODES_BASE - 0)
#define M_DRIVER_NUM_CUSTOM_STATUS_CODES 1

struct M_driver_driver;

namespace M::Target::CUDA {

M_driver_status_t cudaErrorStatus(M_driver_driver *driver, CUresult result,
                                  llvm::Twine const &context);

M_driver_status_t nvmlErrorStatus(M_driver_driver *driver, nvmlReturn_t result,
                                  llvm::Twine const &context);

} // namespace M::Target::CUDA

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
