//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ERROR_H
#define ERROR_H

#include <stdint.h>

#define M_DRIVER_STATUS_UNIMPLEMENTED ((M_driver_status_t)(-9))
#define M_DRIVER_NUM_ERRORS 10

extern const char *statusMessages[M_DRIVER_NUM_ERRORS];

#endif /* ERROR_H */
