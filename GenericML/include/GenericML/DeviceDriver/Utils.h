//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_LIB_DEVICEDRIVER_UTILS_H
#define GENERICML_LIB_DEVICEDRIVER_UTILS_H

#include "Support/ErrorOr.h"

namespace M::Driver {
struct Status {
  M::ErrorOrSuccess errorOr;
};
} // namespace M::Driver

#endif // GENERICML_LIB_DEVICEDRIVER_UTILS_H
