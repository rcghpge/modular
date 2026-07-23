//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_LIB_DEVICEDRIVER_UTILS_H
#define GENERICML_LIB_DEVICEDRIVER_UTILS_H

#include "Support/ErrorOr.h"

namespace M::GraphCompiler::Driver {
struct Status {

  Status() : errorOr(success()) {}

  M::ErrorOrSuccess errorOr;
};
} // namespace M::GraphCompiler::Driver

#endif // GENERICML_LIB_DEVICEDRIVER_UTILS_H
