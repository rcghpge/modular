//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_DRIVERAPI_PYTHON_DTYPE_H
#define SDK_DRIVERAPI_PYTHON_DTYPE_H

#include "Support/ML/DType.h"

namespace M::Driver::Python {

DType convertIntToDType(int dtype);

int convertDTypeToInt(DType dtype);

} // namespace M::Driver::Python

#endif // SDK_DRIVERAPI_PYTHON_DTYPE_H
