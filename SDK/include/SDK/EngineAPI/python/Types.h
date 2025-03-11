//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef SDK_ENGINEAPI_PYTHON_TYPES_H
#define SDK_ENGINEAPI_PYTHON_TYPES_H

#include "GenericML/API/max/c/types.h"
#include "Support/ML/DType.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include <string>

namespace nb = nanobind;

namespace M::core {
M_Dtype mapNanobindType(nb::dlpack::dtype dtype);

nb::dlpack::dtype mapDType(M_Dtype dtype);

std::string printDType(M::DType dtype);

} // namespace M::core

#endif // SDK_ENGINEAPI_PYTHON_TYPES_H
