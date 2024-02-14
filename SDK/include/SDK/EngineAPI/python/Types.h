//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef SDK_ENGINEAPI_PYTHON_TYPES_H
#define SDK_ENGINEAPI_PYTHON_TYPES_H

#include "GenericML/API/max/c/types.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace M::core {
M_Dtype mapPybind11Type(pybind11::dtype dtype);

pybind11::dtype mapDType(M_Dtype dtype);

std::string printDType(M_Dtype dtype);

} // namespace M::core

#endif // SDK_ENGINEAPI_PYTHON_TYPES_H
