//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_API_PYTHON_TYPES_H
#define GENERICML_API_PYTHON_TYPES_H

#include "GenericML/API/modular/c/types.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace core {
M_Dtype mapPybind11Type(pybind11::dtype dtype);

pybind11::dtype mapDType(M_Dtype dtype);

std::string printDType(M_Dtype dtype);

} // namespace core

#endif // GENERICML_API_PYTHON_TYPES_H
