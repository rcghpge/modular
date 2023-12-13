//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_API_PYTHON_RESULTPROCESSING_H
#define GENERICML_API_PYTHON_RESULTPROCESSING_H

#include "GenericML/API/modular/c/value.h"
#include "GenericML/API/modular/cpp/aliases.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include <vector>

using namespace M;

namespace py = pybind11;

namespace core {

/// Creates a new `py::array` by copying the tensor data from a `GML::Tensor`.
py::object processTensorResult(TensorPtr outTensor);

/// Recursively converts the `value` containing nested generic
/// `Dict`, `List`, `Tuple`, `None`, `GML::Tensor` and primitive type values
/// to a nested pybind object value composed of `array`, `dict`, `list`,
/// `tuple`, `none`, `double`, `long` and `bool`. Tensor data arrays are copied.
py::object processValueResult(ValuePtr value);

} // namespace core

#endif // GENERICML_API_PYTHON_RESULTPROCESSING_H
