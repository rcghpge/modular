//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_ENGINEAPI_PYTHON_RESULTPROCESSING_H
#define SDK_ENGINEAPI_PYTHON_RESULTPROCESSING_H

#include "GenericML/API/max/c/value.h"
#include "GenericML/API/max/cpp/aliases.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include <vector>

namespace py = pybind11;

namespace M::core {

/// Creates a new `py::array` by copying the tensor data from a
/// `Tensor`.
py::object processTensorResult(TensorPtr outTensor);

/// Recursively converts the `value` containing nested generic
/// `Dict`, `List`, `Tuple`, `None`, `Tensor` and primitive type
/// values to a nested pybind object value composed of `array`, `dict`, `list`,
/// `tuple`, `none`, `double`, `long` and `bool`. Tensor data arrays are copied.
py::object processValueResult(ValuePtr value);

} // namespace M::core

#endif // SDK_ENGINEAPI_PYTHON_RESULTPROCESSING_H
