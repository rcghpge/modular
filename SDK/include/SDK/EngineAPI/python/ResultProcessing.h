//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_ENGINEAPI_PYTHON_RESULTPROCESSING_H
#define SDK_ENGINEAPI_PYTHON_RESULTPROCESSING_H

#include "GenericML/API/max/cpp/aliases.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace M::core {

/// Creates a new `py::array` by copying the tensor data from a
/// `Tensor`.
py::object processTensorResult(TensorPtr outTensor);

} // namespace M::core

#endif // SDK_ENGINEAPI_PYTHON_RESULTPROCESSING_H
