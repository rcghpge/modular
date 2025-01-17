//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_ENGINEAPI_PYTHON_RESULTPROCESSING_H
#define SDK_ENGINEAPI_PYTHON_RESULTPROCESSING_H

#include "GenericML/API/max/cpp/aliases.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;

namespace M::core {

/// Creates a new `nb::ndarray` by copying the tensor data from a
/// `Tensor`.
nb::object processTensorResult(TensorPtr outTensor);

} // namespace M::core

#endif // SDK_ENGINEAPI_PYTHON_RESULTPROCESSING_H
