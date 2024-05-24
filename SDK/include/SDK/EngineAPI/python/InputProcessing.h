//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_ENGINEAPI_PYTHON_INPUTPROCESSING_H
#define SDK_ENGINEAPI_PYTHON_INPUTPROCESSING_H

#include "GenericML/API/max/c/value.h"
#include "GenericML/API/max/cpp/aliases.h"
#include "SDK/EngineAPI/python/TensorSpec.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace M::core {

/// Verifies that the array in the pybind object `obj` has the same shape and
/// data type as the tensor spec.
void verifyTensorInputSpecs(const py::object &obj,
                            const TensorSpec &inputTensorSpec);

/// Borrows the data from the array in the pybind object `obj` into a new
/// `Tensor` wrapped in an `AsyncValue` and sets it's name to
/// `name`.
M_AsyncValue *processTensorInput(const py::object &obj, const std::string &name,
                                 M_RuntimeContext *context);

/// Borrows the data from the array in the pybind object `obj` into a new
/// `Tensor` wrapped in an `AsyncValue`.
M_AsyncValue *processTensorInput(const py::object &obj,
                                 M_RuntimeContext *context);

/// Recursively converts the pybind object `pyObj` containing a value composed
/// of `array`, `dict`, `list`, `tuple`, `none`, `double`, `long`, `bool`
/// objects to a nested value of generic composed (`Dict`, `List`, Tuple`),
/// tensor (`Tensor`) and primitive types. Tensor data arrays are
/// borrowed and not copied.
M_AsyncValue *processValueInput(const py::object &pyObj,
                                M_RuntimeContext *context, M_Status *status);

} // namespace M::core

#endif // SDK_ENGINEAPI_PYTHON_INPUTPROCESSING_H
