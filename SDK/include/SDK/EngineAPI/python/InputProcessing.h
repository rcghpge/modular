//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_ENGINEAPI_PYTHON_INPUTPROCESSING_H
#define SDK_ENGINEAPI_PYTHON_INPUTPROCESSING_H

#include "SDK/EngineAPI/python/TensorSpec.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;

namespace M::core {

/// Verifies that `tensor` has the same shape and
/// data type as the tensor spec.
void verifyTensorInputSpecs(const TensorData &obj,
                            const TensorSpec &inputTensorSpec);

/// Borrows the data from `tensor` into a new
/// `Tensor` wrapped in an `AsyncValue` and sets its name to
/// `name`.
M_AsyncValue *processTensorInput(const TensorData &obj, const std::string &name,
                                 M_RuntimeContext *context);

/// Recursively converts the nanobind object `pyObj` containing a value composed
/// of `array`, `dict`, `list`, `tuple`, `none`, `double`, `long`, `bool`
/// objects to a nested value of generic composed (`Dict`, `List`, Tuple`),
/// tensor (`Tensor`) and primitive types. Tensor data arrays are
/// borrowed and not copied.
M_AsyncValue *processValueInput(const nb::object &pyObj,
                                M_RuntimeContext *context, M_Status *status);

} // namespace M::core

#endif // SDK_ENGINEAPI_PYTHON_INPUTPROCESSING_H
