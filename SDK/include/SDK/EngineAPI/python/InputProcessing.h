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

/// Recursively converts the nanobind object `pyObj` containing a value composed
/// of `array`, `dict`, `list`, `tuple`, `none`, `double`, `long`, `bool`
/// objects to a nested value of generic composed (`Dict`, `List`, Tuple`),
/// tensor (`Tensor`) and primitive types. Tensor data arrays are
/// borrowed and not copied.
M_AsyncValue *processValueInput(const nb::object &pyObj,
                                M_RuntimeContext *context, M_Status *status);

} // namespace M::core

#endif // SDK_ENGINEAPI_PYTHON_INPUTPROCESSING_H
