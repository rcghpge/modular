//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SDK/EngineAPI/python/InputProcessing.h"
#include "GenericML/API/max/c/internal/value.h"
#include "GenericML/API/max/c/value.h"
#include "GenericML/API/max/cpp/aliases.h"
#include "SDK/EngineAPI/python/MojoValue.h"
#include "SDK/EngineAPI/python/Types.h"
#include "Support/CPython/PythonObject.h"
#include <vector>

using namespace M;

namespace nb = nanobind;

/// Verifies that the array in the nanobind object `obj` has the same shape and
/// data type as the tensor spec.
void core::verifyTensorInputSpecs(const nb::object &obj,
                                  const TensorSpec &inputTensorSpec) {
  TensorData tensor = nb::cast<core::TensorData>(obj);
  // Assert the input tensor has the same tensor specification
  // Assert same dtype.
  if (M::DType expectedDtype = inputTensorSpec.getDtype();
      tensor.dtype != expectedDtype)
    throw std::invalid_argument(
        "Input tensor has incompatible type. Expected type " +
        printDType(expectedDtype) + " but got " + printDType(tensor.dtype) +
        " instead.");

  // Assert same number of dims
  auto shapeOr = inputTensorSpec.getShape();
  if (shapeOr) {
    const std::vector<std::optional<int64_t>> &shape = *shapeOr;
    size_t actualNumDims = tensor.shape.size();
    if (unsigned long expectedNDim = shape.size();
        actualNumDims != expectedNDim) {
      throw std::invalid_argument(
          "Input tensor has incompatible number of dimensions. Expected "
          "number of dimensions to be " +
          std::to_string(expectedNDim) + " but input has " +
          std::to_string(actualNumDims) + " dimensions.");
    }

    // Assert same shape
    for (ssize_t d = 0, ndim = actualNumDims; d < ndim; ++d) {
      auto expectedDimOr = shape.at(d);
      if (!expectedDimOr)
        continue;
      if (int64_t actualDim = tensor.shape[d]; *expectedDimOr != actualDim) {
        throw std::invalid_argument(
            "Input tensor has incompatible dimension along axis " +
            std::to_string(d) + ". Expected " + std::to_string(*expectedDimOr) +
            " but found " + std::to_string(actualDim) + ".");
      }
    }
  }
}

/// Borrows the data from the array in the nanobind object `obj` into a new
/// `Tensor` wrapped in an `AsyncValue` and sets its name to
/// `name`.
M_AsyncValue *core::processTensorInput(const nb::object &obj,
                                       const std::string &name,
                                       M_RuntimeContext *context) {
  TensorData tensor = nb::cast<core::TensorData>(obj);
  TensorSpecPtr tensorSpec{M_newTensorSpec(
      tensor.shape.data(), tensor.shape.size(),
      static_cast<M_Dtype>(tensor.dtype.getValue()), name.c_str())};
  // tensor points to a buffer owned by the input tensor.
  // Since .execute() is synchronous currently, the tensor will survive at
  // least as long as we need it: we won't use the input tensor anymore
  // after .execute() returns.
  //
  // TODO(#7919): Figure out lifetime management for input tensors in the
  // asynchronous case.
  return M_createBorrowedTensor(tensor.dataPtr, tensorSpec.get(), context);
}

/// Recursively converts the nanobind object `pyObj` containing a value composed
/// of `array`, `dict`, `list`, `tuple`, `none`, `double`, `long`, `bool`
/// objects to a nested value of generic composed (`Dict`, `List`, Tuple`),
/// tensor (`Tensor`) and primitive types. Tensor data arrays are
/// borrowed and not copied.
M_AsyncValue *core::processValueInput(const nb::object &pyObj,
                                      M_RuntimeContext *context,
                                      M_Status *status) {
  if (nb::isinstance<core::TensorData>(pyObj))
    return processTensorInput(pyObj, /*name=*/"", context);

  if (nb::dict pyDict; nb::try_cast(pyObj, pyDict)) {
    M_AsyncValue *dictValue = M_createDictAsyncValue(context);
    M_AsyncDict *dict = M_getDictFromValue(dictValue);
    for (auto elem : pyDict) {
      M_AsyncValue *key =
          processValueInput(nb::cast<nb::object>(elem.first), context, status);
      M_AsyncValue *value =
          processValueInput(nb::cast<nb::object>(elem.second), context, status);
      M_insertIntoDict(dict, key, value);
      M_freeValue(key);
      M_freeValue(value);
    }
    M_freeDict(dict);
    return dictValue;
  }

  if (nb::list pyList; nb::try_cast(pyObj, pyList)) {
    M_AsyncValue *listValue = M_createListAsyncValue(context);
    M_AsyncList *list = M_getListFromValue(listValue);
    for (auto elem : pyList) {
      M_AsyncValue *value =
          processValueInput(nb::cast<nb::object>(elem), context, status);
      M_appendToList(list, value);
      M_freeValue(value);
    }
    M_freeList(list);
    return listValue;
  }

  if (nb::tuple pyTuple; nb::try_cast(pyObj, pyTuple)) {
    M_AsyncValue *tupleValue = M_createTupleAsyncValue(context);
    M_AsyncTuple *tuple = M_getTupleFromValue(tupleValue);
    for (auto elem : pyTuple) {
      M_AsyncValue *value =
          processValueInput(nb::cast<nb::object>(elem), context, status);
      M_borrowIntoTuple(tuple, value);
      M_freeValue(value);
    }
    M_freeTuple(tuple);
    return tupleValue;
  }

  if (pyObj.is_none())
    return M_createNoneAsyncValue(context);

  if (nb::str str; nb::try_cast(pyObj, str)) {
    return M_createStringAsyncValue(str.c_str(), context);
  }

  if (int64_t i; nb::try_cast(pyObj, i))
    return M_createLongAsyncValue(i, context);

  if (double d; nb::try_cast(pyObj, d))
    return M_createDoubleAsyncValue(d, context);

  if (bool b; nb::try_cast(pyObj, b))
    return M_createBoolAsyncValue(b, context);

  if (core::MojoValue * mojoVal; nb::try_cast(pyObj, mojoVal))
    return M_createMojoAsyncValueBorrowingFromExisting(mojoVal->value.get(),
                                                       context);

  using namespace ::M::CPython;
  PythonObjectWrapper *wrapper = new PythonObjectWrapper(pyObj.ptr());

  return M_createMojoAsyncValueFromPythonObjectWrapper(
      static_cast<void *>(wrapper), &freePythonObjectWrapper, context);
}
