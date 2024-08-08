//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SDK/EngineAPI/python/InputProcessing.h"
#include "GenericML/API/max/c/value.h"
#include "GenericML/API/max/cpp/aliases.h"
#include "SDK/EngineAPI/python/Types.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "llvm/ADT/StringRef.h"
#include <vector>

using namespace M;

namespace py = pybind11;

/// Verifies that the array in the pybind object `obj` has the same shape and
/// data type as the tensor spec.
void core::verifyTensorInputSpecs(const py::object &obj,
                                  const TensorSpec &inputTensorSpec) {
  const auto tensor = obj.cast<py::array>();

  // Assert the input tensor has the same tensor specification
  // Assert same dtype.
  if (pybind11::dtype expectedDtype = mapDType(inputTensorSpec.getDtype());
      !tensor.dtype().is(expectedDtype))
    throw std::invalid_argument(
        "Input tensor has incompatible type. Expected type " +
        printDType(inputTensorSpec.getDtype()) + " but got " +
        printDType(mapPybind11Type(tensor.dtype())) + " instead.");

  // Assert same number of dims
  auto shapeOr = inputTensorSpec.getShape();
  if (shapeOr) {
    const std::vector<std::optional<int64_t>> &shape = *shapeOr;
    if (unsigned long expectedNDim = shape.size();
        (unsigned long)tensor.ndim() != expectedNDim) {
      throw std::invalid_argument(
          "Input tensor has incompatible number of dimensions. Expected "
          "number of dimensions to be " +
          std::to_string(expectedNDim) + " but input has " +
          std::to_string(tensor.ndim()) + " dimensions.");
    }

    // Assert same shape
    for (ssize_t d = 0, ndim = tensor.ndim(); d < ndim; ++d) {
      auto expectedDimOr = shape.at(d);
      if (expectedDimOr && *expectedDimOr != tensor.shape(d)) {
        throw std::invalid_argument(
            "Input tensor has incompatible dimension along axis " +
            std::to_string(d) + ". Expected " + std::to_string(*expectedDimOr) +
            " but found " + std::to_string(tensor.shape(d)) + ".");
      }
    }
  }
}

/// Borrows the data from the array in the pybind object `obj` into a new
/// `Tensor` wrapped in an `AsyncValue` and sets its name to
/// `name`.
M_AsyncValue *core::processTensorInput(const py::object &obj,
                                       const std::string &name,
                                       M_RuntimeContext *context) {
  const auto tensor = obj.cast<py::array>();

  std::vector<int64_t> inputShape;
  for (ssize_t d = 0; d < tensor.ndim(); ++d)
    inputShape.push_back(tensor.shape(d));

  TensorSpecPtr tensorSpec{M_newTensorSpec(inputShape.data(), inputShape.size(),
                                           mapPybind11Type(tensor.dtype()),
                                           name.c_str()),
                           &M_freeTensorSpec};
  // tensor points to a buffer owned by the input py::array.
  // Since .execute() is synchronous currently, the tensor will survive at
  // least as long as we need it: we won't use the input tensor anymore
  // after .execute() returns.
  //
  // TODO(#7919): Figure out lifetime management for input tensors in the
  // asynchronous case.
  return M_createBorrowedTensor(tensor.data(), tensorSpec.get(), context);
}

/// Borrows the data from the array in the pybind object `obj` into a new
/// `Tensor` wrapped in an `AsyncValue`.
M_AsyncValue *core::processTensorInput(const py::object &obj,
                                       M_RuntimeContext *context) {
  std::string name("");
  return processTensorInput(obj, name, context);
}

/// Recursively converts the pybind object `pyObj` containing a value composed
/// of `array`, `dict`, `list`, `tuple`, `none`, `double`, `long`, `bool`
/// objects to a nested value of generic composed (`Dict`, `List`, Tuple`),
/// tensor (`Tensor`) and primitive types. Tensor data arrays are
/// borrowed and not copied.
M_AsyncValue *core::processValueInput(const py::object &pyObj,
                                      M_RuntimeContext *context,
                                      M_Status *status) {
  if (py::isinstance<py::array>(pyObj))
    return processTensorInput(pyObj, context);

  if (py::isinstance<py::dict>(pyObj)) {
    const auto pyDict = pyObj.cast<py::dict>();
    M_AsyncValue *dictValue = M_createDictAsyncValue(context);
    M_AsyncDict *dict = M_getDictFromValue(dictValue);
    for (auto elem : pyDict) {
      M_AsyncValue *key =
          processValueInput(elem.first.cast<py::object>(), context, status);
      M_AsyncValue *value =
          processValueInput(elem.second.cast<py::object>(), context, status);
      M_insertIntoDict(dict, key, value);
      M_freeValue(key);
      M_freeValue(value);
    }
    M_freeDict(dict);
    return dictValue;
  }

  if (py::isinstance<py::list>(pyObj)) {
    const auto pyList = pyObj.cast<py::list>();
    M_AsyncValue *listValue = M_createListAsyncValue(context);
    M_AsyncList *list = M_getListFromValue(listValue);
    for (auto elem : pyList) {
      M_AsyncValue *value =
          processValueInput(elem.cast<py::object>(), context, status);
      M_appendToList(list, value);
      M_freeValue(value);
    }
    M_freeList(list);
    return listValue;
  }

  if (py::isinstance<py::tuple>(pyObj)) {
    const auto pyTuple = pyObj.cast<py::tuple>();
    M_AsyncValue *tupleValue = M_createTupleAsyncValue(context);
    M_AsyncTuple *tuple = M_getTupleFromValue(tupleValue);
    for (auto elem : pyTuple) {
      M_AsyncValue *value =
          processValueInput(elem.cast<py::object>(), context, status);
      M_borrowIntoTuple(tuple, value);
      M_freeValue(value);
    }
    M_freeTuple(tuple);
    return tupleValue;
  }

  if (py::isinstance<py::none>(pyObj))
    return M_createNoneAsyncValue(context);

  if (py::isinstance<py::str>(pyObj)) {
    std::string str = pyObj.cast<std::string>();
    return M_createStringAsyncValue(str.c_str(), context);
  }

  if (py::isinstance<py::float_>(pyObj))
    return M_createDoubleAsyncValue(pyObj.cast<double>(), context);

  if (py::isinstance<py::int_>(pyObj))
    return M_createLongAsyncValue(pyObj.cast<int64_t>(), context);

  if (py::isinstance<py::bool_>(pyObj))
    return M_createBoolAsyncValue(pyObj.cast<bool>(), context);

  return nullptr;
}
