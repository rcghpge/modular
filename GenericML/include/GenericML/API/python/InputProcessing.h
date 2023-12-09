//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_API_PYTHON_INPUTPROCESSING_H
#define GENERICML_API_PYTHON_INPUTPROCESSING_H

#include "modular/c/value.h"
#include "modular/cpp/aliases.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "python/TensorSpec.h"
#include "python/Types.h"
#include "llvm/ADT/StringRef.h"
#include <vector>

using namespace M;

namespace py = pybind11;

namespace core {

/// Verifies that the array in the pybind object `obj` has the same shape and
/// data type as the tensor spec.
void verifyTensorInputSpecs(const py::object &obj,
                            const TensorSpec &inputTensorSpec) {
  const auto tensor = obj.cast<py::array>();

  // Assert the input tensor has the same tensor specification
  // Assert same dtype.
  if (pybind11::dtype expectedDtype = mapDType(inputTensorSpec.getDtype());
      !tensor.dtype().is(expectedDtype))
    throw std::runtime_error(
        "Input tensor has incompatible type. Expected type " +
        printDType(inputTensorSpec.getDtype()) + " but got " +
        printDType(mapPybind11Type(tensor.dtype())) + " instead.");

  // Assert same number of dims
  if (unsigned long expectedNDim = inputTensorSpec.getShape().size();
      (unsigned long)tensor.ndim() != expectedNDim) {
    throw std::runtime_error(
        "Input tensor has incompatible number of dimensions. Expected "
        "number of dimensions to be " +
        std::to_string(expectedNDim) + " but input has " +
        std::to_string(tensor.ndim()) + " dimensions.");
  }

  // Assert same shape
  std::vector<int64_t> inputShape;
  for (ssize_t d = 0, ndim = tensor.ndim(); d < ndim; ++d) {
    int64_t expectedDim = M_getDimAt(inputTensorSpec.specPtr.get(), d);
    if (expectedDim != apiDynamicDimension && expectedDim != tensor.shape(d)) {
      throw std::runtime_error(
          "Input tensor has incompatible dimension along axis " +
          std::to_string(d) + ". Expected " + std::to_string(expectedDim) +
          " but found " + std::to_string(tensor.shape(d)) + ".");
    }
  }
}

/// Borrows the data from the array in the pybind object `obj` into a new
/// `GML::Tensor` wrapped in an `AsyncValue` and sets it's name to `name`.
M_AsyncValue *processTensorInput(const py::object &obj, const std::string &name,
                                 M_RuntimeContext *context, M_Status *status) {
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
  if (M_isError(status))
    throw std::runtime_error(M_getError(status));
}

/// Borrows the data from the array in the pybind object `obj` into a new
/// `GML::Tensor` wrapped in an `AsyncValue`.
M_AsyncValue *processTensorInput(const py::object &obj,
                                 M_RuntimeContext *context, M_Status *status) {
  std::string name("");
  return processTensorInput(obj, name, context, status);
}

} // namespace core

#endif // GENERICML_API_PYTHON_INPUTPROCESSING_H
