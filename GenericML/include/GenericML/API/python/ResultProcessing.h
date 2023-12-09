//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_API_PYTHON_RESULTPROCESSING_H
#define GENERICML_API_PYTHON_RESULTPROCESSING_H

#include "GenericML/API/modular/c/value.h"
#include "GenericML/API/modular/cpp/aliases.h"
#include "GenericML/API/python/Types.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include <vector>

using namespace M;

namespace py = pybind11;

namespace core {

/// Creates a new `py::array` by copying the tensor data from a `GML::Tensor`.
py::object processTensorResult(TensorPtr outTensor) {
  TensorSpecPtr tensorSpec{M_getTensorSpec(outTensor.get()), &M_freeTensorSpec};
  size_t numDims = M_getRank(tensorSpec.get());
  std::vector<int64_t> shape;
  for (size_t d = 0; d < numDims; ++d)
    shape.push_back(M_getDimAt(tensorSpec.get(), d));

  // TODO(8114): Currently always copy out.
  // The underlying M::TensorMap -> GML::Tensor -> M::BufferRef is borrowed,
  // so we cannot release its buffer to Python.

  size_t numElements = M_getTensorNumElements(outTensor.get());
  M_Dtype type = M_getTensorType(outTensor.get());
  size_t outputTensorSizeInBytes = M_sizeOf(type) * numElements;
  const void *outTensorSrcBuffer = M_getTensorData(outTensor.get());
  // `outTensorDestBuffer` is the underlying buffer backing the ouput tensor
  //  returned to Python. This will be garbage collected by Python as per
  // usual.
  void *outTensorDestBuffer = ::operator new(outputTensorSizeInBytes);
  std::memcpy(outTensorDestBuffer, outTensorSrcBuffer, outputTensorSizeInBytes);
  py::array outNdArray =
      py::array(mapDType(type), shape, /*strides=*/{}, outTensorDestBuffer);
  return outNdArray;
}

} // namespace core

#endif // GENERICML_API_PYTHON_RESULTPROCESSING_H
