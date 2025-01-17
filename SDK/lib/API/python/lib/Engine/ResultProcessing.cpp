//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SDK/EngineAPI/python/ResultProcessing.h"
#include "GenericML/API/max/c/internal/value.h"
#include "GenericML/API/max/c/value.h"
#include "GenericML/API/max/cpp/aliases.h"
#include "SDK/EngineAPI/python/MojoValue.h"
#include "SDK/EngineAPI/python/Types.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include <vector>

using namespace M;

namespace nb = nanobind;

/// Creates a new `nb::ndarray` by copying the tensor data from a `Tensor`.
nb::object core::processTensorResult(TensorPtr outTensor) {
  TensorSpecPtr tensorSpec{M_getTensorSpec(outTensor.get()), &M_freeTensorSpec};
  int64_t numDims = M_getRank(tensorSpec.get());
  if (numDims == M_getDynamicRankValue())
    throw std::invalid_argument("dynamic ranked tensor is not supported yet.");

  std::vector<size_t> shape;
  for (size_t d = 0; d < (size_t)numDims; ++d)
    shape.push_back(static_cast<size_t>(M_getDimAt(tensorSpec.get(), d)));

  // TODO(8114): Currently always copy out.
  // The underlying M::TensorMap -> Tensor -> M::BufferRef is
  // borrowed, so we cannot release its buffer to Python.

  size_t numElements = M_getTensorNumElements(outTensor.get());
  M_Dtype type = M_getTensorType(outTensor.get());
  size_t outputTensorSizeInBytes = M_sizeOf(type) * numElements;
  const void *outTensorSrcBuffer = M_getTensorData(outTensor.get());
  // `outTensorDestBuffer` is the underlying buffer backing the ouput tensor
  //  returned to Python. This will be garbage collected by Python as per
  // usual.
  void *outTensorDestBuffer = ::operator new(outputTensorSizeInBytes);
  std::memcpy(outTensorDestBuffer, outTensorSrcBuffer, outputTensorSizeInBytes);
  return nb::ndarray<nb::numpy>(
             /*data=*/outTensorDestBuffer, /*ndim=*/(size_t)numDims,
             /*shape=*/shape.data(), /*owner=*/{}, /*strides=*/nullptr,
             /*dtype=*/mapDType(type))
      .cast();
}
