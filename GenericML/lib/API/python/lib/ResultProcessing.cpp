//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/API/python/ResultProcessing.h"
#include "GenericML/API/modular/c/value.h"
#include "GenericML/API/modular/cpp/aliases.h"
#include "PyTorch/API/modular/c/pytorch/types.h"
#include "PyTorch/API/modular/c/pytorch/value.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "python/Types.h"
#include <vector>

using namespace M;

namespace py = pybind11;

/// Creates a new `py::array` by copying the tensor data from a `GML::Tensor`.
py::object core::processTensorResult(TensorPtr outTensor) {
  TensorSpecPtr tensorSpec{M_getTensorSpec(outTensor.get()), &M_freeTensorSpec};
  int64_t numDims = M_getRank(tensorSpec.get());
  if (numDims == M_getDynamicRankValue())
    throw std::runtime_error("dynamic ranked tensor is not supported yet.");

  std::vector<int64_t> shape;
  for (size_t d = 0; d < (size_t)numDims; ++d)
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

/// Recursively converts the `value` containing nested generic
/// `Dict`, `List`, `Tuple`, `None`, `GML::Tensor` and primitive type values
/// to a nested pybind object value composed of `array`, `dict`, `list`,
/// `tuple`, `none`, `double`, `long` and `bool`. Tensor data arrays are copied.
py::object core::processValueResult(ValuePtr value) {
  switch (M_Torch_getValueType(value.get())) {
  case M_Torch_ValueType::M_TORCH_DICT_VALUE: {
    M_Torch_AsyncDict *dict = M_Torch_getDictFromValue(value.get());
    py::dict pyDict;
    size_t size = M_Torch_getDictSize(dict);
    for (size_t i = 0, e = M_Torch_getDictSize(dict); i < e; ++i) {
      for (size_t i = 0; i < size; ++i) {
        pyDict[processValueResult(
            ValuePtr(M_Torch_getDictKey(dict, i), M_freeValue))] =
            processValueResult(
                ValuePtr{M_Torch_getDictValue(dict, i), M_freeValue});
      }
    }
    M_Torch_freeDict(dict);
    return pyDict;
  }
  case M_Torch_ValueType::M_TORCH_LIST_VALUE: {
    M_Torch_AsyncList *list = M_Torch_getListFromValue(value.get());
    size_t size = M_Torch_getListSize(list);
    py::list pyList;
    for (size_t i = 0; i < size; ++i) {
      pyList.append(processValueResult(
          ValuePtr{M_Torch_getListValue(list, i), M_freeValue}));
    }
    M_Torch_freeList(list);
    return pyList;
  }
  case M_Torch_ValueType::M_TORCH_TUPLE_VALUE: {
    M_Torch_AsyncTuple *tuple = M_Torch_getTupleFromValue(value.get());
    size_t size = M_Torch_getTupleSize(tuple);
    py::tuple pyTuple(size);
    for (size_t i = 0; i < size; ++i) {
      pyTuple[i] = processValueResult(
          ValuePtr{M_Torch_getTupleValue(tuple, i), M_freeValue});
    }
    M_Torch_freeTuple(tuple);
    return pyTuple;
  }
  case M_Torch_ValueType::M_TORCH_NONE_VALUE:
    return py::none();
  case M_TORCH_STRING_VALUE:
    return py::cast(M_getStringFromValue(value.get()));
  case M_TORCH_LONG_VALUE:
    return py::cast(M_getLongFromValue(value.get()));
  case M_TORCH_DOUBLE_VALUE:
    return py::cast(M_getDoubleFromValue(value.get()));
  case M_TORCH_BOOL_VALUE:
    return py::cast(M_getBoolFromValue(value.get()));
  case M_Torch_ValueType::M_TORCH_TENSOR_VALUE: {
    TensorPtr tensorPtr{M_getTensorFromValue(value.get()), &M_freeTensor};
    return processTensorResult(std::move(tensorPtr));
  }
  default:
    throw std::runtime_error("Unhandled value type.");
  }
}
