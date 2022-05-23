//===- TFLInputGeneration.cpp ---------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "TFLite/Tools/TFLInputGeneration.h"
#include "GenericML/Kernels.h"
#include "GenericML/Support/TensorDType.h"
#include "Support/Error.h"
#include "TFLite/Support/TFLiteType.h"
#include "TFLite/TFLRuntimeModel.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

static ErrorOrSuccess fillWithElements(InputGenerationKind inputGenKind,
                                       void *ptr, size_t numElements,
                                       TensorDType dtype) {

  if (dtype.isInvalid())
    return Error("unspecified element type to initialize");

  uint8_t elementBuffer[TensorDType::kMaxElementSizeInBytes];

  switch (inputGenKind) {
  case InputGenerationKind::kNone:
    return Error("");
  case InputGenerationKind::kRandom:
    return Kernel::fillRandom(ptr, numElements, dtype);

  case InputGenerationKind::kOnes: {
    ErrorOrSuccess result = Kernel::getScalarOne(elementBuffer, dtype);
    if (failed(result))
      return result;
    break;
  }
  case InputGenerationKind::kNegOnes: {
    ErrorOrSuccess result = Kernel::getScalarNegativeOne(elementBuffer, dtype);
    if (failed(result))
      return result;
    break;
  }
  case InputGenerationKind::kZeros:
    memset(elementBuffer, 0, sizeof(elementBuffer));
    break;
  }

  // Fill the tensor with the 0, 1, -1 value.
  return Kernel::fillHomogeneous(ptr, numElements, dtype, elementBuffer);
}

ErrorOr<SmallVector<void *, 3>>
M::initializeInputMemory(InputGenerationKind inputGenKind,
                         const TFLRuntimeModel &model) {
  SmallVector<void *, 3> inputTensorData;
  for (const auto &[shape, eltType] :
       llvm::zip(model.getInputShapes(), model.getInputElementTypes())) {

    // Do the malloc.
    ssize_t size = eltType.getSizeInBytes(shape.getNumElements());
    assert(size != -1);
    inputTensorData.push_back(malloc(size));

    // Do the fill
    size_t numElements = shape.getNumElements();
    auto err = fillWithElements(inputGenKind, inputTensorData.back(),
                                numElements, eltType);

    if (failed(err)) {
      for (auto t : inputTensorData)
        free(t);
      return err.takeError();
    }
  }
  return inputTensorData;
}
