//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "DriverInternals.h"

#include "GenericML/Support/Model.h"
#include "GenericML/Support/Tensor.h"
#include "Kernels/Misc.h"
#include "LLCL/ForwardDecls.h"
#include "Support/ErrorOr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

static ErrorOrSuccess fillWithElements(Tensor &tensor,
                                       InputGenerationKind inputGenKind,
                                       const QuantSpec &quantSpec) {
  uint8_t elementBuffer[DType::kMaxElementSizeInBytes];
  auto eltType = tensor.getEltType();

  switch (inputGenKind) {
  case InputGenerationKind::kRandom:
    return Kernel::fillRandom(tensor.getMutableBuffer(),
                              tensor.getShape().getNumElements(), eltType);

  case InputGenerationKind::kOnes: {
    ErrorOrSuccess result = Kernel::getScalarOne(elementBuffer, eltType);
    if (failed(result))
      return result;
    break;
  }
  case InputGenerationKind::kNegOnes: {
    ErrorOrSuccess result =
        Kernel::getScalarNegativeOne(elementBuffer, eltType);
    if (failed(result))
      return result;
    break;
  }
  case InputGenerationKind::kZeros:
    if (quantSpec.isQuantized()) {
      // FIXME: we need to handle all types, and refactor this into a kernel
      if (tensor.getEltType() != DType::ui8 &&
          tensor.getEltType() != DType::si8)
        return Error("when filling quantized tensors with zeros, only 8 bit "
                     "storage types are supported");
      auto params = quantSpec.getParams();
      if (quantSpec.getKind() == QuantSpec::Kind::uniformPerAxis &&
          params.size() != 1)
        return Error("when filling quantized tensors with zeros, only a single "
                     "scale/zeroPoint is supported");
      memset(elementBuffer, (uint8_t)params[0].zeroPoint,
             sizeof(elementBuffer));
    } else
      memset(elementBuffer, 0, sizeof(elementBuffer));
    break;
  }

  // Fill the tensor with the 0, 1, -1 value.
  return Kernel::fillHomogeneous(tensor.getMutableBuffer(),
                                 tensor.getShape().getNumElements(), eltType,
                                 elementBuffer);
}

M::ErrorOr<SmallVector<Tensor, 3>>
M::initializeInputMemory(InputGenerationKind inputGenKind,
                         ArrayRef<TensorShape> userInputShapes,
                         const Model *model, Runtime &runtime) {
  SmallVector<TensorShape> inputShapes(userInputShapes.begin(),
                                       userInputShapes.end());
  if (!inputShapes.empty()) {
    if (inputShapes.size() != model->getInputSpecs().size()) {
      return Error(Twine("Number of specified input shapes (") +
                   Twine(inputShapes.size()) +
                   Twine(") doesn't match number of model inputs (") +
                   Twine(model->getInputSpecs().size()) + Twine(")"));
    }
  } else {
    auto modelSpecs = model->getInputSpecs();
    inputShapes.append(modelSpecs.begin(), modelSpecs.end());
  }

  SmallVector<Tensor, 3> inputTensors;
  for (const auto &[userShape, modelSpec] :
       llvm::zip(inputShapes, model->getInputSpecs())) {
    if (userShape.getRank() != modelSpec.getRank())
      return Error(Twine("Specified shape ") + Twine(userShape.getAsString()) +
                   Twine(" doesn't match the rank of model input shape ") +
                   Twine(modelSpec.getAsString()));
    for (const auto &[userDim, modelDim] :
         llvm::zip(userShape.getDims(), modelSpec.getDims())) {
      if (mlir::ShapedType::isDynamic(modelDim)) {
        if (mlir::ShapedType::isDynamic(userDim))
          return Error(
              Twine("Must specify a fixed input shape for model input shape ") +
              Twine(modelSpec.getAsString()));
      } else if (userDim != modelDim) {
        return Error(Twine("Specified dimension ") + Twine(userDim) +
                     Twine(" doesn't match model input dimension ") +
                     Twine(modelDim) + Twine(" for shape ") +
                     Twine(modelSpec.getAsString()));
      }
    }

    // Allocate the tensor.
    Tensor input = Tensor::createOwned({userShape, modelSpec.getEltType()},
                                       runtime, kPreferredMemoryAlignment);
    RETURN_ERROR(
        fillWithElements(input, inputGenKind, modelSpec.getQuantSpec()));
    inputTensors.push_back(std::move(input));
  }
  return inputTensors;
}
