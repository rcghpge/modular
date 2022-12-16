//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/DriverInternals.h"
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

static ErrorOr<Tensor> allocateInput(const TensorShape &userShape,
                                     const QuantTensorSpec &modelSpec,
                                     Runtime &runtime) {
  if (userShape.getRank() != modelSpec.getRank())
    return Error(Twine("Specified shape ") + userShape.getAsString() +
                 " doesn't match the rank of model input shape " +
                 modelSpec.getAsString());
  for (const auto &[userDim, modelDim] :
       llvm::zip(userShape.getDims(), modelSpec.getDims())) {
    if (mlir::ShapedType::isDynamic(modelDim)) {
      if (mlir::ShapedType::isDynamic(userDim))
        return Error(
            Twine("Must specify a fixed input shape for model input shape ") +
            modelSpec.getAsString());
    } else if (userDim != modelDim) {
      return Error(Twine("Specified dimension ") + Twine(userDim) +
                   " doesn't match model input dimension " + Twine(modelDim) +
                   " for shape " + modelSpec.getAsString());
    }
  }

  return Tensor::createOwned({userShape, modelSpec.getEltType()}, runtime,
                             kPreferredMemoryAlignment);
}

static ErrorOrSuccess fillInput(InputGenerationKind inputGenKind,
                                const QuantTensorSpec &modelSpec,
                                Tensor &tensor) {
  RETURN_ERROR(
      fillWithElements(tensor, inputGenKind, modelSpec.getQuantSpec()));
  return success();
}

static ErrorOr<Tensor> allocateAndFillInput(InputGenerationKind inputGenKind,
                                            const TensorShape &userShape,
                                            const QuantTensorSpec &modelSpec,
                                            Runtime &runtime) {
  UNWRAP_ERROR(result, allocateInput(userShape, modelSpec, runtime));
  RETURN_ERROR(fillInput(inputGenKind, modelSpec, result));
  return result;
}

M::ErrorOr<SmallVector<Tensor, 3>> M::initializeInputMemory(
    InputGenerationKind inputGenKind, ArrayRef<TensorShape> userInputShapes,
    ArrayRef<InputDataSpec> inputData, const Model *model, Runtime &runtime) {
  SmallVector<TensorShape> inputShapes(userInputShapes.begin(),
                                       userInputShapes.end());
  auto numInputs = model->getInputSpecs().size();
  if (!inputShapes.empty()) {
    if (inputShapes.size() != numInputs) {
      return Error(Twine("Number of specified input shapes (") +
                   Twine(inputShapes.size()) +
                   ") doesn't match number of model inputs (" +
                   Twine(numInputs) + ")");
    }
  } else {
    auto modelSpecs = model->getInputSpecs();
    inputShapes.append(modelSpecs.begin(), modelSpecs.end());
  }

  for (const auto &inputData : inputData) {
    if (inputData.first >= numInputs) {
      return Error(
          Twine("Tensor index (") + Twine(inputData.first) +
          ") passed to --input-data larger than number of inputs in model (" +
          Twine(numInputs) + ").");
    }
  }

  SmallVector<Tensor, 3> inputTensors;
  for (size_t i = 0; i < numInputs; i++) {
    const auto &userShape = inputShapes[i];
    const auto &modelSpec = model->getInputSpecs()[i];
    auto inputDataIt =
        std::find_if(inputData.begin(), inputData.end(),
                     [=](const auto &v) { return v.first == i; });
    // if per tensor input-data is specified, use it instead of the default from
    // --model-inputs
    if (inputDataIt != inputData.end())
      inputGenKind = std::get<InputGenerationKind>(inputDataIt->second);
    UNWRAP_ERROR(result, allocateAndFillInput(inputGenKind, userShape,
                                              modelSpec, runtime));
    inputTensors.push_back(std::move(result));
  }
  return inputTensors;
}
