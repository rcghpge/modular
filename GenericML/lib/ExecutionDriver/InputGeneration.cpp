//===- InputGeneration.cpp ------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "DriverInternals.h"

#include "GenericML/Kernels.h"
#include "GenericML/Support/Model.h"
#include "GenericML/Support/Tensor.h"
#include "LLCL/ForwardDecls.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

static ErrorOrSuccess fillWithElements(Tensor &tensor,
                                       InputGenerationKind inputGenKind,
                                       const QuantSpec &quantSpec) {
  uint8_t elementBuffer[TensorEltType::kMaxElementSizeInBytes];
  auto eltType = tensor.getEltType();

  switch (inputGenKind) {
  case InputGenerationKind::kRandom:
    return Kernel::fillRandom(tensor.getPointer(),
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
      if (tensor.getEltType() != TensorEltType::ui8 &&
          tensor.getEltType() != TensorEltType::si8)
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
  return Kernel::fillHomogeneous(tensor.getPointer(),
                                 tensor.getShape().getNumElements(), eltType,
                                 elementBuffer);
}

M::ErrorOr<SmallVector<Tensor, 3>>
M::initializeInputMemory(InputGenerationKind inputGenKind, const Model *model,
                         Runtime &runtime) {
  SmallVector<Tensor, 3> inputTensors;

  for (const auto &spec : model->getInputSpecs()) {
    // Allocate the tensor.
    Tensor input = Tensor::getUninit(spec, runtime);
    if (auto err = fillWithElements(input, inputGenKind, spec.getQuantSpec()))
      return err.takeError();

    inputTensors.push_back(std::move(input));
  }
  return inputTensors;
}
