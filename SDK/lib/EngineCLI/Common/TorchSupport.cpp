//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SDK/EngineCLI/Common/TorchSupport.h"
#include "PyTorch/API/max/c/pytorch/config.h"
#include "PyTorch/API/max/cpp/pytorch/aliases.h"
#include "SDK/EngineCLI/Common/APIBridge.h"
#include "SDK/EngineCLI/Common/InputGenerationSchema.h"
#include "Support/ErrorOr.h"
#include "max/cpp/aliases.h"
#include "llvm/ADT/Twine.h"

using namespace M;

ErrorOrSuccess
M::EngineCLI::setTorchInputSpecs(M_CompileConfig *config,
                                 const InputGenerationSchema &schema) {
  if (!schema.inputs)
    return success();
  std::vector<TorchInputSpecPtr> specOwners;
  std::vector<M_TorchInputSpec *> specPointers;
  for (const auto &input : *schema.inputs) {
    if (!input.inputName || (!input.shape && !input.compileShape))
      continue;
    TensorSpec compileShape =
        input.compileShape ? *input.compileShape : *input.shape;
    ErrorOr<TorchInputSpecPtr> capiSpecOr =
        tensorSpecToCAPITorchSpec(compileShape);
    if (capiSpecOr.isError())
      return Error(Twine("Failed to create Torch input spec for input ") +
                   *input.inputName + ": " + capiSpecOr.getError());
    specPointers.push_back(capiSpecOr->get());
    specOwners.push_back(std::move(*capiSpecOr));
  }
  M_setTorchInputSpecs(config, specPointers.data(), specPointers.size());
  return success();
}
