//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/EngineCLI/Common/TorchSupport.h"
#include "GenericML/EngineCLI/Common/APIBridge.h"
#include "GenericML/EngineCLI/Common/InputGenerationSchema.h"
#include "PyTorch/API/modular/c/pytorch/config.h"
#include "Support/ErrorOr.h"
#include "modular/cpp/aliases.h"
#include "llvm/ADT/Twine.h"

using namespace M;

ErrorOrSuccess
M::EngineCLI::setTorchInputSpecs(M_CompileConfig *config,
                                 const InputGenerationSchema &schema) {
  if (!schema.inputs)
    return success();
  std::vector<TensorSpecPtr> specOwners;
  std::vector<M_TensorSpec *> specPointers;
  for (const auto &input : *schema.inputs) {
    if (!input.inputName || !input.shape)
      continue;
    TensorSpecPtr capiSpec =
        tensorSpecToCAPI(input.inputName->c_str(), *input.shape);
    if (!capiSpec)
      return Error(Twine("M_newTensorSpec failed for input ") +
                   *input.inputName + " (out of memory?)");
    specPointers.push_back(capiSpec.get());
    specOwners.push_back(std::move(capiSpec));
  }
  M_setTorchInputSpecs(config, specPointers.data(), specPointers.size());
  return success();
}
