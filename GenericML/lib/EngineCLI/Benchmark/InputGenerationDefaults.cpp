//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/EngineCLI/Benchmark/InputGenerationDefaults.h"
#include "GenericML/EngineCLI/Common/APIBridge.h"
#include "GenericML/EngineCLI/Common/InputGenerationSchema.h"
#include "modular/cpp/aliases.h"

using namespace M;
using namespace M::EngineCLI;
using namespace M::BenchmarkCLI;

void M::BenchmarkCLI::applyInputGenerationDefaults(
    InputGenerationSchema &schema, const M_AsyncCompiledModel *model,
    M_Status *status) {
  applyInputListDefaults(schema, model, status);
  if (schema.inputs) {
    for (auto &inputSpec : *schema.inputs) {
      applyInputShapeDefaults(inputSpec, model, status);
      applyInputDataDefaults(inputSpec);
    }
  }
}

void M::BenchmarkCLI::applyInputListDefaults(InputGenerationSchema &schema,
                                             const M_AsyncCompiledModel *model,
                                             M_Status *status) {
  if (schema.inputs)
    return;
  TensorNamesPtr tensorNames(M_getInputNames(model, status),
                             M_freeTensorNameArray);
  schema.inputs.emplace();
  const char *inputName;
  for (size_t index = 0;
       (inputName = M_getTensorNameAt(tensorNames.get(), index)); ++index) {
    InputGenerationInput input;
    input.inputName = inputName;
    schema.inputs->emplace_back(std::move(input));
  }
}

void M::BenchmarkCLI::applyInputShapeDefaults(InputGenerationInput &inputSpec,
                                              const M_AsyncCompiledModel *model,
                                              M_Status *status) {
  if (inputSpec.shape)
    return;
  if (!inputSpec.inputName)
    return;
  TensorSpecPtr capiTensorSpec(
      M_getModelInputSpecByName(model, inputSpec.inputName->c_str(), status),
      M_freeTensorSpec);
  if (!capiTensorSpec)
    return;
  inputSpec.shape = capiToTensorSpec(capiTensorSpec.get());
}

void M::BenchmarkCLI::applyInputDataDefaults(InputGenerationInput &inputSpec) {
  if (inputSpec.data)
    return;
  inputSpec.data = InputGenerationData{};
  inputSpec.data->random = InputGenerationRandom{};
  inputSpec.data->random->uniform = InputGenerationRandomUniform{};
}
