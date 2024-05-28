//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SDK/EngineCLI/Benchmark/InputGenerationDefaults.h"
#include "SDK/EngineCLI/Common/APIBridge.h"
#include "SDK/EngineCLI/Common/InputGenerationSchema.h"
#include "Support/AssertStream.h"
#include "max/cpp/aliases.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

using namespace M;
using namespace M::EngineCLI;
using namespace M::BenchmarkCLI;

static bool shouldOverwriteBatch(const M_AsyncCompiledModel *model,
                                 M_Status *status) {
  TensorNamesPtr names(M_getInputNames(model, status), M_freeTensorNameArray);
  if (!names)
    return false;
  const char *name;
  for (size_t i = 0; (name = M_getTensorNameAt(names.get(), i)); ++i) {
    TensorSpecPtr spec(M_getModelInputSpecByName(model, name, status),
                       M_freeTensorSpec);
    if (!spec)
      return false;
    auto rank = M_getRank(spec.get());
    if (rank == M_getDynamicRankValue() || rank < 1)
      return false;
    if (M_getDimAt(spec.get(), 0) != M_getDynamicDimensionValue())
      return false;
  }
  return true;
}

void M::BenchmarkCLI::applyInputGenerationDefaults(
    InputGenerationSchema &schema, const M_AsyncCompiledModel *model,
    M_Status *status) {
  applyInputListDefaults(schema, model, status);
  if (schema.inputs) {
    bool overwriteBatch = shouldOverwriteBatch(model, status);
    for (auto &inputSpec : *schema.inputs) {
      applyInputShapeDefaults(inputSpec, model, overwriteBatch, status);
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
                                              bool overwriteBatch,
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
  if (overwriteBatch) {
    GENERICML_ASSERT(inputSpec.shape->hasRank());
    auto dims = inputSpec.shape->getDimsCopy();
    GENERICML_ASSERT(!dims.empty());
    GENERICML_ASSERT(mlir::ShapedType::isDynamic(dims[0]));
    dims[0] = 1;
    inputSpec.shape = TensorSpec(dims, inputSpec.shape->getEltType());
  }
}

void M::BenchmarkCLI::applyInputDataDefaults(InputGenerationInput &inputSpec) {
  if (inputSpec.data)
    return;
  inputSpec.data = InputGenerationData{};
  inputSpec.data->random = InputGenerationRandom{};
  inputSpec.data->random->uniform = InputGenerationRandomUniform{};
}
