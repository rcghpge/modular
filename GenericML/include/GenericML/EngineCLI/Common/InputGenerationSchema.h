//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_ENGINECLI_COMMON_INPUTGENERATIONSCHEMA_H
#define GENERICML_ENGINECLI_COMMON_INPUTGENERATIONSCHEMA_H

#include "Support/LLVMForwardDecls.h"
#include "Support/LLVMYAMLForwardDecls.h"
#include "Support/ML/TensorSpec.h"
#include <optional>
#include <string>
#include <vector>

namespace M::EngineCLI {

struct InputGenerationRandomUniform {
  std::optional<std::string> min;
  std::optional<std::string> max;
};

struct InputGenerationRandomNormal {
  std::optional<double> mean;
  std::optional<double> stdev;
};

struct InputGenerationRandom {
  std::optional<InputGenerationRandomUniform> uniform;
  std::optional<InputGenerationRandomNormal> normal;
};

struct InputGenerationFile {
  std::string path;
  std::optional<std::string> format;
};

struct InputGenerationData {
  std::optional<InputGenerationRandom> random;
  std::optional<std::string> constant;
  std::optional<InputGenerationFile> file;
  // TODO(akirchhoff): Record input
};

struct InputGenerationInput {
  std::optional<std::string> inputName;
  std::optional<int> inputIndex;
  std::optional<TensorSpec> shape;
  // TODO(akirchhoff): Batching specification
  std::optional<InputGenerationData> data;
};

struct InputGenerationSchema {
  std::optional<std::vector<InputGenerationInput>> inputs;
};

ErrorOr<InputGenerationSchema> readInputGenerationSchema(StringRef path);
void dumpInputGenerationSchema(const InputGenerationSchema &schema);

} // namespace M::EngineCLI

LLVM_FWD_YAML_DECLARE_MAPPING_TRAITS(M::EngineCLI::InputGenerationRandomUniform)
LLVM_FWD_YAML_DECLARE_MAPPING_TRAITS(M::EngineCLI::InputGenerationRandomNormal)
LLVM_FWD_YAML_DECLARE_MAPPING_TRAITS(M::EngineCLI::InputGenerationRandom)
LLVM_FWD_YAML_DECLARE_MAPPING_TRAITS(M::EngineCLI::InputGenerationFile)
LLVM_FWD_YAML_DECLARE_MAPPING_TRAITS(M::EngineCLI::InputGenerationData)
LLVM_FWD_YAML_DECLARE_MAPPING_TRAITS(M::EngineCLI::InputGenerationInput)
LLVM_FWD_YAML_IS_SEQUENCE_VECTOR(M::EngineCLI::InputGenerationInput)
LLVM_FWD_YAML_DECLARE_MAPPING_TRAITS(M::EngineCLI::InputGenerationSchema)

#endif // GENERICML_ENGINECLI_COMMON_INPUTGENERATIONSCHEMA_H
