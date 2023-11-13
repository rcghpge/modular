//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_BENCHMARK_CLI_INPUT_GENERATION_SCHEMA_H
#define GENERICML_BENCHMARK_CLI_INPUT_GENERATION_SCHEMA_H

#include "Support/LLVMForwardDecls.h"
#include "Support/LLVMYAMLForwardDecls.h"
#include "Support/ML/TensorSpec.h"
#include <optional>
#include <string>
#include <vector>

namespace M::BenchmarkCLI {

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

struct InputGenerationData {
  std::optional<InputGenerationRandom> random;
  std::optional<std::string> constant;
  // TODO(akirchhoff): File input
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
bool roundTripInputGenerationSchema(StringRef path);

} // namespace M::BenchmarkCLI

LLVM_FWD_YAML_DECLARE_MAPPING_TRAITS(
    M::BenchmarkCLI::InputGenerationRandomUniform)
LLVM_FWD_YAML_DECLARE_MAPPING_TRAITS(
    M::BenchmarkCLI::InputGenerationRandomNormal)
LLVM_FWD_YAML_DECLARE_MAPPING_TRAITS(M::BenchmarkCLI::InputGenerationRandom)
LLVM_FWD_YAML_DECLARE_MAPPING_TRAITS(M::BenchmarkCLI::InputGenerationData)
LLVM_FWD_YAML_DECLARE_MAPPING_TRAITS(M::BenchmarkCLI::InputGenerationInput)
LLVM_FWD_YAML_IS_SEQUENCE_VECTOR(M::BenchmarkCLI::InputGenerationInput)
LLVM_FWD_YAML_DECLARE_MAPPING_TRAITS(M::BenchmarkCLI::InputGenerationSchema)

#endif // GENERICML_BENCHMARK_CLI_INPUT_GENERATION_SCHEMA_H
