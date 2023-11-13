//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/BenchmarkCLI/InputGenerationSchema.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/YAMLTraits.h"

using namespace M;
using namespace M::BenchmarkCLI;

void llvm::yaml::MappingTraits<InputGenerationRandomUniform>::mapping(
    llvm::yaml::IO &io, InputGenerationRandomUniform &uniform) {
  io.mapOptional("min", uniform.min);
  io.mapOptional("max", uniform.max);
}

void llvm::yaml::MappingTraits<InputGenerationRandomNormal>::mapping(
    llvm::yaml::IO &io, InputGenerationRandomNormal &normal) {
  io.mapOptional("mean", normal.mean);
  io.mapOptional("stdev", normal.stdev);
}

void llvm::yaml::MappingTraits<InputGenerationRandom>::mapping(
    llvm::yaml::IO &io, InputGenerationRandom &random) {
  io.mapOptional("uniform", random.uniform);
  io.mapOptional("normal", random.normal);
}

void llvm::yaml::MappingTraits<InputGenerationData>::mapping(
    llvm::yaml::IO &io, InputGenerationData &data) {
  io.mapOptional("random", data.random);
  io.mapOptional("constant", data.constant);
}

void llvm::yaml::MappingTraits<InputGenerationInput>::mapping(
    llvm::yaml::IO &io, InputGenerationInput &input) {
  io.mapOptional("input_name", input.inputName);
  io.mapOptional("input_index", input.inputIndex);
  io.mapOptional("shape", input.shape);
  // TODO(akirchhoff): Batching specification
  io.mapOptional("data", input.data);
}

void llvm::yaml::MappingTraits<InputGenerationSchema>::mapping(
    llvm::yaml::IO &io, InputGenerationSchema &schema) {
  io.mapOptional("inputs", schema.inputs);
}

ErrorOr<InputGenerationSchema>
BenchmarkCLI::readInputGenerationSchema(StringRef path) {
  InputGenerationSchema schema;
  std::string error;
  auto buffer = mlir::openInputFile(path, &error);
  if (!buffer)
    return Error(std::move(error));
  llvm::yaml::Input yin(*buffer);
  yin >> schema;
  if (yin.error())
    return Error("YAML parse error");
  return std::move(schema);
}

bool BenchmarkCLI::roundTripInputGenerationSchema(StringRef path) {
  auto schemaOr = readInputGenerationSchema(path);
  if (schemaOr.isError()) {
    llvm::errs() << schemaOr.getError() << '\n';
    return false;
  }
  InputGenerationSchema schema = std::move(*schemaOr);
  // TODO(akirchhoff): Validate
  llvm::outs() << "Input generation schema is valid\n";
  {
    llvm::outs() << "Round-tripped input generation schema:\n";
    llvm::yaml::Output yout(llvm::outs());
    yout << schema;
  }
  return true;
}
