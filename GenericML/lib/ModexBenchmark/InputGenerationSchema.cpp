//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/ModexBenchmark/InputGenerationSchema.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/YAMLTraits.h"

using namespace M;
using namespace M::ModexBench;

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

bool ModexBench::roundTripInputGenerationSchema(StringRef path) {
  InputGenerationSchema schema;
  {
    std::string error;
    auto buffer = mlir::openInputFile(path, &error);
    if (!buffer) {
      llvm::errs() << error;
      return false;
    }
    llvm::yaml::Input yin(*buffer);
    yin >> schema;
    if (yin.error())
      return false;
  }
  // TODO(akirchhoff): Validate
  llvm::outs() << "Input generation schema is valid\n";
  {
    llvm::outs() << "Round-tripped input generation schema:\n";
    llvm::yaml::Output yout(llvm::outs());
    yout << schema;
  }
  return true;
}
