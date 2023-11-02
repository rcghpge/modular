//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MODEX_BENCHMARK_INPUT_GENERATOR_H
#define GENERICML_MODEX_BENCHMARK_INPUT_GENERATOR_H

#include "GenericML/Support/Tensor.h"
#include "Support/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include <memory>

namespace M::ModexBench {

class InputSchemaErrorReporter {
public:
  virtual ~InputSchemaErrorReporter() = default;

  virtual void enterScope(StringRef scopeName) = 0;
  virtual void exitScope(StringRef scopeName) = 0;
  virtual void reportError(Twine errorDetail) = 0;
};

class InputRandomDistribution {
public:
  explicit InputRandomDistribution(DType dtype) : dtype(dtype) {}
  virtual ~InputRandomDistribution() = default;

  DType getDType() const { return dtype; }
  virtual void fill(void *out, size_t numElements) const = 0;

private:
  DType dtype;
};

struct InputGenerationRandomUniform;
std::unique_ptr<InputRandomDistribution>
createUniformInputRandomDistribution(DType dtype,
                                     const InputGenerationRandomUniform &spec,
                                     InputSchemaErrorReporter &errorReporter);

struct InputGenerationRandomNormal;
std::unique_ptr<InputRandomDistribution>
createNormalInputRandomDistribution(DType dtype,
                                    const InputGenerationRandomNormal &spec,
                                    InputSchemaErrorReporter &errorReporter);

struct InputGenerationRandom;
std::unique_ptr<InputRandomDistribution>
createInputRandomDistribution(DType dtype, const InputGenerationRandom &spec,
                              InputSchemaErrorReporter &errorReporter);

class InputDataGenerator {
public:
  virtual ~InputDataGenerator() = default;

  // TODO(akirchhoff): This should be revised when we have a better batching
  // story
  virtual ErrorOr<GML::Tensor> generate(LLCL::CompactRuntimePtr runtime,
                                        const TensorShape &shape) const = 0;
};

std::unique_ptr<InputDataGenerator> createRandomInputDataGenerator(
    std::unique_ptr<InputRandomDistribution> distribution);

struct InputGenerationRandom;
std::unique_ptr<InputDataGenerator>
createRandomInputDataGenerator(DType dtype, const InputGenerationRandom &spec,
                               InputSchemaErrorReporter &errorReporter);

std::unique_ptr<InputDataGenerator>
createConstantInputDataGenerator(DType dtype, StringRef constant,
                                 InputSchemaErrorReporter &errorReporter);

struct InputGenerationData;
std::unique_ptr<InputDataGenerator>
createInputDataGenerator(std::optional<DType> dtype,
                         const InputGenerationData &spec,
                         InputSchemaErrorReporter &errorReporter);

} // namespace M::ModexBench

#endif // GENERICML_MODEX_BENCHMARK_INPUT_GENERATOR_H
