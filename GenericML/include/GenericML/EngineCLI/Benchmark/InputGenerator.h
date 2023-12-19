//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_ENGINECLI_BENCHMARK_INPUT_GENERATOR_H
#define GENERICML_ENGINECLI_BENCHMARK_INPUT_GENERATOR_H

#include "Support/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ML/DType.h"
#include "modular/c/types.h"
#include "llvm/ADT/Twine.h"
#include <memory>

namespace M {
class TensorShape;
} // namespace M

namespace M::EngineCLI {
struct InputGenerationRandomUniform;
struct InputGenerationRandomNormal;
struct InputGenerationRandom;
struct InputGenerationFile;
struct InputGenerationData;
struct InputGenerationInput;
struct InputGenerationSchema;
} // namespace M::EngineCLI

namespace M::BenchmarkCLI {

// TODO: Reparent BenchmarkCLI so this using is no longer necessary.
using namespace M::EngineCLI;

class Tensor;

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

std::unique_ptr<InputRandomDistribution>
createUniformInputRandomDistribution(DType dtype,
                                     const InputGenerationRandomUniform &spec,
                                     InputSchemaErrorReporter &errorReporter);

std::unique_ptr<InputRandomDistribution>
createNormalInputRandomDistribution(DType dtype,
                                    const InputGenerationRandomNormal &spec,
                                    InputSchemaErrorReporter &errorReporter);

std::unique_ptr<InputRandomDistribution>
createInputRandomDistribution(DType dtype, const InputGenerationRandom &spec,
                              InputSchemaErrorReporter &errorReporter);

class InputDataGenerator {
public:
  virtual ~InputDataGenerator() = default;

  virtual ssize_t getNumQueries() const = 0;
  // TODO(akirchhoff): This should be revised when we have a better batching
  // story
  virtual ErrorOr<Tensor> generate(size_t queryIndex,
                                   const TensorShape &shape) const = 0;
};

std::unique_ptr<InputDataGenerator> createRandomInputDataGenerator(
    std::unique_ptr<InputRandomDistribution> distribution);

std::unique_ptr<InputDataGenerator>
createRandomInputDataGenerator(DType dtype, const InputGenerationRandom &spec,
                               InputSchemaErrorReporter &errorReporter);

std::unique_ptr<InputDataGenerator>
createConstantInputDataGenerator(DType dtype, StringRef constant,
                                 InputSchemaErrorReporter &errorReporter);

std::unique_ptr<InputDataGenerator>
createFileInputDataGenerator(std::optional<DType> dtype,
                             const InputGenerationFile &spec,
                             InputSchemaErrorReporter &errorReporter);

std::unique_ptr<InputDataGenerator>
createInputDataGenerator(std::optional<DType> dtype,
                         const InputGenerationData &spec,
                         InputSchemaErrorReporter &errorReporter);

class GeneratedInput {
public:
  virtual ~GeneratedInput() = default;

  virtual void borrowInto(M_AsyncTensorMap *atm, M_Status *status) const = 0;
};

class InputGenerator {
public:
  virtual ~InputGenerator() = default;

  virtual ssize_t getNumQueries() const = 0;
  virtual ErrorOr<std::unique_ptr<GeneratedInput>>
  generateInput(size_t queryIndex) = 0;
};

std::unique_ptr<InputGenerator>
createInputGenerator(const InputGenerationInput &spec,
                     InputSchemaErrorReporter &errorReporter);

class GeneratedQuery {
public:
  virtual ~GeneratedQuery() = default;

  virtual M_AsyncTensorMap *getAsyncTensorMap() const = 0;
};

class QueryGenerator {
public:
  virtual ~QueryGenerator() = default;

  // Returns -1 if there is no fixed size and any number of queries can be
  // generated.
  virtual ssize_t getNumQueries() const = 0;
  virtual ErrorOr<std::unique_ptr<GeneratedQuery>>
  generateQuery(M_RuntimeContext *runtimeContext, size_t queryIndex) const = 0;
};

std::unique_ptr<QueryGenerator> createQueryGenerator(
    std::vector<std::unique_ptr<InputGenerator>> inputGenerators,
    InputSchemaErrorReporter &errorReporter);

std::unique_ptr<QueryGenerator>
createQueryGenerator(const InputGenerationSchema &schema,
                     InputSchemaErrorReporter &errorReporter);

} // namespace M::BenchmarkCLI

#endif // GENERICML_ENGINECLI_BENCHMARK_INPUT_GENERATOR_H
