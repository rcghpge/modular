//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MODEX_BENCHMARK_INPUT_GENERATOR_H
#define GENERICML_MODEX_BENCHMARK_INPUT_GENERATOR_H

#include "GenericML/API/modular/c/types.h"
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

class GeneratedInput {
public:
  virtual ~GeneratedInput() = default;

  virtual void borrowInto(M_AsyncTensorMap *atm, M_Status *status) const = 0;
};

class InputGenerator {
public:
  virtual ~InputGenerator() = default;

  // TODO: This LLCL::CompactRuntimePtr is used only for access to an
  // allocator.  This is kind of crappy and will make the benchmark tool harder
  // to open-source later.  Should find a way to get rid of the requirement for
  // it later.
  virtual ErrorOr<std::unique_ptr<GeneratedInput>>
  generateInput(LLCL::CompactRuntimePtr runtime, size_t queryIndex) = 0;
};

struct InputGenerationInput;
std::unique_ptr<InputGenerator>
createInputGenerator(const InputGenerationInput &spec,
                     InputSchemaErrorReporter &errorReporter);

class GeneratedQuery {
public:
  virtual ~GeneratedQuery() = default;

  virtual M_AsyncTensorMap *getAsyncTensorMap() const;
};

class QueryGenerator {
public:
  virtual ~QueryGenerator() = default;

  // Returns -1 if there is no fixed size and any number of queries can be
  // generated.
  // TODO: Want to get rid of the allocRuntime parameter if we can eventually.
  virtual ssize_t getNumQueries() const = 0;
  virtual ErrorOr<std::unique_ptr<GeneratedQuery>>
  generateQuery(LLCL::CompactRuntimePtr allocRuntime,
                M_RuntimeContext *runtimeContext, size_t queryIndex) const = 0;
};

std::unique_ptr<QueryGenerator> createQueryGenerator(
    std::vector<std::unique_ptr<InputGenerator>> inputGenerators);

struct InputGenerationSchema;
std::unique_ptr<QueryGenerator>
createQueryGenerator(const InputGenerationSchema &schema,
                     InputSchemaErrorReporter &errorReporter);

} // namespace M::ModexBench

#endif // GENERICML_MODEX_BENCHMARK_INPUT_GENERATOR_H
