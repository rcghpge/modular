//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/BenchmarkCLI/InputGenerator.h"
#include "GenericML/API/modular/cpp/aliases.h"
#include "GenericML/BenchmarkCLI/InputGenerationSchema.h"
#include "Support/MathExtras.h"
#include <type_traits>

using namespace M;
using namespace M::BenchmarkCLI;

template <typename Fn>
static auto withinScope(InputSchemaErrorReporter &errorReporter,
                        StringRef scopeName, const Fn &fn) {
  errorReporter.enterScope(scopeName);
  auto result = fn();
  errorReporter.exitScope(scopeName);
  return std::move(result);
}

static bool parseElementValue(StringRef value, float &out,
                              InputSchemaErrorReporter &errorReporter) {
  double outD;
  bool ok = !value.getAsDouble(outD);
  if (ok)
    out = static_cast<float>(outD);
  else
    errorReporter.reportError("Unable to parse '" + llvm::Twine(value) +
                              "' as f32 value");
  return ok;
}

template <typename T>
static std::enable_if_t<
    std::is_integral_v<T> && DTypeForCXXType<T>::kind != DType::invalid, bool>
parseElementValue(StringRef value, T &out,
                  InputSchemaErrorReporter &errorReporter) {
  if (value.getAsInteger(0, out)) {
    errorReporter.reportError("Unable to parse '" + llvm::Twine(value) +
                              "' as " + DTypeForCXXType<T>::kind.getAsString() +
                              " value");
    return false;
  }
  return true;
}

template <typename T>
[[maybe_unused]] static bool
parseDefaultedElementValue(StringRef value, T &out, T defaultValue,
                           InputSchemaErrorReporter &errorReporter) {
  if (value.empty()) {
    out = defaultValue;
    return true;
  }
  return parseElementValue(value, out, errorReporter);
}

template <typename T>
static bool
parseDefaultedElementValue(const std::optional<std::string> &value, T &out,
                           T defaultValue,
                           InputSchemaErrorReporter &errorReporter) {
  if (!value.has_value() || value->empty()) {
    out = defaultValue;
    return true;
  }
  return parseElementValue(*value, out, errorReporter);
}

namespace M::BenchmarkCLI::Detail {

class UniformFloatInputRandomDistribution final
    : public InputRandomDistribution {
public:
  explicit UniformFloatInputRandomDistribution(DType dtype, float min,
                                               float max)
      : InputRandomDistribution(dtype), min(min), max(max) {}

  void fill(void *out, size_t numElements) const override;

private:
  float min;
  float max;
};

template <typename T>
class UniformIntegerInputRandomDistribution final
    : public InputRandomDistribution {
public:
  explicit UniformIntegerInputRandomDistribution(DType dtype, T min, T max)
      : InputRandomDistribution(dtype), min(min), max(max) {}

  void fill(void *out, size_t numElements) const override;

private:
  T min;
  T max;
};

} // namespace M::BenchmarkCLI::Detail

namespace BCD = BenchmarkCLI::Detail;

void BCD::UniformFloatInputRandomDistribution::fill(void *out,
                                                    size_t numElements) const {
  assert(getDType() == DType::f32);
  float *outF32 = static_cast<float *>(out);
  fillWithRandomFloats({outF32, numElements}, min, max);
}

template <typename T>
void BCD::UniformIntegerInputRandomDistribution<T>::fill(
    void *out, size_t numElements) const {
  assert(getDType() == DTypeForCXXType<T>::kind);
  T *outT = static_cast<T *>(out);
  fillWithRandomInts({outT, numElements}, min, max);
}

static std::unique_ptr<InputRandomDistribution>
createUniformFloatInputRandomDistribution(
    DType dtype, const InputGenerationRandomUniform &spec,
    InputSchemaErrorReporter &errorReporter) {
  float min, max;
  bool minOK = withinScope(errorReporter, "min", [&]() {
    return parseDefaultedElementValue(spec.min, min, -1.0f, errorReporter);
  });
  bool maxOK = withinScope(errorReporter, "max", [&]() {
    return parseDefaultedElementValue(spec.max, max, 1.0f, errorReporter);
  });
  if (!minOK || !maxOK)
    return nullptr;
  if (min > max) {
    errorReporter.reportError("'min' is greater than 'max'");
    return nullptr;
  }
  return std::make_unique<BCD::UniformFloatInputRandomDistribution>(dtype, min,
                                                                    max);
}

template <typename T>
static std::unique_ptr<InputRandomDistribution>
createUniformIntegerInputRandomDistribution(
    DType dtype, const InputGenerationRandomUniform &spec,
    InputSchemaErrorReporter &errorReporter) {
  T min, max;
  bool minOK = withinScope(errorReporter, "min", [&]() {
    return parseDefaultedElementValue(
        spec.min, min, std::numeric_limits<T>::lowest(), errorReporter);
  });
  bool maxOK = withinScope(errorReporter, "max", [&]() {
    return parseDefaultedElementValue(
        spec.max, max, std::numeric_limits<T>::max(), errorReporter);
  });
  if (!minOK || !maxOK)
    return nullptr;
  if (min > max) {
    errorReporter.reportError("'min' is greater than 'max'");
    return nullptr;
  }
  return std::make_unique<BCD::UniformIntegerInputRandomDistribution<T>>(
      dtype, min, max);
}

std::unique_ptr<InputRandomDistribution>
M::BenchmarkCLI::createUniformInputRandomDistribution(
    DType dtype, const InputGenerationRandomUniform &spec,
    InputSchemaErrorReporter &errorReporter) {
  switch (dtype.getValue()) {
  case DType::f32:
    return createUniformFloatInputRandomDistribution(dtype, spec,
                                                     errorReporter);
  case DType::si8:
    return createUniformIntegerInputRandomDistribution<int8_t>(dtype, spec,
                                                               errorReporter);
  case DType::ui8:
    return createUniformIntegerInputRandomDistribution<uint8_t>(dtype, spec,
                                                                errorReporter);
  case DType::si16:
    return createUniformIntegerInputRandomDistribution<int16_t>(dtype, spec,
                                                                errorReporter);
  case DType::ui16:
    return createUniformIntegerInputRandomDistribution<uint16_t>(dtype, spec,
                                                                 errorReporter);
  case DType::si32:
    return createUniformIntegerInputRandomDistribution<int32_t>(dtype, spec,
                                                                errorReporter);
  case DType::ui32:
    return createUniformIntegerInputRandomDistribution<uint32_t>(dtype, spec,
                                                                 errorReporter);
  case DType::si64:
    return createUniformIntegerInputRandomDistribution<int64_t>(dtype, spec,
                                                                errorReporter);
  case DType::ui64:
    return createUniformIntegerInputRandomDistribution<uint64_t>(dtype, spec,
                                                                 errorReporter);
  default:
    errorReporter.reportError(llvm::Twine("Unsupported dtype ") +
                              dtype.getAsString());
    return nullptr;
  }
}

std::unique_ptr<InputRandomDistribution>
M::BenchmarkCLI::createNormalInputRandomDistribution(
    DType dtype, const InputGenerationRandomNormal &spec,
    InputSchemaErrorReporter &errorReporter) {
  errorReporter.reportError("Normal distribution not yet implemented");
  return nullptr;
}

std::unique_ptr<InputRandomDistribution>
M::BenchmarkCLI::createInputRandomDistribution(
    DType dtype, const InputGenerationRandom &spec,
    InputSchemaErrorReporter &errorReporter) {
  int numSpecified = 0;
  std::unique_ptr<InputRandomDistribution> dist;
  if (spec.uniform) {
    ++numSpecified;
    errorReporter.enterScope("uniform");
    dist = createUniformInputRandomDistribution(dtype, *spec.uniform,
                                                errorReporter);
    errorReporter.exitScope("uniform");
  }
  if (spec.normal) {
    ++numSpecified;
    errorReporter.enterScope("normal");
    dist =
        createNormalInputRandomDistribution(dtype, *spec.normal, errorReporter);
    errorReporter.exitScope("normal");
  }
  if (numSpecified == 0) {
    errorReporter.reportError("No random distribution specified");
    return nullptr;
  }
  if (numSpecified > 1) {
    errorReporter.reportError("Only one random distribution may be specified");
    return nullptr;
  }
  return dist;
}

namespace M::BenchmarkCLI::Detail {
class RandomInputDataGenerator final : public InputDataGenerator {
public:
  explicit RandomInputDataGenerator(
      std::unique_ptr<InputRandomDistribution> dist)
      : dist(std::move(dist)) {}

  ErrorOr<GML::Tensor> generate(LLCL::CompactRuntimePtr runtime,
                                const TensorShape &shape) const override;

private:
  std::unique_ptr<InputRandomDistribution> dist;
};
} // namespace M::BenchmarkCLI::Detail

ErrorOr<GML::Tensor>
BCD::RandomInputDataGenerator::generate(LLCL::CompactRuntimePtr runtime,
                                        const TensorShape &shape) const {
  if (!shape.isStatic())
    return Error("Random input requires fully defined input shape");
  GML::Tensor tensor = GML::Tensor::createOwned({shape, dist->getDType()},
                                                runtime, std::nullopt);
  dist->fill(tensor.getMutableBuffer(), shape.getNumElements());
  return std::move(tensor);
}

std::unique_ptr<InputDataGenerator>
M::BenchmarkCLI::createRandomInputDataGenerator(
    std::unique_ptr<InputRandomDistribution> distribution) {
  return std::make_unique<BCD::RandomInputDataGenerator>(
      std::move(distribution));
}

std::unique_ptr<InputDataGenerator>
M::BenchmarkCLI::createRandomInputDataGenerator(
    DType dtype, const InputGenerationRandom &spec,
    InputSchemaErrorReporter &errorReporter) {
  auto dist = createInputRandomDistribution(dtype, spec, errorReporter);
  if (!dist)
    return nullptr;
  return createRandomInputDataGenerator(std::move(dist));
}

std::unique_ptr<InputDataGenerator>
M::BenchmarkCLI::createConstantInputDataGenerator(
    DType dtype, StringRef constant, InputSchemaErrorReporter &errorReporter) {
  errorReporter.reportError("Constant input generation not yet implemented");
  return nullptr;
}

std::unique_ptr<InputDataGenerator> M::BenchmarkCLI::createInputDataGenerator(
    std::optional<DType> dtype, const InputGenerationData &spec,
    InputSchemaErrorReporter &errorReporter) {
  int numSpecified = 0;
  std::unique_ptr<InputDataGenerator> gen;
  if (spec.random) {
    ++numSpecified;
    errorReporter.enterScope("random");
    if (dtype)
      gen = createRandomInputDataGenerator(*dtype, *spec.random, errorReporter);
    else
      errorReporter.reportError(
          "Random input data source requires dtype to be known");
    errorReporter.exitScope("random");
  }
  if (spec.constant) {
    ++numSpecified;
    errorReporter.enterScope("constant");
    if (dtype)
      gen = createConstantInputDataGenerator(*dtype, *spec.constant,
                                             errorReporter);
    else
      errorReporter.reportError(
          "Constant input data source requires dtype to be known");
    errorReporter.exitScope("constant");
  }
  if (numSpecified == 0) {
    errorReporter.reportError("No data generation source specified");
    return nullptr;
  }
  if (numSpecified > 1) {
    errorReporter.reportError(
        "Only one data generation source may be specified");
    return nullptr;
  }
  return gen;
}

static M_Dtype dtypeToCAPI(DType dty) {
  switch (dty.getValue()) {
  case DType::si8:
    return M_INT8;
  case DType::si16:
    return M_INT16;
  case DType::si32:
    return M_INT32;
  case DType::si64:
    return M_INT64;
  case DType::ui8:
    return M_UINT8;
  case DType::ui16:
    return M_UINT16;
  case DType::ui32:
    return M_UINT32;
  case DType::ui64:
    return M_UINT64;
  case DType::f16:
    return M_FLOAT16;
  case DType::f32:
    return M_FLOAT32;
  case DType::f64:
    return M_FLOAT64;
  case DType::kBool:
    return M_BOOL;
  default:
    return M_UNKNOWN;
  }
}

static TensorSpecPtr createTensorSpec(const char *name,
                                      const TensorSpec &spec) {
  // Copy is required because TensorShape stores ssize_t, but API takes
  // int64_t, and these are not necessarily the same.
  SmallVector<int64_t, 8> shapeCopy;
  std::copy(spec.begin(), spec.end(), std::back_inserter(shapeCopy));
  return TensorSpecPtr(M_newTensorSpec(shapeCopy.data(), spec.getRank(),
                                       dtypeToCAPI(spec.getEltType()), name),
                       M_freeTensorSpec);
}

namespace M::BenchmarkCLI::Detail {
class GeneratedInputImpl final : public GeneratedInput {
public:
  static ErrorOr<GeneratedInputImpl> create(std::string name,
                                            GML::Tensor tensor);

  void borrowInto(M_AsyncTensorMap *atm, M_Status *status) const override;

private:
  GeneratedInputImpl(std::string name, TensorSpecPtr spec, GML::Tensor tensor);

  std::string name;
  TensorSpecPtr spec;
  GML::Tensor tensor;
};
} // namespace M::BenchmarkCLI::Detail

BCD::GeneratedInputImpl::GeneratedInputImpl(std::string name,
                                            TensorSpecPtr spec,
                                            GML::Tensor tensor)
    : name(std::move(name)), spec(std::move(spec)), tensor(std::move(tensor)) {}

ErrorOr<BCD::GeneratedInputImpl>
BCD::GeneratedInputImpl::create(std::string name, GML::Tensor tensor) {
  TensorSpecPtr spec = createTensorSpec(name.c_str(), tensor.getSpec());
  if (!spec)
    return Error(Twine("Unable to create M_TensorSpec for ") + name);
  return GeneratedInputImpl(name, std::move(spec), std::move(tensor));
}

void BCD::GeneratedInputImpl::borrowInto(M_AsyncTensorMap *atm,
                                         M_Status *status) const {
  M_borrowTensorInto(atm, tensor.getBuffer(), spec.get(), status);
}

namespace M::BenchmarkCLI::Detail {
class InputGeneratorImpl final : public InputGenerator {
public:
  InputGeneratorImpl(std::string name, std::optional<TensorSpec> shape,
                     std::unique_ptr<InputDataGenerator> dataGenerator)
      : name(std::move(name)), shape(std::move(shape)),
        dataGenerator(std::move(dataGenerator)) {}

  ErrorOr<std::unique_ptr<GeneratedInput>>
  generateInput(LLCL::CompactRuntimePtr runtime, size_t queryIndex) override;

private:
  std::string name;
  std::optional<TensorSpec> shape;
  std::unique_ptr<InputDataGenerator> dataGenerator;
};
} // namespace M::BenchmarkCLI::Detail

ErrorOr<std::unique_ptr<GeneratedInput>>
BCD::InputGeneratorImpl::generateInput(LLCL::CompactRuntimePtr runtime,
                                       size_t queryIndex) {
  // TODO: Use query index.
  auto tensorOr =
      dataGenerator->generate(runtime, shape.value_or(TensorSpec()));
  if (tensorOr.isError())
    return tensorOr.takeError();
  auto inputOr = GeneratedInputImpl::create(name, std::move(*tensorOr));
  if (inputOr.isError())
    return inputOr.takeError();
  return std::make_unique<GeneratedInputImpl>(std::move(*inputOr));
}

std::unique_ptr<InputGenerator>
M::BenchmarkCLI::createInputGenerator(const InputGenerationInput &spec,
                                      InputSchemaErrorReporter &errorReporter) {
  bool ok = true;
  std::optional<std::variant<std::string, int>> nameOrIndex;
  if (spec.inputName.has_value())
    nameOrIndex = *spec.inputName;
  if (spec.inputIndex.has_value()) {
    if (nameOrIndex.has_value()) {
      errorReporter.reportError(
          "Cannot specify both input name and input index");
      ok = false;
    } else
      nameOrIndex = *spec.inputIndex;
  }
  if (!nameOrIndex.has_value()) {
    errorReporter.enterScope("input_name");
    errorReporter.reportError("Input name must be specified");
    errorReporter.exitScope("input_name");
    ok = false;
  }
  if (nameOrIndex.has_value() && std::holds_alternative<int>(*nameOrIndex)) {
    errorReporter.enterScope("input_index");
    errorReporter.reportError(
        "Inputs specified by index are not yet implemented");
    errorReporter.exitScope("input_index");
    ok = false;
  }
  std::optional<TensorSpec> shape = spec.shape;
  std::unique_ptr<InputDataGenerator> dataGenerator;
  errorReporter.enterScope("data");
  dataGenerator = createInputDataGenerator(
      shape.has_value() ? std::optional(shape->getEltType()) : std::nullopt,
      spec.data.value_or(InputGenerationData()), errorReporter);
  errorReporter.exitScope("data");
  if (!dataGenerator)
    ok = false;
  if (!ok)
    return nullptr;
  assert(nameOrIndex.has_value() &&
         std::holds_alternative<std::string>(*nameOrIndex));
  return std::make_unique<BCD::InputGeneratorImpl>(
      std::get<0>(std::move(*nameOrIndex)), std::move(shape),
      std::move(dataGenerator));
}

namespace M::BenchmarkCLI::Detail {
class GeneratedQueryImpl final : public GeneratedQuery {
public:
  GeneratedQueryImpl(TensorMapPtr atm,
                     std::vector<std::unique_ptr<GeneratedInput>> tensors)
      : atm(std::move(atm)), tensors(std::move(tensors)) {}

  M_AsyncTensorMap *getAsyncTensorMap() const override;

private:
  TensorMapPtr atm;
  std::vector<std::unique_ptr<GeneratedInput>> tensors;
};
} // namespace M::BenchmarkCLI::Detail

M_AsyncTensorMap *BCD::GeneratedQueryImpl::getAsyncTensorMap() const {
  return atm.get();
}

namespace M::BenchmarkCLI::Detail {
class QueryGeneratorImpl final : public QueryGenerator {
public:
  explicit QueryGeneratorImpl(
      std::vector<std::unique_ptr<InputGenerator>> inputGenerators)
      : inputGenerators(std::move(inputGenerators)) {}

  ssize_t getNumQueries() const override;
  ErrorOr<std::unique_ptr<GeneratedQuery>>
  generateQuery(LLCL::CompactRuntimePtr allocRuntime,
                M_RuntimeContext *runtimeContext,
                size_t queryIndex) const override;

private:
  std::vector<std::unique_ptr<InputGenerator>> inputGenerators;
};
} // namespace M::BenchmarkCLI::Detail

ssize_t BCD::QueryGeneratorImpl::getNumQueries() const { return -1; }

ErrorOr<std::unique_ptr<GeneratedQuery>>
BCD::QueryGeneratorImpl::generateQuery(LLCL::CompactRuntimePtr allocRuntime,
                                       M_RuntimeContext *runtimeContext,
                                       size_t queryIndex) const {
  std::vector<std::unique_ptr<GeneratedInput>> tensors;
  for (auto &inputGenerator : inputGenerators) {
    auto generatedInputOr =
        inputGenerator->generateInput(allocRuntime, queryIndex);
    // N.B.: We do not wrap the error with context about which input it is,
    // since the input generator itself is best positioned to provide this
    // information (it knows its name, for example, while we do not).
    if (generatedInputOr.isError())
      return generatedInputOr.takeError();
    tensors.push_back(std::move(*generatedInputOr));
  }
  TensorMapPtr atm(M_newAsyncTensorMap(runtimeContext), M_freeAsyncTensorMap);
  StatusPtr status(M_newStatus(), M_freeStatus);
  for (auto &generatedInput : tensors) {
    generatedInput->borrowInto(atm.get(), status.get());
    if (M_isError(status.get()))
      return Error(M_getError(status.get()));
  }
  return std::make_unique<BCD::GeneratedQueryImpl>(std::move(atm),
                                                   std::move(tensors));
}

std::unique_ptr<QueryGenerator> M::BenchmarkCLI::createQueryGenerator(
    std::vector<std::unique_ptr<InputGenerator>> inputGenerators) {
  return std::make_unique<BCD::QueryGeneratorImpl>(std::move(inputGenerators));
}

std::unique_ptr<QueryGenerator>
M::BenchmarkCLI::createQueryGenerator(const InputGenerationSchema &schema,
                                      InputSchemaErrorReporter &errorReporter) {
  bool ok = true;
  std::vector<std::unique_ptr<InputGenerator>> inputGenerators;
  errorReporter.enterScope("inputs");
  if (schema.inputs.has_value()) {
    int i = 0;
    for (auto &inputSpec : *schema.inputs) {
      std::string iStr = Twine(i).str();
      errorReporter.enterScope(iStr);
      auto generator = createInputGenerator(inputSpec, errorReporter);
      if (generator)
        inputGenerators.push_back(std::move(generator));
      else
        ok = false;
      errorReporter.exitScope(iStr);
    }
  } else {
    errorReporter.reportError("Inputs must be specified");
    ok = false;
  }
  errorReporter.exitScope("inputs");
  if (!ok)
    return nullptr;
  return createQueryGenerator(std::move(inputGenerators));
}
