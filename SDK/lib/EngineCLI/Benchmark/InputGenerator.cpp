//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SDK/EngineCLI/Benchmark/InputGenerator.h"
#include "SDK/EngineCLI/Benchmark/NPYReader.h"
#include "SDK/EngineCLI/Benchmark/Tensor.h"
#include "SDK/EngineCLI/Common/APIBridge.h"
#include "SDK/EngineCLI/Common/InputGenerationSchema.h"
#include "Support/AssertStream.h"
#include "Support/MathExtras.h"
#include "max/cpp/aliases.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/MemoryBuffer.h"
#include <type_traits>
#include <variant>

using namespace M;
using namespace M::EngineCLI;
using namespace M::BenchmarkCLI;

template <typename Fn>
static auto withinScope(InputSchemaErrorReporter &errorReporter,
                        StringRef scopeName, const Fn &fn) {
  errorReporter.enterScope(scopeName);
  auto result = fn();
  errorReporter.exitScope(scopeName);
  return std::move(result);
}

static bool parseElementValue(StringRef value, bool &out,
                              InputSchemaErrorReporter &errorReporter) {
  if (value == "false") {
    out = false;
    return true;
  }
  if (value == "true") {
    out = true;
    return true;
  }
  errorReporter.reportError("Unable to parse '" + llvm::Twine(value) +
                            "' as bool value");
  return false;
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

static bool parseElementValue(StringRef value, double &out,
                              InputSchemaErrorReporter &errorReporter) {
  bool ok = !value.getAsDouble(out);
  if (!ok)
    errorReporter.reportError("Unable to parse '" + llvm::Twine(value) +
                              "' as f64 value");
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

template <DType::Cases kDType>
struct DTypeGenerationTraits {};

#define NATIVELY_REPRESENTED(dt)                                               \
  template <>                                                                  \
  struct DTypeGenerationTraits<dt> {                                           \
    using StorageType = CXXTypeForDType<dt>::CXXType;                          \
    using GenerationType = StorageType;                                        \
  }
NATIVELY_REPRESENTED(DType::si8);
NATIVELY_REPRESENTED(DType::ui8);
NATIVELY_REPRESENTED(DType::si16);
NATIVELY_REPRESENTED(DType::ui16);
NATIVELY_REPRESENTED(DType::si32);
NATIVELY_REPRESENTED(DType::ui32);
NATIVELY_REPRESENTED(DType::si64);
NATIVELY_REPRESENTED(DType::ui64);
NATIVELY_REPRESENTED(DType::f32);
NATIVELY_REPRESENTED(DType::f64);
NATIVELY_REPRESENTED(DType::kBool);
#undef NATIVELY_REPRESENTED

template <>
struct DTypeGenerationTraits<DType::f16> {
  using StorageType = uint16_t;
  using GenerationType = float;

  static uint16_t generationToStorage(float v) {
    APFloat f(v);
    // Ignore floating point exceptions during conversion.
    bool losesInfo;
    static_cast<void>(f.convert(APFloat::IEEEhalf(),
                                APFloat::rmNearestTiesToEven, &losesInfo));
    // Also ignore info loss.
    return static_cast<uint16_t>(f.bitcastToAPInt().getZExtValue());
  }
};

template <DType::Cases kDType>
class UniformFloatInputRandomDistribution final
    : public InputRandomDistribution {
  using StorageType = typename DTypeGenerationTraits<kDType>::StorageType;
  using GenerationType = typename DTypeGenerationTraits<kDType>::GenerationType;

public:
  explicit UniformFloatInputRandomDistribution(GenerationType min,
                                               GenerationType max)
      : InputRandomDistribution(kDType), min(min), max(max) {}

  void fill(void *out, size_t numElements) const override;

private:
  GenerationType min;
  GenerationType max;
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

template <DType::Cases kDType>
void BCD::UniformFloatInputRandomDistribution<kDType>::fill(
    void *out, size_t numElements) const {
  using StorageType = typename BCD::DTypeGenerationTraits<kDType>::StorageType;
  using GenerationType =
      typename BCD::DTypeGenerationTraits<kDType>::GenerationType;
  assert(getDType() == kDType);
  MutableArrayRef<StorageType> outStorage{static_cast<StorageType *>(out),
                                          numElements};
  if constexpr (std::is_same_v<StorageType, GenerationType>) {
    fillWithRandomFloats(outStorage, min, max);
  } else {
    std::uniform_real_distribution<GenerationType> distribution(min, max);
    std::default_random_engine randEngine(/*seed=*/0);
    std::generate(outStorage.begin(), outStorage.end(), [&]() {
      return DTypeGenerationTraits<kDType>::generationToStorage(
          distribution(randEngine));
    });
  }
}

template <typename T>
void BCD::UniformIntegerInputRandomDistribution<T>::fill(
    void *out, size_t numElements) const {
  assert(getDType() == DTypeForCXXType<T>::kind);
  T *outT = static_cast<T *>(out);
  fillWithRandomInts({outT, numElements}, min, max);
}

template <DType::Cases kDType>
static std::unique_ptr<InputRandomDistribution>
createUniformFloatInputRandomDistribution(
    const InputGenerationRandomUniform &spec,
    InputSchemaErrorReporter &errorReporter) {
  using GenerationType =
      typename BCD::DTypeGenerationTraits<kDType>::GenerationType;
  GenerationType min, max;
  bool minOK = withinScope(errorReporter, "min", [&]() {
    return parseDefaultedElementValue(spec.min, min, GenerationType(-1),
                                      errorReporter);
  });
  bool maxOK = withinScope(errorReporter, "max", [&]() {
    return parseDefaultedElementValue(spec.max, max, GenerationType(1),
                                      errorReporter);
  });
  if (!minOK || !maxOK)
    return nullptr;
  if (min > max) {
    errorReporter.reportError("'min' is greater than 'max'");
    return nullptr;
  }
  return std::make_unique<BCD::UniformFloatInputRandomDistribution<kDType>>(
      min, max);
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
  case DType::f16:
    return createUniformFloatInputRandomDistribution<DType::f16>(spec,
                                                                 errorReporter);
  case DType::f32:
    return createUniformFloatInputRandomDistribution<DType::f32>(spec,
                                                                 errorReporter);
  case DType::f64:
    return createUniformFloatInputRandomDistribution<DType::f64>(spec,
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

  ssize_t getNumQueries() const override;
  ErrorOr<Tensor> generate(size_t queryIndex,
                           const TensorShape &shape) const override;

private:
  std::unique_ptr<InputRandomDistribution> dist;
};
} // namespace M::BenchmarkCLI::Detail

ssize_t BCD::RandomInputDataGenerator::getNumQueries() const { return -1; }

ErrorOr<BenchmarkCLI::Tensor>
BCD::RandomInputDataGenerator::generate(size_t queryIndex,
                                        const TensorShape &shape) const {
  // TODO: Use query index.
  if (!shape.isStatic())
    return Error("Random input requires fully defined input shape");
  TensorSpec spec(shape, dist->getDType());
  std::unique_ptr<uint8_t[]> data =
      std::make_unique<uint8_t[]>(spec.getSizeInBytes());
  dist->fill(data.get(), spec.getNumElements());
  const void *dataPtr = data.get();
  return Tensor(spec, dataPtr, Tensor::own(std::move(data)));
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

namespace M::BenchmarkCLI::Detail {
template <typename T>
class ConstantInputDataGenerator final : public InputDataGenerator {
public:
  explicit ConstantInputDataGenerator(DType dtype, T value)
      : dtype(dtype), value(value) {}

  ssize_t getNumQueries() const override { return -1; }
  ErrorOr<Tensor> generate(size_t queryIndex,
                           const TensorShape &shape) const override;

private:
  DType dtype;
  T value;
};
} // namespace M::BenchmarkCLI::Detail

template <typename T>
ErrorOr<BenchmarkCLI::Tensor>
BCD::ConstantInputDataGenerator<T>::generate(size_t queryIndex,
                                             const TensorShape &shape) const {
  (void)queryIndex; // Constants do not change based on query index.
  if (!shape.isStatic())
    return Error("Constant input requires fully defined input shape");
  TensorSpec spec(shape, dtype);
  std::unique_ptr<T[]> data = std::make_unique<T[]>(spec.getNumElements());
  std::fill(data.get(), data.get() + spec.getNumElements(), value);
  const void *dataPtr = data.get();
  return Tensor(spec, dataPtr, Tensor::own(std::move(data)));
}

template <DType::Cases kDType>
static std::unique_ptr<InputDataGenerator>
createConstantInputDataGeneratorImpl(StringRef constant,
                                     InputSchemaErrorReporter &errorReporter) {
  using StorageType = typename BCD::DTypeGenerationTraits<kDType>::StorageType;
  using GenerationType =
      typename BCD::DTypeGenerationTraits<kDType>::GenerationType;
  GenerationType generationValue;
  if (!parseElementValue(constant, generationValue, errorReporter))
    return nullptr;
  StorageType storageValue;
  if constexpr (std::is_same_v<StorageType, GenerationType>)
    storageValue = generationValue;
  else
    storageValue = BCD::DTypeGenerationTraits<kDType>::generationToStorage(
        generationValue);
  return std::make_unique<BCD::ConstantInputDataGenerator<StorageType>>(
      kDType, storageValue);
}

std::unique_ptr<InputDataGenerator>
M::BenchmarkCLI::createConstantInputDataGenerator(
    DType dtype, StringRef constant, InputSchemaErrorReporter &errorReporter) {
  switch (dtype.getValue()) {
#define DTYPE_CASE(dt)                                                         \
  case dt:                                                                     \
    return createConstantInputDataGeneratorImpl<dt>(constant, errorReporter)
    DTYPE_CASE(DType::f16);
    DTYPE_CASE(DType::f32);
    DTYPE_CASE(DType::f64);
    DTYPE_CASE(DType::si8);
    DTYPE_CASE(DType::ui8);
    DTYPE_CASE(DType::si16);
    DTYPE_CASE(DType::ui16);
    DTYPE_CASE(DType::si32);
    DTYPE_CASE(DType::ui32);
    DTYPE_CASE(DType::si64);
    DTYPE_CASE(DType::ui64);
    DTYPE_CASE(DType::kBool);
#undef DTYPE_CASE
  default:
    errorReporter.reportError(Twine("Constant input generation for dtype '") +
                              dtype.getAsString() + "' not supported");
    return nullptr;
  }
}

namespace M::BenchmarkCLI::Detail {
class FileInputDataGenerator final : public InputDataGenerator {
public:
  explicit FileInputDataGenerator(Tensor wholeTensor);

  ssize_t getNumQueries() const override;
  ErrorOr<Tensor> generate(size_t queryIndex,
                           const TensorShape &shape) const override;

private:
  std::shared_ptr<Tensor> wholeTensor;
};
} // namespace M::BenchmarkCLI::Detail

BCD::FileInputDataGenerator::FileInputDataGenerator(Tensor wholeTensor)
    : wholeTensor(std::make_shared<Tensor>(std::move(wholeTensor))) {}

ssize_t BCD::FileInputDataGenerator::getNumQueries() const {
  return wholeTensor->getSpec()[0];
}

ErrorOr<BenchmarkCLI::Tensor>
BCD::FileInputDataGenerator::generate(size_t queryIndex,
                                      const TensorShape &shape) const {
  TensorShape actualShape(wholeTensor->getSpec().begin() + 1,
                          wholeTensor->getSpec().end());
  if (ErrorOrSuccess error = shape.isRefinedBy(actualShape); error.isError())
    return Error(Twine("Declared shape ") + shape.getAsString() +
                 " incompatible with file intrinsic shape " +
                 actualShape.getAsString() + ": " + error.getError());
  TensorSpec sliceSpec(actualShape, wholeTensor->getEltType());
  const void *sliceData =
      reinterpret_cast<const char *>(wholeTensor->getBuffer()) +
      queryIndex * sliceSpec.getSizeInBytes();
  // wholeTensor is a shared_ptr, so this Tensor::own merely owns one shared
  // reference to the wholeTensor, not transferring exclusive ownership.
  return Tensor(std::move(sliceSpec), sliceData, Tensor::own(wholeTensor));
}

std::unique_ptr<InputDataGenerator>
M::BenchmarkCLI::createFileInputDataGenerator(
    std::optional<DType> dtype, const InputGenerationFile &spec,
    InputSchemaErrorReporter &errorReporter) {
  auto bufferOr = llvm::MemoryBuffer::getFile(spec.path,
                                              /*IsText=*/false,
                                              /*RequiresNullTerminator=*/false);
  if (!bufferOr) {
    errorReporter.enterScope("path");
    errorReporter.reportError(Twine("Unable to open input file: ") +
                              bufferOr.getError().message());
    errorReporter.exitScope("path");
    return nullptr;
  }
  StringRef format;
  if (spec.format.has_value()) {
    format = *spec.format;
  } else if (hasNPYFileMagic(**bufferOr)) {
    format = "npy";
  } else {
    errorReporter.enterScope("format");
    errorReporter.reportError(
        "Format not specified and unable to automatically detect");
    errorReporter.exitScope("format");
    return nullptr;
  }
  Tensor tensor;
  if (format == "npy") {
    ErrorOr<Tensor> tensorOr = readNPYFile(std::move(*bufferOr));
    if (tensorOr.isError()) {
      errorReporter.reportError(Twine("Unable to read NPY file: ") +
                                tensorOr.getError());
      return nullptr;
    }
    tensor = tensorOr.takeValue();
  } else {
    errorReporter.enterScope("format");
    errorReporter.reportError("Unknown file format");
    errorReporter.exitScope("format");
    return nullptr;
  }
  if (dtype.has_value() && tensor.getSpec().getEltType() != dtype) {
    errorReporter.reportError(
        Twine("Data type of file input '") +
        tensor.getSpec().getEltType().getAsString() +
        "' inconsistent with previously declared dtype '" +
        dtype->getAsString() + "'");
    return nullptr;
  }
  if (tensor.getSpec().getRank() == 0) {
    errorReporter.reportError(
        "Tensors sourced from input file must have rank > 0");
    return nullptr;
  }
  return std::make_unique<BCD::FileInputDataGenerator>(std::move(tensor));
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
  if (spec.file) {
    ++numSpecified;
    errorReporter.enterScope("file");
    gen = createFileInputDataGenerator(dtype, *spec.file, errorReporter);
    errorReporter.exitScope("file");
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

namespace M::BenchmarkCLI::Detail {
class GeneratedInputImpl final : public GeneratedInput {
public:
  static ErrorOr<GeneratedInputImpl> create(std::string name, Tensor tensor);

  void borrowInto(M_AsyncTensorMap *atm, M_Status *status) const override;

private:
  GeneratedInputImpl(std::string name, TensorSpecPtr spec, Tensor tensor);

  std::string name;
  TensorSpecPtr spec;
  Tensor tensor;
};
} // namespace M::BenchmarkCLI::Detail

BCD::GeneratedInputImpl::GeneratedInputImpl(std::string name,
                                            TensorSpecPtr spec, Tensor tensor)
    : name(std::move(name)), spec(std::move(spec)), tensor(std::move(tensor)) {}

ErrorOr<BCD::GeneratedInputImpl>
BCD::GeneratedInputImpl::create(std::string name, Tensor tensor) {
  ErrorOr<TensorSpecPtr> specOr =
      tensorSpecToCAPI(name.c_str(), tensor.getSpec());
  if (specOr.isError())
    return Error(Twine("Unable to create M_TensorSpec for ") + name + ": " +
                 specOr.getError());
  return GeneratedInputImpl(name, std::move(*specOr), std::move(tensor));
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

  ssize_t getNumQueries() const override;
  ErrorOr<std::unique_ptr<GeneratedInput>>
  generateInput(size_t queryIndex) override;

private:
  std::string name;
  std::optional<TensorSpec> shape;
  std::unique_ptr<InputDataGenerator> dataGenerator;
};
} // namespace M::BenchmarkCLI::Detail

ssize_t BCD::InputGeneratorImpl::getNumQueries() const {
  return dataGenerator->getNumQueries();
}

ErrorOr<std::unique_ptr<GeneratedInput>>
BCD::InputGeneratorImpl::generateInput(size_t queryIndex) {
  auto tensorOr =
      dataGenerator->generate(queryIndex, shape.value_or(TensorSpec()));
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
  QueryGeneratorImpl(
      ssize_t numQueries,
      std::vector<std::unique_ptr<InputGenerator>> inputGenerators)
      : numQueries(numQueries), inputGenerators(std::move(inputGenerators)) {}

  ssize_t getNumQueries() const override;
  ErrorOr<std::unique_ptr<GeneratedQuery>>
  generateQuery(M_RuntimeContext *runtimeContext,
                size_t queryIndex) const override;

private:
  ssize_t numQueries;
  std::vector<std::unique_ptr<InputGenerator>> inputGenerators;
};
} // namespace M::BenchmarkCLI::Detail

ssize_t BCD::QueryGeneratorImpl::getNumQueries() const { return numQueries; }

ErrorOr<std::unique_ptr<GeneratedQuery>>
BCD::QueryGeneratorImpl::generateQuery(M_RuntimeContext *runtimeContext,
                                       size_t queryIndex) const {
  std::vector<std::unique_ptr<GeneratedInput>> tensors;
  for (auto &inputGenerator : inputGenerators) {
    auto generatedInputOr = inputGenerator->generateInput(queryIndex);
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
    std::vector<std::unique_ptr<InputGenerator>> inputGenerators,
    InputSchemaErrorReporter &errorReporter) {
  ssize_t numQueries = -1;
  size_t numQueriesSource;
  bool numQueriesConflict = false;
  for (auto [genIndex, inputGen] : llvm::enumerate(inputGenerators)) {
    ssize_t inputNumQueries = inputGen->getNumQueries();
    if (inputNumQueries == -1)
      continue;
    if (numQueries == -1) {
      numQueries = inputNumQueries;
      numQueriesSource = genIndex;
    } else if (numQueries != inputNumQueries) {
      errorReporter.enterScope("inputs");
      errorReporter.enterScope(Twine(genIndex).str());
      errorReporter.reportError("Query count " + Twine(inputNumQueries) +
                                " inconsistent with other inputs");
      errorReporter.exitScope(Twine(genIndex).str());
      errorReporter.enterScope(Twine(numQueriesSource).str());
      errorReporter.reportError("Note: Conflicting query count (" +
                                Twine(numQueries) + ") came from this input");
      errorReporter.exitScope(Twine(numQueriesSource).str());
      errorReporter.exitScope("inputs");
      numQueriesConflict = true;
    }
  }
  if (numQueriesConflict)
    return nullptr;
  return std::make_unique<BCD::QueryGeneratorImpl>(numQueries,
                                                   std::move(inputGenerators));
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
  return createQueryGenerator(std::move(inputGenerators), errorReporter);
}
