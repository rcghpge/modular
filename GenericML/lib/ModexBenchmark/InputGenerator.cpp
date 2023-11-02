//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/ModexBenchmark/InputGenerator.h"
#include "GenericML/ModexBenchmark/InputGenerationSchema.h"
#include "Support/MathExtras.h"

using namespace M;
using namespace M::ModexBench;

template <typename Fn>
static auto withinScope(InputSchemaErrorReporter &errorReporter,
                        StringRef scopeName, const Fn &fn) {
  errorReporter.enterScope(scopeName);
  auto result = fn();
  errorReporter.exitScope(scopeName);
  return std::move(result);
}

static bool parseF32Value(StringRef value, float &out,
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

[[maybe_unused]] static bool
parseDefaultedF32Value(StringRef value, float &out, float defaultValue,
                       InputSchemaErrorReporter &errorReporter) {
  if (value.empty()) {
    out = defaultValue;
    return true;
  }
  return parseF32Value(value, out, errorReporter);
}

static bool parseDefaultedF32Value(const std::optional<std::string> &value,
                                   float &out, float defaultValue,
                                   InputSchemaErrorReporter &errorReporter) {
  if (!value.has_value() || value->empty()) {
    out = defaultValue;
    return true;
  }
  return parseF32Value(*value, out, errorReporter);
}

namespace M::ModexBench::Detail {
class UniformInputRandomDistribution final : public InputRandomDistribution {
public:
  explicit UniformInputRandomDistribution(DType dtype, float min, float max)
      : InputRandomDistribution(dtype), min(min), max(max) {}

  void fill(void *out, size_t numElements) const override;

private:
  float min;
  float max;
};
} // namespace M::ModexBench::Detail

namespace MBD = ModexBench::Detail;

void MBD::UniformInputRandomDistribution::fill(void *out,
                                               size_t numElements) const {
  assert(getDType() == DType::f32);
  float *outF32 = static_cast<float *>(out);
  fillWithRandomFloats({outF32, numElements}, min, max);
}

std::unique_ptr<InputRandomDistribution>
M::ModexBench::createUniformInputRandomDistribution(
    DType dtype, const InputGenerationRandomUniform &spec,
    InputSchemaErrorReporter &errorReporter) {
  if (dtype != DType::f32) {
    errorReporter.reportError("Only f32 supported at this time");
    return nullptr;
  }
  float min, max;
  bool minOK = withinScope(errorReporter, "min", [&]() {
    return parseDefaultedF32Value(spec.min, min, -1.0f, errorReporter);
  });
  bool maxOK = withinScope(errorReporter, "max", [&]() {
    return parseDefaultedF32Value(spec.max, max, 1.0f, errorReporter);
  });
  if (!minOK || !maxOK)
    return nullptr;
  if (min > max) {
    errorReporter.reportError("'min' is greater than 'max'");
    return nullptr;
  }
  return std::make_unique<Detail::UniformInputRandomDistribution>(dtype, min,
                                                                  max);
}

std::unique_ptr<InputRandomDistribution>
M::ModexBench::createNormalInputRandomDistribution(
    DType dtype, const InputGenerationRandomNormal &spec,
    InputSchemaErrorReporter &errorReporter) {
  errorReporter.reportError("Normal distribution not yet implemented");
  return nullptr;
}

std::unique_ptr<InputRandomDistribution>
M::ModexBench::createInputRandomDistribution(
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

namespace M::ModexBench::Detail {
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
} // namespace M::ModexBench::Detail

ErrorOr<GML::Tensor>
MBD::RandomInputDataGenerator::generate(LLCL::CompactRuntimePtr runtime,
                                        const TensorShape &shape) const {
  if (!shape.isStatic())
    return Error("Random input requires fully defined input shape");
  GML::Tensor tensor = GML::Tensor::createOwned({shape, dist->getDType()},
                                                runtime, std::nullopt);
  dist->fill(tensor.getMutableBuffer(), shape.getNumElements());
  return std::move(tensor);
}

std::unique_ptr<InputDataGenerator>
M::ModexBench::createRandomInputDataGenerator(
    std::unique_ptr<InputRandomDistribution> distribution) {
  return std::make_unique<MBD::RandomInputDataGenerator>(
      std::move(distribution));
}

std::unique_ptr<InputDataGenerator>
M::ModexBench::createRandomInputDataGenerator(
    DType dtype, const InputGenerationRandom &spec,
    InputSchemaErrorReporter &errorReporter) {
  auto dist = createInputRandomDistribution(dtype, spec, errorReporter);
  if (!dist)
    return nullptr;
  return createRandomInputDataGenerator(std::move(dist));
}

std::unique_ptr<InputDataGenerator>
M::ModexBench::createConstantInputDataGenerator(
    DType dtype, StringRef constant, InputSchemaErrorReporter &errorReporter) {
  errorReporter.reportError("Constant input generation not yet implemented");
  return nullptr;
}

std::unique_ptr<InputDataGenerator> M::ModexBench::createInputDataGenerator(
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
