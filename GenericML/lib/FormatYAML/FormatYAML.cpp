//===- GenericML/FormatYAML.cpp -------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is a file serializes and deserializes the types found in GenericML such
// as TensorSpec, QuantSpec, and ModelSpec.
//
//===----------------------------------------------------------------------===//

#include "GenericML/FormatYAML/FormatYAML.h"
#include "GenericML/Support/ModelSpec.h"
#include "GenericML/Support/QuantSpec.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace M;

/// Provide a way to check if a template is defined so we can not define this
/// thing on platforms where ScalarTraits<T> is defined.
template <class T, std::size_t = sizeof(T)>
std::true_type is_complete_impl(T *);
std::false_type is_complete_impl(...);
template <class T>
using is_complete = decltype(is_complete_impl(std::declval<T *>()));

//===----------------------------------------------------------------------===//
// QuantTensorSpec
//===----------------------------------------------------------------------===//

namespace llvm::yaml {
/// Provide a forwarding from ssize_t to int64_t for YAML ScalarTraits.
template <typename T>
struct ScalarTraits<T, std::enable_if_t<!is_complete<ScalarTraits<T>>::value &&
                                        std::is_same_v<T, ssize_t>>> {
  static void output(const ssize_t &val, void *p, llvm::raw_ostream &out) {
    ScalarTraits<int64_t>::output((int64_t)val, p, out);
  }
  static StringRef input(StringRef scalar, void *p, ssize_t &value) {
    int64_t v;
    auto out = ScalarTraits<int64_t>::input(scalar, p, v);
    value = v;
    return out;
  }
  static QuotingType mustQuote(StringRef s) {
    return ScalarTraits<int64_t>::mustQuote(s);
  }
};

/// Provide the QuantSpec::ScaleZeroPointPair mapping trait for YAML conversion.
template <>
struct MappingTraits<M::QuantSpec::ScaleZeroPointPair> {
  static void mapping(IO &io,
                      M::QuantSpec::ScaleZeroPointPair &scaleZeroPointPair) {
    io.mapRequired("scale", scaleZeroPointPair.scale);
    io.mapRequired("zero_point", scaleZeroPointPair.zeroPoint);
  }
};
} // namespace llvm::yaml

/// Use LLVM_YAML_IS_SEQUENCE_VECTOR to specify SequenceTraits<> on the
/// SmallVector<QuantSpec::ScaleZeroPointPair> type.
LLVM_YAML_IS_SEQUENCE_VECTOR(M::QuantSpec::ScaleZeroPointPair);

/// Parses a string of the form dim0xdim1x...xDType into a TensorSpec.
static ErrorOrSuccess parseTensorSpec(llvm::yaml::IO &io, StringRef str,
                                      TensorSpec &tensorSpec) {
  // Split the string by the "x" character.
  SmallVector<StringRef, 5> splitStr;
  str.split(splitStr, "x");

  // Allocate the shape based on how many dimensions we have in the split
  // string. We remove 1 element, since the DType is always present. We also
  // account for the case where there are no elements --- i.e. the tensor is a
  // scalar.
  SmallVector<int64_t, 5> shape;
  // If the string is empty, then there is nothing to be done.
  if (!splitStr.empty()) {
    shape.reserve(splitStr.size());
    // Go through the split values ignoring the last element.
    for (auto &it : llvm::drop_end(splitStr)) {
      // Try to parse the string into an integer.
      int64_t value;
      if (it.getAsInteger(10, value)) {
        return Error(Twine("could not parse dimension integer from string: ") +
                     str + " because " + it +
                     " cannot be parsed as an integer");
      }
      shape.emplace_back(value);
    }
  }

  // We have parsed the shape, now we parse the DType.
  auto dtype = DType::getFromString(splitStr.back());
  if (failed(dtype))
    return Error(Twine("could not parse dtype from string: ") + str +
                 " because " + splitStr.back() + " is not a valid DType");

  // Create the tensor spec from the shape and dtype information.
  tensorSpec = TensorSpec(shape, *dtype);
  return success();
}

/// fromYAML will deserialize a QuantTensorSpec from YAML.
static void fromYAML(llvm::yaml::IO &io, QuantTensorSpec &quantTensorSpec) {
  // Read the "shape" field and parse it into a TensorSpec.
  std::string shapeStr;
  io.mapRequired("shape", shapeStr);
  TensorSpec tensorSpec;
  if (auto err = parseTensorSpec(io, shapeStr, tensorSpec)) {
    io.setError(err.getError());
    return;
  }

  // Read the "quantized" field. The "quantized" field is optional and so is not
  // required to exist.
  bool isQuantized = false;
  io.mapOptional("quantized", isQuantized, false);

  // If the quantization field is not set, then we are done.
  if (!isQuantized) {
    quantTensorSpec = QuantTensorSpec(std::move(tensorSpec));
    return;
  }

  // Otherwise, the quantized field is set, then we read the rest of the
  // QuantSpec information. We first parse the "kind" field.
  StringRef kindStr;
  io.mapRequired("kind", kindStr);
  auto kind = QuantSpec::kindFromString(kindStr);

  // The "quant_dim" field is optionally available when the quant kind is
  // M::QuantSpec::Kind::uniformPerAxis
  uint8_t quantDim;
  io.mapOptional("quant_dim", quantDim, 0);

  // The "params" field constains the optionally available vector of
  // ScaleZeroPointPair.
  SmallVector<M::QuantSpec::ScaleZeroPointPair> scaleZeroPointPairs;
  io.mapOptional("params", scaleZeroPointPairs);

  // Create the quant tensor spec from the tensor spec and the quantspec parsed
  // information.
  quantTensorSpec = QuantTensorSpec(
      std::move(tensorSpec), QuantSpec(kind, quantDim, scaleZeroPointPairs));
}

/// fromYAML will either serialize or deserialize a QuantTensorSpec.
static void formatYAML(llvm::yaml::IO &io, QuantTensorSpec &quantTensorSpec) {
  // If we are inputting a YAML file, then we need to do some extra work to
  // construct the QuantTensorSpec.
  if (!io.outputting())
    return fromYAML(io, quantTensorSpec);

  // Otherwise, we are outputting to a YAML format.
  bool isQuantized = quantTensorSpec.isQuantized();
  // We do not want the shape string to include the quantization information. We
  // just want it to include the shape string (of the form dim0xdim1x...xdtype).
  // So, we downcast the QuantTensorSpec to a TensorSpec and get the string
  // shape serialization from the TensorSpec.
  std::string shapeStr =
      static_cast<M::TensorSpec>(quantTensorSpec).getAsString();
  io.mapRequired("shape", shapeStr);
  io.mapOptional("quantized", isQuantized, false);
  // If the tensor is quantized, then we grab the quant parameters and add them
  // to the YAML object.
  if (quantTensorSpec.isQuantized()) {
    auto &quantSpec = quantTensorSpec.getQuantSpec();
    auto params = quantSpec.getParams();
    auto kind = M::QuantSpec::kindAsString(quantSpec.getKind());
    io.mapRequired("kind", kind);
    if (quantSpec.getKind() == M::QuantSpec::Kind::uniformPerAxis) {
      auto quantDim = quantSpec.getQuantDim();
      io.mapRequired("quant_dim", quantDim);
    }
    if (!params.empty()) {
      // The yamlization does not accept an ArrayRef, so we need to construct a
      // SmallVector for serialization.
      SmallVector<M::QuantSpec::ScaleZeroPointPair> paramsVec(params.begin(),
                                                              params.end());
      io.mapRequired("params", paramsVec);
    }
  }
}

/// Provide the QuantTensorSpec mapping trait for YAML conversion.
void llvm::yaml::MappingTraits<M::QuantTensorSpec>::mapping(
    IO &io, M::QuantTensorSpec &quantTensorSpec) {
  return formatYAML(io, quantTensorSpec);
}

//===----------------------------------------------------------------------===//
// ModelIOSpec
//===----------------------------------------------------------------------===//

namespace llvm::yaml {
template <>
struct MappingTraits<M::ModelSpec::ModelIOSpec> {
  static void mapping(IO &io, M::ModelSpec::ModelIOSpec &inputOutput) {
    // The name may be empty, so we map that as optional.
    io.mapOptional("name", inputOutput.first);
    // We need to call formatYAML on the QuantTensorSpec to avoid another
    // nesting in the YAML dictionary.
    formatYAML(io, inputOutput.second);
  }
};
} // namespace llvm::yaml

/// Use LLVM_YAML_IS_SEQUENCE_VECTOR to specify SequenceTraits<> on the
/// SmallVector<ModelIOSpec> type.
LLVM_YAML_IS_SEQUENCE_VECTOR(M::ModelSpec::ModelIOSpec);

//===----------------------------------------------------------------------===//
// M::ModelSpec::NameSubstitutions
//===----------------------------------------------------------------------===//

namespace llvm::yaml {
template <>
struct CustomMappingTraits<M::ModelSpec::NameSubstitutions> {
  static void inputOne(IO &io, StringRef key,
                       M::ModelSpec::NameSubstitutions &nameSubstitutions) {
    // Note that the YAML machinery ensures key is not already in the map.
    assert(nameSubstitutions.preferredToNativeMap.count(key) == 0 &&
           "model name already in substitution");
    io.mapRequired(key.str().c_str(),
                   nameSubstitutions.preferredToNativeMap[key]);
  }

  static void output(IO &io,
                     M::ModelSpec::NameSubstitutions &nameSubstitutions) {
    for (auto &kv : nameSubstitutions.preferredToNativeMap)
      io.mapRequired(kv.getKey().str().c_str(), kv.getValue());
  }
};
} // namespace llvm::yaml

//===----------------------------------------------------------------------===//
// M::ModelSpec::ModelBinary
//===----------------------------------------------------------------------===//

namespace llvm::yaml {
template <>
struct MappingTraits<M::ModelSpec::ModelBinary> {
  static void mapping(IO &io, M::ModelSpec::ModelBinary &binary) {
    io.mapRequired("format", binary.format);
    io.mapRequired("url", binary.url);

    // Do not print the "archived_file" field if we are outputting and the
    // weights.archivedFile is empty.
    if (!io.outputting() || !binary.archivedFile.empty())
      io.mapOptional("archived_file", binary.archivedFile);

    if (!io.outputting() || !binary.sha1sum.empty())
      io.mapOptional("sha1sum", binary.sha1sum);

    if (!io.outputting() || !binary.metadata.empty())
      io.mapOptional("metadata", binary.metadata);

    io.mapOptional("input_substitutions", binary.inputSubstitutions);
    io.mapOptional("output_substitutions", binary.outputSubstitutions);
  }
  /// Check if the binary format and URL are valid.
  static std::string validate(IO &io, M::ModelSpec::ModelBinary &binary) {
    // Check that the format is valid.
    // TODO: The format will be an enum in the future.
    if (binary.format != "bef" && binary.format != "onnx" &&
        binary.format != "tensorflow-graphdef" &&
        binary.format != "tensorflow-savedmodel" && binary.format != "tflite")
      return std::string("the ") + binary.format +
             " format specified is not valid";

    // Check that the URL is valid.
    llvm::Regex re(
        R"(^(https?)://((([^\/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?))");
    if (!re.match(binary.url))
      return std::string("the ") + binary.url + " URL specified is not valid";

    // Return an empty string to signify that there is no error.
    return std::string{};
  }
};
} // namespace llvm::yaml

/// Use LLVM_YAML_IS_SEQUENCE_VECTOR to specify SequenceTraits<> on the
/// SmallVector<M::ModelSpec::ModelBinary> and
/// SmallVector<M::ModelSpec::NameSubstitution> types.
LLVM_YAML_IS_SEQUENCE_VECTOR(M::ModelSpec::ModelBinary);
LLVM_YAML_IS_SEQUENCE_VECTOR(M::ModelSpec::NameSubstitutions);

//===----------------------------------------------------------------------===//
// ModelSpec
//===----------------------------------------------------------------------===//

/// Provide the ModelSpec mapping trait for YAML conversion.
void llvm::yaml::MappingTraits<M::ModelSpec>::mapping(IO &io,
                                                      M::ModelSpec &modelSpec) {
  io.mapRequired("name", modelSpec.name);
  if (!io.outputting() || !modelSpec.task.empty())
    io.mapOptional("task", modelSpec.task);
  if (!io.outputting() || !modelSpec.description.empty())
    io.mapOptional("description", modelSpec.description);
  io.mapRequired("model", modelSpec.model);
  io.mapRequired("inputs", modelSpec.inputs);
  io.mapRequired("outputs", modelSpec.outputs);
}

namespace {
/// Check if input/output name substitutions are valid with respect to given
/// input/output model names. Returns non-empty error string if not.
std::string validateNameSubstitutions(
    StringRef format, StringRef inputOrOutput,
    ArrayRef<M::ModelSpec::ModelIOSpec> names,
    const M::ModelSpec::NameSubstitutions &nameSubstitutions) {
  llvm::StringMap<std::string> nativeToPreferredMap;
  for (const auto &kv : nameSubstitutions.preferredToNativeMap) {
    // The 'preferred' name must exist.
    if (llvm::find_if(names, [&kv](const M::ModelSpec::ModelIOSpec &spec) {
          return spec.first == kv.getKey();
        }) == names.end()) {
      return (Twine("No such preferred ") + inputOrOutput + " name '" +
              kv.getKey() + "' for " + inputOrOutput +
              " substitution in format " + format)
          .str();
    }
    // The substitution must be bijective.
    if (!nativeToPreferredMap.try_emplace(kv.getValue(), kv.getKey()).second) {
      return (Twine("Already substituted native ") + inputOrOutput + " name '" +
              kv.getValue() + "' in format " + format)
          .str();
    };
  }
  // No error.
  return {};
}
} // namespace

std::string
llvm::yaml::MappingTraits<M::ModelSpec>::validate(IO &io,
                                                  M::ModelSpec &modelSpec) {
  for (const auto &modelBinary : modelSpec.model) {
    std::string err =
        validateNameSubstitutions(modelBinary.format, "input", modelSpec.inputs,
                                  modelBinary.inputSubstitutions);
    if (!err.empty())
      return err;

    err = validateNameSubstitutions(modelBinary.format, "output",
                                    modelSpec.outputs,
                                    modelBinary.outputSubstitutions);
    if (!err.empty())
      return err;
  }
  // No error.
  return {};
}

M::ErrorOr<M::ModelSpec>
M::loadModelSpecFromYAML(std::unique_ptr<llvm::MemoryBuffer> yamlBuffer) {
  llvm::yaml::Input yin(yamlBuffer->getBuffer());
  M::ModelSpec modelSpec;
  yin >> modelSpec;
  if (yin.error())
    return Error("unable to parse the model spec from the YAML buffer");
  return modelSpec;
}
