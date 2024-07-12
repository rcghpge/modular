//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is a file serializes and deserializes the types found in GenericML such
// as TensorSpec, QuantSpec, and ModelSpec.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/FormatYAML.h"
#include "GenericML/Support/ModelSpec.h"
#include "GenericML/Support/QuantSpec.h"
#include "GenericML/Support/ResultInfo.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <utility>

using namespace M;

//===----------------------------------------------------------------------===//
// QuantTensorSpec
//===----------------------------------------------------------------------===//

namespace llvm::yaml {
/// Provide a forwarding from ssize_t to int64_t for YAML ScalarTraits.
template <typename T>
struct ScalarTraits<
    T, std::enable_if_t<!llvm::is_detected<ScalarTraits, T>::value &&
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
LLVM_YAML_IS_SEQUENCE_VECTOR(M::QuantSpec::ScaleZeroPointPair)

/// fromYAML will deserialize a QuantTensorSpec from YAML.
static void fromYAML(llvm::yaml::IO &io, QuantTensorSpec &quantTensorSpec) {
  // Read the "shape" field and parse it into a TensorSpec.
  std::string shapeStr;
  io.mapOptional("shape", shapeStr);
  if (shapeStr.empty()) {
    quantTensorSpec = QuantTensorSpec();
    return;
  }
  auto errorOrTensorSpec = TensorSpec::parseFromString(shapeStr);
  if (failed(errorOrTensorSpec)) {
    io.setError(errorOrTensorSpec.getError());
    return;
  }
  TensorSpec tensorSpec = *errorOrTensorSpec;

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
  StringRef kindStr("");
  io.mapRequired("kind", kindStr);
  auto kind = QuantSpec::kindFromString(kindStr);

  // The "quant_dim" field is optionally available when the quant kind is
  // M::QuantSpec::Kind::uniformPerAxis
  uint8_t quantDim = 0;
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
    // name and data may be empty, so we map them as optional.
    io.mapOptional("name", inputOutput.name);
    io.mapOptional("data", inputOutput.data);
    io.mapOptional("size", inputOutput.size);
    io.mapOptional("sha1sum", inputOutput.sha1sum);

    // We need to call formatYAML on the QuantTensorSpec to avoid another
    // nesting in the YAML dictionary.
    formatYAML(io, inputOutput.spec);
  }
};
} // namespace llvm::yaml

/// Use LLVM_YAML_IS_SEQUENCE_VECTOR to specify SequenceTraits<> on the
/// SmallVector<ModelIOSpec> type.
LLVM_YAML_IS_SEQUENCE_VECTOR(M::ModelSpec::ModelIOSpec)

//===----------------------------------------------------------------------===//
// M::ModelSpec::MetricSpec
//===----------------------------------------------------------------------===//

namespace llvm::yaml {
template <>
struct MappingTraits<M::ModelSpec::Metric> {
  static void mapping(IO &io, M::ModelSpec::Metric &metric) {
    io.mapRequired("name", metric.name);
    io.mapOptional("expected", metric.expected);
    io.mapOptional("atol", metric.atol);
    io.mapOptional("rtol", metric.rtol);
  }
};
} // namespace llvm::yaml

/// Use LLVM_YAML_IS_SEQUENCE_VECTOR to specify SequenceTraits<> on the
/// SmallVector<Metric> type.
LLVM_YAML_IS_SEQUENCE_VECTOR(M::ModelSpec::Metric)

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

static bool isValidModelFormat(StringRef format) {
  return llvm::is_contained({"faux", "mef", "onnx", "torchscript"}, format);
}

namespace llvm::yaml {
template <>
struct MappingTraits<M::ModelSpec::ModelBinary> {
  static void mapping(IO &io, M::ModelSpec::ModelBinary &binary) {
    io.mapRequired("format", binary.format);
    io.mapRequired("url", binary.url);

    if (!io.outputting() || !binary.size.empty())
      io.mapOptional("size", binary.size);

    if (!io.outputting() || !binary.sha1sum.empty())
      io.mapOptional("sha1sum", binary.sha1sum);

    if (!io.outputting() || !binary.metadata.empty())
      io.mapOptional("metadata", binary.metadata);

    if (!io.outputting() || !binary.dependencies.empty())
      io.mapOptional("dependencies", binary.dependencies);

    io.mapOptional("input_substitutions", binary.inputSubstitutions);
    io.mapOptional("output_substitutions", binary.outputSubstitutions);
  }
  /// Check if the binary format and URL are valid.
  static std::string validate(IO &io, M::ModelSpec::ModelBinary &binary) {
    // Check that the format is valid.
    // TODO: The format will be an enum in the future.
    if (!isValidModelFormat(binary.format))
      return std::string("the ") + binary.format +
             " format specified is not valid";

    // Check that the URL is valid.
    llvm::Regex re(R"(^(https?|file|s3)://)");
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
LLVM_YAML_IS_SEQUENCE_VECTOR(M::ModelSpec::ModelBinary)
LLVM_YAML_IS_SEQUENCE_VECTOR(M::ModelSpec::NameSubstitutions)

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
  if (!io.outputting() || !modelSpec.data_name.empty())
    io.mapOptional("data_name", modelSpec.data_name);
  if (!io.outputting() || !modelSpec.data_description.empty())
    io.mapOptional("data_description", modelSpec.data_description);
  io.mapRequired("model", modelSpec.model);
  io.mapRequired("inputs", modelSpec.inputs);
  io.mapRequired("outputs", modelSpec.outputs);
  io.mapOptional("metrics", modelSpec.metrics);
  io.mapOptional("compile_specs", modelSpec.compile_specs);
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
          return spec.name == kv.getKey();
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

void llvm::yaml::MappingTraits<M::TensorResultInfo>::mapping(
    IO &io, M::TensorResultInfo &info) {
  io.mapRequired("name", info.name);
  io.mapOptional("shape", info.shape);
}

void llvm::yaml::MappingTraits<M::VariantResultInfo>::mapping(
    IO &io, M::VariantResultInfo &info) {
  io.mapRequired("name", info.name);
  io.mapRequired("variant", info.variant);
}

void llvm::yaml::MappingTraits<std::shared_ptr<M::ResultInfo>>::mapping(
    IO &io, std::shared_ptr<M::ResultInfo> &info) {
  info->map(io);
}

ErrorOr<AsyncRT::AnyAsyncValueRef>
M::getResultForBinaryOutput(const AsyncRT::AnyAsyncValueRef &out,
                            ProcessedResults &processedResults) {
  M::AsyncRT::CompactRuntimePtr runtime = out.getRuntime();

  if (GML::List *castListValue = dyn_cast<GML::List>(out)) {
    // Create a new list.
    AsyncRT::AnyAsyncValueRef out =
        AsyncRT::AnyAsyncValueRef::createReady<GML::List>(runtime);
    out.get<GML::List>().getElements().reserve(
        castListValue->getElements().size());
    for (GML::Variant &elem : castListValue->getElements()) {
      ErrorOr<AsyncRT::AnyAsyncValueRef> newElemOr =
          getResultForBinaryOutput(elem.getValue(), processedResults);
      if (newElemOr->isError())
        return newElemOr.takeError();
      out.get<GML::List>().getElements().push_back(newElemOr.takeValue());
    }
    return out;
  }
  if (GML::Tuple *castTupleValue = dyn_cast<GML::Tuple>(out)) {
    // Create a new tuple.
    AsyncRT::AnyAsyncValueRef out =
        AsyncRT::AnyAsyncValueRef::createReady<GML::Tuple>(runtime);
    out.get<GML::Tuple>().getElements().reserve(
        castTupleValue->getElements().size());
    for (GML::Variant &elem : castTupleValue->getElements()) {
      ErrorOr<AsyncRT::AnyAsyncValueRef> newElemOr =
          getResultForBinaryOutput(elem.getValue(), processedResults);
      if (newElemOr->isError())
        return newElemOr.takeError();
      out.get<GML::Tuple>().getElements().push_back(newElemOr.takeValue());
    }
    return out;
  }
  if (GML::Dict *castDictValue = dyn_cast<GML::Dict>(out)) {
    // Create a new dict.
    AsyncRT::AnyAsyncValueRef out =
        AsyncRT::AnyAsyncValueRef::createReady<GML::Dict>(runtime);
    out.get<GML::Dict>().getElements().reserve(
        castDictValue->getElements().size());
    for (auto &[key, value] : castDictValue->getElements()) {
      ErrorOr<AsyncRT::AnyAsyncValueRef> newKeyOr =
          getResultForBinaryOutput(key.getValue(), processedResults);
      ErrorOr<AsyncRT::AnyAsyncValueRef> newValueOr =
          getResultForBinaryOutput(value.getValue(), processedResults);
      if (newKeyOr.isError())
        return newKeyOr.takeError();
      if (newValueOr.isError())
        return newValueOr.takeError();
      out.get<GML::Dict>().getElements().insert(
          {newKeyOr.takeValue(), newValueOr.takeValue()});
    }
    return out;
  }
  if (Tensor *castTensorValue = dyn_cast<Tensor>(out)) {
    // Insert tensor if it isn't registered already.
    std::string name = processedResults.insertTensor(castTensorValue);
    // Replace tensor with `TensorResultInfo`. We re-export the tensor
    // information, so if the tensors differ in shape due to a reshape, they are
    // still the same if the underlying buffer (pointer and size) is the same.
    AsyncRT::AnyAsyncValueRef out =
        AsyncRT::AnyAsyncValueRef::createReady<TensorResultInfo>(
            runtime, name, *castTensorValue);
    return out;
  }
  // Copy references to primitive types (string, double, long, bool, None).
  if (isa<std::string>(out) || isa<float>(out) || isa<double>(out) ||
      isa<bool>(out) || isa<int32_t>(out) || isa<int64_t>(out) ||
      isa<uint32_t>(out) || isa<uint64_t>(out) || isa<GML::None>(out))
    return out.copy();
  else
    return Error("unknown result");
}

llvm::yaml::MappingTraits<
    M::GML::Variant>::NormalizedVariant::NormalizedVariant(IO &) {}

llvm::yaml::MappingTraits<M::GML::Variant>::NormalizedVariant::
    NormalizedVariant(IO &, M::GML::Variant &variant) {
  if (auto *castStringValue = dyn_cast<std::string>(variant.getValue()))
    stringValue = *castStringValue;
  else if (auto *castDoubleValue = dyn_cast<double>(variant.getValue()))
    doubleValue = *castDoubleValue;
  else if (auto *castLongValue = dyn_cast<int64_t>(variant.getValue()))
    longValue = *castLongValue;
  else if (auto *castBoolValue = dyn_cast<bool>(variant.getValue()))
    boolValue = *castBoolValue;
  else if (auto *castListValue = dyn_cast<GML::List>(variant.getValue()))
    listValue = *castListValue;
  else if (auto *castTupleValue = dyn_cast<GML::Tuple>(variant.getValue()))
    tupleValue = *castTupleValue;
  else if (auto *castDictValue = dyn_cast<GML::Dict>(variant.getValue()))
    dictValue = *castDictValue;
  else if (auto *castNoneValue = dyn_cast<GML::None>(variant.getValue()))
    noneValue = *castNoneValue;
  else if (auto *castTensorResultInfo =
               dyn_cast<TensorResultInfo>(variant.getValue()))
    tensorResultInfoValue = *castTensorResultInfo;
  // Unsupported types are not mapped.
}

M::GML::Variant
llvm::yaml::MappingTraits<M::GML::Variant>::NormalizedVariant::denormalize(
    IO &) {
  AsyncRT::Runtime &runtime = Runtime::getCurrentRuntime();
  M::GML::Variant variant;
  if (stringValue) {
    variant = AsyncRT::AnyAsyncValueRef::createReady<std::string>(runtime,
                                                                  *stringValue);
  } else if (doubleValue) {
    variant =
        AsyncRT::AnyAsyncValueRef::createReady<double>(runtime, *doubleValue);
  } else if (longValue) {
    variant =
        AsyncRT::AnyAsyncValueRef::createReady<int64_t>(runtime, *longValue);
  } else if (boolValue) {
    variant = AsyncRT::AnyAsyncValueRef::createReady<bool>(runtime, *boolValue);
  } else if (listValue) {
    variant =
        AsyncRT::AnyAsyncValueRef::createReady<GML::List>(runtime, *listValue);
  } else if (tupleValue) {
    variant = AsyncRT::AnyAsyncValueRef::createReady<GML::Tuple>(runtime,
                                                                 *tupleValue);
  } else if (dictValue) {
    variant =
        AsyncRT::AnyAsyncValueRef::createReady<GML::Dict>(runtime, *dictValue);
  } else if (noneValue) {
    variant =
        AsyncRT::AnyAsyncValueRef::createReady<GML::None>(runtime, *noneValue);
  }
  return variant;
}

void llvm::yaml::MappingTraits<M::GML::Variant>::mapping(
    IO &io, M::GML::Variant &variant) {
  MappingNormalization<NormalizedVariant, M::GML::Variant> keys(io, variant);
  io.mapOptional("string", keys->stringValue, std::optional<std::string>());
  io.mapOptional("double", keys->doubleValue, std::optional<double>());
  io.mapOptional("long", keys->longValue, std::optional<int64_t>());
  io.mapOptional("bool", keys->boolValue, std::optional<bool>());
  io.mapOptional("none", keys->noneValue, std::optional<M::GML::None>());
  io.mapOptional("list", keys->listValue, std::optional<M::GML::List>());
  io.mapOptional("tuple", keys->tupleValue, std::optional<M::GML::Tuple>());
  io.mapOptional("dict", keys->dictValue, std::optional<M::GML::Dict>());
  io.mapOptional("tensor-result-info", keys->tensorResultInfoValue,
                 std::optional<TensorResultInfo>());
}

void llvm::yaml::MappingTraits<M::GML::None>::mapping(IO &io,
                                                      M::GML::None &list) {}

void llvm::yaml::MappingTraits<M::GML::List>::mapping(IO &io,
                                                      M::GML::List &list) {
  io.mapRequired("list", list.getElements());
}

void llvm::yaml::MappingTraits<M::GML::Tuple>::mapping(IO &io,
                                                       M::GML::Tuple &tuple) {
  io.mapRequired("tuple", tuple.getElements());
}

void llvm::yaml::MappingTraits<YamlDictKeyValuePair>::mapping(
    IO &io, YamlDictKeyValuePair &kv) {
  io.mapRequired("key", kv.first);
  io.mapRequired("value", kv.second);
}

llvm::yaml::MappingTraits<M::GML::Dict>::NormalizedDict::NormalizedDict(IO &) {}

llvm::yaml::MappingTraits<M::GML::Dict>::NormalizedDict::NormalizedDict(
    IO &, M::GML::Dict &dict) {
  for (auto &entry : dict.getElements())
    normalizedDict.emplace_back(entry.first, entry.second);
}

M::GML::Dict
llvm::yaml::MappingTraits<M::GML::Dict>::NormalizedDict::denormalize(IO &) {
  M::GML::Dict dict;
  for (auto &entry : normalizedDict) {
    dict.getElements().insert(std::make_pair(entry.first, entry.second));
  }
  return dict;
}

void llvm::yaml::MappingTraits<M::GML::Dict>::mapping(IO &io,
                                                      M::GML::Dict &dict) {
  MappingNormalization<NormalizedDict, M::GML::Dict> keys(io, dict);
  io.mapRequired("dict", keys->normalizedDict);
}
