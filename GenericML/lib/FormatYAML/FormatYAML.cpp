//===- GenericML/FormatYAML.cpp -------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is a file serializes the types found in GenericML such as TensorSpec and
// QuantSpec.
//
//===----------------------------------------------------------------------===//

#include "GenericML/FormatYAML/FormatYAML.h"
#include "GenericML/Support/QuantSpec.h"

/// Provide a way to check if a template is defined so we can not define this
/// thing on platforms where ScalarTraits<T> is defined.
template <class T, std::size_t = sizeof(T)>
std::true_type is_complete_impl(T *);
std::false_type is_complete_impl(...);
template <class T>
using is_complete = decltype(is_complete_impl(std::declval<T *>()));

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

/// Provide the QuantTensorSpec mapping trait for YAML conversion.
void ::llvm::yaml::MappingTraits<M::QuantTensorSpec>::mapping(
    IO &io, M::QuantTensorSpec &quantTensorSpec) {
  SmallVector<ssize_t> dims = quantTensorSpec.getDims();
  std::string eltTypeStr = quantTensorSpec.getEltType().getAsString();
  bool isQuantized = quantTensorSpec.isQuantized();
  // We do not want the shape string to include the quantization information. We
  // just want it to include the shape string (of the form
  // dim0xdim1x...xelemType). So, we downcast the QuantTensorSpec to a
  // TensorSpec and get the string shape serialization from the TensorSpec.
  std::string shapeStr =
      static_cast<M::TensorSpec>(quantTensorSpec).getAsString();
  io.mapRequired("shape", shapeStr);
  io.mapRequired("quantized", isQuantized);
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
    if (params.size()) {
      // The yamlization does not accept an ArrayRef, so we need to construct a
      // SmallVector for serialization.
      SmallVector<M::QuantSpec::ScaleZeroPointPair> paramsVec(params.begin(),
                                                              params.end());
      io.mapRequired("params", paramsVec);
    }
  }
}
