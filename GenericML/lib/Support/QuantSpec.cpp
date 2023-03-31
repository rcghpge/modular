//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/QuantSpec.h"
#include "GraphRT/MEF/MEFTypes.h"
#include "Support/ErrorOr.h"
#include "Support/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

//===----------------------------------------------------------------------===//
// Stringification and printing methods
//===----------------------------------------------------------------------===//

StringRef QuantSpec::kindAsString(QuantSpec::Kind kind) {
  switch (kind) {
  case QuantSpec::Kind::invalid:
    return "invalid";
  case QuantSpec::Kind::notQuantized:
    return "notQuantized";
  case QuantSpec::Kind::uniform:
    return "uniform";
  case QuantSpec::Kind::uniformPerAxis:
    return "uniformPerAxis";
  }
  llvm_unreachable("unknown QuantSpec kind");
}

QuantSpec::Kind QuantSpec::kindFromString(StringRef kind) {
  return llvm::StringSwitch<QuantSpec::Kind>(kind)
      .Case("invalid", QuantSpec::Kind::invalid)
      .Case("notQuantized", QuantSpec::Kind::notQuantized)
      .Case("uniform", QuantSpec::Kind::uniform)
      .Case("uniformPerAxis", QuantSpec::Kind::uniformPerAxis)
      .Default(QuantSpec::Kind::invalid);
}

/// Returns a string representation of the kind and the quantized dimension if
/// the latter is relevant to the kind.
static std::string getKindAndDimAsString(const QuantSpec *qSpec) {
  Twine kind(QuantSpec::kindAsString(qSpec->getKind()));
  // Add the quantDim to the string if it the kind is uniformPerAxis.
  if (qSpec->getKind() == QuantSpec::Kind::uniformPerAxis)
    return (kind + ":" + Twine(qSpec->getQuantDim())).str();
  return kind.str();
}

void QuantSpec::print(raw_ostream &os) const {
  os << getKindAndDimAsString(this);
  auto params = getParams();
  if (!params.empty()) {
    os << " {";
    llvm::interleave(
        getParams(),
        [&](ScaleZeroPointPair p) { os << p.scale << ":" << p.zeroPoint; },
        [&]() { os << ", "; });
    os << "}";
  }
}

std::string QuantSpec::getAsString() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  print(os);
  return os.str();
}

void QuantSpec::dump() const { print(llvm::errs()); }

void QuantTensorSpec::print(raw_ostream &os) const {
  TensorSpec::print(os);
  if (isQuantized()) {
    os << " ";
    getQuantSpec().print(os);
  }
}

std::string QuantTensorSpec::getAsString() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  print(os);
  return os.str();
}

void QuantTensorSpec::dump() const { print(llvm::errs()); }

//===----------------------------------------------------------------------===//
// Static helpers
//===----------------------------------------------------------------------===//

QuantSpec QuantSpec::getUniformPerAxis(uint8_t quantDim, ArrayRef<float> scales,
                                       ArrayRef<int64_t> zeroPoints) {
  // TODO: change this to take ArrayRef<ScaleZeroPointPair>, and change the
  // MEF encoding to encode similarly.
  return QuantSpec(
      Kind::uniformPerAxis, quantDim,
      map_to_vector(llvm::zip(scales, zeroPoints), [](auto paramPair) {
        auto [scale, zeroPoint] = paramPair;
        return ScaleZeroPointPair{scale, static_cast<int16_t>(zeroPoint)};
      }));
}

ErrorOr<QuantTensorSpec> QuantTensorSpec::get(MEFType type) {
  SmallVector<int64_t> dims;
  auto tensorType = dyn_cast<MEFTensorType>(type);
  if (!tensorType)
    return Error("MEFType cannot be cast to MEFTensorType");

  auto mefType = tensorType->decode(dims);
  if (auto qType = dyn_cast<MEFQuantizedType>(mefType)) {
    float scale;
    int64_t zeroPoint;
    auto [storageType, expressedType] = qType->decode(scale, zeroPoint);
    return QuantTensorSpec(TensorSpec(dims, storageType.getKind()),
                           QuantSpec::getUniform(scale, zeroPoint));
  }
  if (auto qType = dyn_cast<MEFQuantizedPerAxisType>(mefType)) {
    SmallVector<float> scales;
    SmallVector<int64_t> zeroPoints;
    uint8_t quantDim;
    auto [storageType, expressedType] =
        qType->decode(scales, zeroPoints, quantDim);

    return QuantTensorSpec(
        TensorSpec(dims, storageType.getKind()),
        QuantSpec::getUniformPerAxis(quantDim, scales, zeroPoints));
  }
  // MEFType are GenericML DType kinds.
  return QuantTensorSpec(TensorSpec(dims, mefType.getKind()), {});
}
