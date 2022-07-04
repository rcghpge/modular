//===- QuantSpec.cpp ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/QuantSpec.h"
#include "GraphRT/BEF/BEFTypes.h"
#include "Support/ErrorOr.h"
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

/// Returns a string representation of the kind and the quantized dimension if
/// the latter is relevant to the kind.
static std::string getKindAndDimAsString(const QuantSpec *qSpec) {
  Twine kind(QuantSpec::kindAsString(qSpec->getKind()));
  // Add the quandDim to the string if it the kind is uniformPerAxis.
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
  os << " ";
  getQuantSpec().print(os);
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

ErrorOr<QuantTensorSpec> QuantTensorSpec::get(BEFType type) {
  SmallVector<int64_t> dims;
  auto tensorType = dyn_cast<BEFTensorType>(type);
  if (!tensorType)
    return Error("BEFType cannot be cast to BEFTensorType");

  auto befType = tensorType->decode(dims);
  if (auto qType = dyn_cast<BEFQuantizedType>(befType)) {
    float scale;
    int64_t zeroPoint;
    auto [storageType, expressedType] = qType->decode(scale, zeroPoint);
    return QuantTensorSpec(TensorSpec(dims, storageType.getKind()),
                           QuantSpec::getUniform(scale, zeroPoint));
  }
  if (auto qType = dyn_cast<BEFQuantizedPerAxisType>(befType)) {
    SmallVector<float> scales;
    SmallVector<int64_t> zeroPoints;
    uint8_t quantDim;
    auto [storageType, expressedType] =
        qType->decode(scales, zeroPoints, quantDim);

    return QuantTensorSpec(
        TensorSpec(dims, storageType.getKind()),
        QuantSpec::getUniformPerAxis(quantDim, scales, zeroPoints));
  }
  // BEFType are GenericML DType kinds.
  return QuantTensorSpec(TensorSpec(dims, befType.getKind()), {});
}
