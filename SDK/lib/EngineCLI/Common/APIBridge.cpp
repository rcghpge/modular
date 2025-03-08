//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SDK/EngineCLI/Common/APIBridge.h"
#include "GenericML/API/c/detail/helpers.h"
#include "Support/AssertStream.h"
#include "Support/ML/TensorSpec.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

using namespace M;
using namespace M::EngineCLI;

TensorSpecPtr M::EngineCLI::tensorSpecToCAPI(const char *name,
                                             const TensorSpec &spec) {
  int64_t translatedRank;
  SmallVector<int64_t, kMaxRank> translatedShape;
  if (spec.hasRank()) {
    translatedRank = spec.getRank();
    for (ssize_t dim : spec) {
      if (dim < 0)
        translatedShape.push_back(M_getDynamicDimensionValue());
      else
        translatedShape.push_back(dim);
    }
  } else {
    translatedRank = M_getDynamicRankValue();
  }
  return TensorSpecPtr(M_newTensorSpec(translatedShape.data(), translatedRank,
                                       M::Detail::mapDType(spec.getEltType()),
                                       name));
}

ErrorOr<TorchInputSpecPtr>
M::EngineCLI::tensorSpecToCAPITorchSpec(const TensorSpec &spec) {
  int64_t translatedRank;
  SmallVector<int64_t, kMaxRank> translatedShape;
  if (spec.hasRank()) {
    translatedRank = spec.getRank();
    for (ssize_t dim : spec) {
      if (dim < 0)
        translatedShape.push_back(M_getDynamicDimensionValue());
      else
        translatedShape.push_back(dim);
    }
  } else {
    translatedRank = M_getDynamicRankValue();
  }
  StatusPtr status{M_newStatus()};
  if (!status)
    return Error("Unable to create status");
  TorchInputSpecPtr cSpec{M_newTorchInputSpec(
      translatedShape.data(), /*dimNames=*/nullptr, translatedRank,
      M::Detail::mapDType(spec.getEltType()),
      /*device=*/nullptr, status.get())};
  if (M_isError(status.get()))
    return Error(M_getError(status.get()));
  return cSpec;
}

TensorSpec M::EngineCLI::capiToTensorSpec(const M_TensorSpec *spec) {
  if (M_isDynamicRanked(spec))
    return TensorSpec(kDynamicallyRanked,
                      M::Detail::mapDType(M_getDtype(spec)));
  SmallVector<int64_t, kMaxRank> shape;
  int64_t rank = M_getRank(spec);
  assert(rank >= 0 && "unexpected dynamic rank");
  for (size_t axis = 0, e = (size_t)rank; axis < e; ++axis) {
    int64_t cDim = M_getDimAt(spec, axis);
    shape.push_back(cDim == M_getDynamicDimensionValue()
                        ? mlir::ShapedType::kDynamic
                        : cDim);
  }
  return TensorSpec(shape, M::Detail::mapDType(M_getDtype(spec)));
}
