//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SDK/EngineCLI/Common/APIBridge.h"
#include "Support/AssertStream.h"
#include "Support/ML/TensorSpec.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

using namespace M;
using namespace M::EngineCLI;

M_Dtype M::EngineCLI::dtypeToCAPI(DType dty) {
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
  case DType::bf16:
    return M_BFLOAT16;
  case DType::kBool:
    return M_BOOL;
  default:
    return M_UNKNOWN;
  }
}

DType M::EngineCLI::capiToDType(M_Dtype dty) {
  switch (dty) {
  case M_INT8:
    return DType::si8;
  case M_INT16:
    return DType::si16;
  case M_INT32:
    return DType::si32;
  case M_INT64:
    return DType::si64;
  case M_UINT8:
    return DType::ui8;
  case M_UINT16:
    return DType::ui16;
  case M_UINT32:
    return DType::ui32;
  case M_UINT64:
    return DType::ui64;
  case M_FLOAT16:
    return DType::f16;
  case M_FLOAT32:
    return DType::f32;
  case M_FLOAT64:
    return DType::f64;
  case M_BFLOAT16:
    return DType::bf16;
  case M_BOOL:
    return DType::kBool;
  default:
    return DType::invalid;
  }
}

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
                                       dtypeToCAPI(spec.getEltType()), name),
                       M_freeTensorSpec);
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
  StatusPtr status{M_newStatus(), M_freeStatus};
  if (!status)
    return Error("Unable to create status");
  TorchInputSpecPtr cSpec{
      M_newTorchInputSpec(translatedShape.data(), /*dimNames=*/nullptr,
                          translatedRank, dtypeToCAPI(spec.getEltType()),
                          status.get()),
      M_freeTorchInputSpec};
  if (M_isError(status.get()))
    return Error(M_getError(status.get()));
  return cSpec;
}

TensorSpec M::EngineCLI::capiToTensorSpec(const M_TensorSpec *spec) {
  if (M_isDynamicRanked(spec))
    return TensorSpec(kDynamicallyRanked, capiToDType(M_getDtype(spec)));
  SmallVector<int64_t, kMaxRank> shape;
  int64_t rank = M_getRank(spec);
  assert(rank >= 0 && "unexpected dynamic rank");
  for (size_t axis = 0, e = (size_t)rank; axis < e; ++axis) {
    int64_t cDim = M_getDimAt(spec, axis);
    shape.push_back(cDim == M_getDynamicDimensionValue()
                        ? mlir::ShapedType::kDynamic
                        : cDim);
  }
  return TensorSpec(shape, capiToDType(M_getDtype(spec)));
}
