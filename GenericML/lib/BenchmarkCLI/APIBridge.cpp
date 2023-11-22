//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/BenchmarkCLI/APIBridge.h"
#include "Support/ML/TensorSpec.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

using namespace M;
using namespace M::BenchmarkCLI;

M_Dtype M::BenchmarkCLI::dtypeToCAPI(DType dty) {
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

DType M::BenchmarkCLI::capiToDType(M_Dtype dty) {
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
  case M_BOOL:
    return DType::kBool;
  default:
    return DType::invalid;
  }
}

TensorSpecPtr M::BenchmarkCLI::tensorSpecToCAPI(const char *name,
                                                const TensorSpec &spec) {
  // TODO: There is no way to represent dynamically-ranked tensors (#26162) or
  // static-rank tensors with dynamic dimensions (#26301).  The C API only
  // documents a way to create specs for fully-static shapes.
  assert(spec.isStatic() && "C API cannot represent non-static shapes");

  // Copy is required because TensorShape stores ssize_t, but API takes
  // int64_t, and these are not necessarily the same.
  SmallVector<int64_t, 8> shapeCopy;
  std::copy(spec.begin(), spec.end(), std::back_inserter(shapeCopy));
  return TensorSpecPtr(M_newTensorSpec(shapeCopy.data(), spec.getRank(),
                                       dtypeToCAPI(spec.getEltType()), name),
                       M_freeTensorSpec);
}

TensorSpec M::BenchmarkCLI::capiToTensorSpec(const M_TensorSpec *spec) {
  SmallVector<int64_t, 8> shape;
  size_t rank = M_getRank(spec);
  for (size_t axis = 0; axis < rank; ++axis) {
    int64_t cDim = M_getDimAt(spec, axis);
    shape.push_back(cDim == -1 ? mlir::ShapedType::kDynamic : cDim);
  }
  return TensorSpec(shape, capiToDType(M_getDtype(spec)));
}
