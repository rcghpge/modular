//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MODialect/MOTypes.h"
#include "GenericML/MODialect/MODialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// Tensor
//===----------------------------------------------------------------------===//

/// Verifies that the given dimensions represent a valid tensor shape, and
/// that the element type in DType form is also representable as an MLIR type.
LogicalResult TensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<int64_t> shape, Type elementType) {
  // TODO: Relax this once we have alignment with MGP on
  //  Dynamic Ranks
  if (shape.empty())
    return emitError() << "shape must have rank greather than 0";
  for (auto dim : shape) {
    if (dim <= 0 && dim != ShapedType::kDynamic) {
      return emitError() << "shape dimensions must be strictly positive or the "
                            "distinguished '?' unknown dimension.";
    }
  }

  return success();
}

/// Returns clone with given element type, which must be representable as a
/// dtype. Required by ShapedTypeInference
ShapedType TensorType::cloneWith(Optional<ArrayRef<int64_t>> shape,
                                 Type elementType) const {
  ArrayRef<int64_t> newShape = shape.value_or(getShape());
  return TensorType::get(getContext(), newShape, elementType);
}

/// Does the Tensor have a static Rank?  Required by ShapedTypeInferece
/// TODO : Relax this once we reach agreement with MGP on dynamic rank tensors.
bool TensorType::hasRank() const { return true; }

// Get the Rank of the Tensor.  Required by ShapedTypeInference
size_t TensorType::getRank() const { return getShape().size(); }

//===----------------------------------------------------------------------===//
// MODialect type support
//===----------------------------------------------------------------------===//

void MODialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "GenericML/MODialect/MOTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "GenericML/MODialect/MOTypes.cpp.inc"
