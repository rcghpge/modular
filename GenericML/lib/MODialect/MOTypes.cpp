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
// TensorSpec
//===----------------------------------------------------------------------===//

/// Verifies that the given dimensions represent a valid tensor shape, and
/// that the element type in DType form is also representable as an MLIR type.
LogicalResult
TensorSpecType::verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<int64_t> shape, Type elementType) {
  // TODO: Relax this once we have alignment with MOP on
  //  Dynamic Ranks
  if (shape.empty())
    return emitError() << "shape must have rank greather than 0";
  for (auto dim : shape) {
    if (dim <= 0 && dim != ShapedType::kDynamicSize) {
      return emitError() << "shape dimensions must be strictly positive or the "
                            "distinguished '?' unknown dimension.";
    }
  }

  return success();
}

/// Returns clone with given element type, which must be representable as a
/// dtype. Required by ShapedTypeInference
ShapedType TensorSpecType::cloneWith(Optional<ArrayRef<int64_t>> shape,
                                     Type elementType) const {
  ArrayRef<int64_t> newShape = shape.value_or(getShape());
  return TensorSpecType::get(getContext(), newShape, elementType);
}

/// Does the Tensor have a static Rank?  Required by ShapedTypeInferece
/// TODO : Relax this once we reach agreement with MOP on dynamic rank tensors.
bool TensorSpecType::hasRank() const { return true; }

// Get the Rank of the Tensor.  Required by ShapedTypeInference
size_t TensorSpecType::getRank() const { return getShape().size(); }

//===----------------------------------------------------------------------===//
// Tensor
//===----------------------------------------------------------------------===//

/// Returns clone of tensor type. Required by ShapedTypeInference
ShapedType TensorType::cloneWith(Optional<ArrayRef<int64_t>> shape,
                                 Type elementType) const {
  auto tensorSpec = mlir::dyn_cast<TensorSpecType>(
      getTensorSpec().cloneWith(shape, elementType));
  return TensorType::get(getContext(), tensorSpec);
}

/// Returns true since all tensors have statically known rank. Required by
/// ShapedTypeInference
bool TensorType::hasRank() const { return getTensorSpec().hasRank(); }

/// Returns tensor spec's shape. Required by ShapedTypeInference
ArrayRef<int64_t> TensorType::getShape() const {
  return getTensorSpec().getShape();
}

/// Returns tensor spec's element type. Required by ShapedTypeInference
Type TensorType::getElementType() const {
  return getTensorSpec().getElementType();
}
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
