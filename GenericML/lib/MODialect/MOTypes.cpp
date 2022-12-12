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
// ShapeParameter
//===----------------------------------------------------------------------===//

static void printShapeParameter(AsmPrinter &p, ArrayRef<int64_t> dims) {
  for (size_t i = 0, e = dims.size(); i < e; i++) {
    if (ShapedType::isDynamic(dims[i])) {
      p << '?';
      p << 'x';
    } else if (dims[i] == kDynamicRank) {
      p << '*';
      // TODO: Remove this conditional when we can support partial ranks
      if (i < e - 1)
        p << 'x';
    } else {
      p << dims[i];
      p << 'x';
    }
  }
}

FailureOr<llvm::SmallVector<int64_t>> parseShapeParameter(AsmParser &p) {
  llvm::SmallVector<int64_t> dims{};
  if (!p.parseOptionalStar()) {
    // This is an unranked tensor.
    dims.push_back(kDynamicRank);
  } else {
    // This should be a ranked tensor
    if (p.parseDimensionList(dims, /*allowDynamic=*/true))
      return failure();
  }
  return dims;
}

//===----------------------------------------------------------------------===//
// Tensor
//===----------------------------------------------------------------------===//

/// Verifies that the given dimensions represent a valid tensor shape, and
/// that the element type in DType form is also representable as an MLIR type.
LogicalResult TensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<int64_t> shape, Type elementType) {
  for (auto dim : shape) {
    if (dim <= 0 && dim != ShapedType::kDynamic && dim != kDynamicRank) {
      return emitError()
             << "shape dimensions must be strictly positive or the "
                "distinguished '?' unknown dimension or '*' unknown rank.";
    }
  }
  return success();
}

/// Returns clone with given element type, which must be representable as a
/// dtype. Required by ShapedTypeInference
ShapedType TensorType::cloneWith(Optional<ArrayRef<int64_t>> shape,
                                 Type elementType) const {
  ArrayRef<int64_t> newShape =
      shape.has_value() ? ArrayRef<int64_t>(shape.value()) : getShape();
  return TensorType::get(getContext(), newShape, elementType);
}

/// Does the Tensor have a static Rank?  Required by ShapedTypeInferece
bool TensorType::hasRank() const {
  auto shape = getShape();
  return std::find(shape.begin(), shape.end(), kDynamicRank) != shape.end();
}

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
