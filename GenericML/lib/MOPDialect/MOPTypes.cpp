//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MOPDialect/MOPTypes.h"
#include "GenericML/MOPDialect/MOPDialect.h"
#include "Support/Compiler/ParsingUtils.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

#include "./ParsingUtils.h"
#include "./Utils.h"

using namespace M;
using namespace MOP;

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

/// Verifies the buffer size and align constraints are valid.
LogicalResult BufferType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 Optional<int64_t> size) {
  if (!isValidSize(size))
    return emitError() << "buffer size must be positive.";
  return success();
}

/// Implements ShapedType::cloneWith
ShapedType BufferType::cloneWith(Optional<ArrayRef<int64_t>> shape,
                                 Type dtype) const {
  assert(!shape && "cannot change shape in BufferType::cloneWith");
  assert(dtype.isUnsignedInteger(8) &&
         "invalid dtype for BufferType::cloneWith");
  return BufferType::get(getContext(), getSize());
}

/// Implements ShapedType::getElementType
Type BufferType::getElementType() const {
  return IntegerType::get(getContext(), /*width=*/8, IntegerType::Unsigned);
}

/// Implements ShapedType::hasRank
bool BufferType::hasRank() const { return true; }

// We defer BufferType::getShape till after ODS import since depends on impl.

//===----------------------------------------------------------------------===//
// ContextType
//===----------------------------------------------------------------------===//

// Nothing to implement.

//===----------------------------------------------------------------------===//
// TensorSpecType
//===----------------------------------------------------------------------===//

/// Parses a list of dimensions followed by a dtype, interpreting '?' as -1.
/// The dype is represented as an MLIR IntegerType or FloatType.
static ParseResult parseShapeAndDType(AsmParser &p,
                                      FailureOr<SmallVector<int64_t>> &shape,
                                      FailureOr<Type> &elementType) {
  shape.emplace();
  if (p.parseDimensionList(*shape, /*allowDynamic=*/true,
                           /*withTrailingX=*/true) ||
      parseDType(p, elementType))
    return failure();
  return success();
}

/// Prints a list of dimensions and dtype, interpreting -1 as '?'.
/// The dtype is represented as an MLIR IntegerType or FloatType.
static void printShapeAndDType(AsmPrinter &p, ArrayRef<int64_t> shape,
                               Type elementType) {
  llvm::interleave(
      shape, p.getStream(),
      [&](int64_t dim) {
        p << (dim == ShapedType::kDynamicSize ? "?" : std::to_string(dim));
      },
      "x");
  p << "x";
  p << mlirTypeToDType(elementType).getAsString();
}

/// Verifies that the given dimensions represent a valid tensor shape, and
/// that the MLIR element type is representable as a M::DType.
LogicalResult
TensorSpecType::verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<int64_t> shape, Type elementType) {
  if (shape.empty() || shape.size() > getMaximumRank())
    return emitError() << "shape must have rank between 1 and "
                       << getMaximumRank() << ".";
  for (auto dim : shape) {
    if (dim <= 0 && dim != ShapedType::kDynamicSize)
      return emitError() << "shape dimensions must be strictly positive or the "
                            "distinguished '?' unknown dimension.";
  }

  DType dtype = mlirTypeToDType(elementType);
  if (dtype.isInvalid())
    return emitError() << "element type '" << elementType
                       << "' cannot be represented as a DType.";

  return success();
}

size_t TensorSpecType::getRank() const { return getShape().size(); }

int64_t TensorSpecType::getNumDynamicDims() const {
  return llvm::count(getShape(), ShapedType::kDynamicSize);
}

Optional<int64_t> TensorSpecType::operator[](size_t i) const {
  assert(i < getShape().size() && "shape index out of range");
  int64_t dim = getShape()[i];
  return dim == ShapedType::kDynamicSize ? Optional<int64_t>() : dim;
}

Optional<int64_t> TensorSpecType::getNumElements() const {
  int64_t prod = 1;
  for (auto dim : getShape()) {
    if (dim == ShapedType::kDynamicSize)
      return {};
    prod *= dim;
  }
  return prod;
}

Optional<int64_t> TensorSpecType::getSizeInBytes() const {
  Optional<int64_t> numElems = getNumElements();
  if (!numElems)
    return {};
  return mlirTypeToDType(getElementType()).getSizeInBytes(*numElems);
}

/// Implements SubElementTypeInterface::walkImmediateSubElements.
void TensorSpecType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkTypesFn(getElementType());
}

/// Implements SubElementTypeInterface::replaceImmediateSubElements.
Type TensorSpecType::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  assert(replAttrs.empty());
  assert(replTypes.size() == 1);
  return TensorSpecType::get(getContext(), getShape(), replTypes[0]);
}

/// Implements ShapedTypeInterface::cloneWith.
ShapedType TensorSpecType::cloneWith(Optional<ArrayRef<int64_t>> shape,
                                     Type elementType) const {
  return TensorSpecType::get(getContext(), shape ? *shape : getShape(),
                             elementType ? elementType : getElementType());
}

/// Implements ShapedTypeInterface::hasRank.
bool TensorSpecType::hasRank() const { return true; }

//===----------------------------------------------------------------------===//
// TensorType
//===----------------------------------------------------------------------===//

/// Parses a compact tensor type description.
static ParseResult parseTensorSpec(AsmParser &p,
                                   FailureOr<TensorSpecType> &tensorSpec) {
  FailureOr<SmallVector<int64_t>> shape;
  FailureOr<Type> elementType;
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || parseShapeAndDType(p, shape, elementType))
    return mlir::failure();

  // Though the following get calls will invoke verify, the resulting errors
  // will be locationless and trigger a core dump. Try to be friendlier.
  if (failed(TensorSpecType::verify([&]() { return p.emitError(loc); }, *shape,
                                    *elementType)))
    return failure();

  tensorSpec.emplace(TensorSpecType::get(p.getContext(), *shape, *elementType));
  return success();
}

/// Prints a compact tensor type description.
static void printTensorSpec(AsmPrinter &p, TensorSpecType tensorSpec) {
  printShapeAndDType(p, tensorSpec.getShape(), tensorSpec.getElementType());
}

BufferType TensorType::getExpectedBufferType() const {
  return BufferType::get(getContext(), getTensorSpec().getSizeInBytes());
}

/// Implements SubElementTypeInterface::walkImmediateSubElements.
void TensorType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkTypesFn(getTensorSpec());
}

/// Implements SubElementTypeInterface::replaceImmediateSubElements.
Type TensorType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                             ArrayRef<Type> replTypes) const {
  assert(replAttrs.empty());
  assert(replTypes.size() == 1);
  TensorSpecType spec = mlir::dyn_cast<TensorSpecType>(replTypes[0]);
  assert(spec);
  return TensorType::get(getContext(), spec);
}

/// Implements ShapedTypeInterface::cloneWith.
ShapedType TensorType::cloneWith(Optional<ArrayRef<int64_t>> shape,
                                 Type elementType) const {
  TensorSpecType tensorSpec = TensorSpecType::get(
      getContext(), shape ? *shape : getTensorSpec().getShape(),
      elementType ? elementType : getTensorSpec().getElementType());
  return TensorType::get(getContext(), tensorSpec);
}

/// Implements ShapedTypeInterface::getElementType.
Type TensorType::getElementType() const {
  return getTensorSpec().getElementType();
}

/// Implements ShapedTypeInterface::hasRank.
bool TensorType::hasRank() const { return true; }

/// Implements ShapedTypeInterface::getShape.
ArrayRef<int64_t> TensorType::getShape() const {
  return getTensorSpec().getShape();
}

//===----------------------------------------------------------------------===//
// MOPDialect type support
//===----------------------------------------------------------------------===//

void MOPDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "GenericML/MOPDialect/MOPTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "GenericML/MOPDialect/MOPTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// BufferType (cont.)
//===----------------------------------------------------------------------===//

const int64_t BufferType::unknownSize = ShapedType::kDynamicSize;

/// Implements ShapedType::getShape
ArrayRef<int64_t> BufferType::getShape() const {
  Optional<int64_t> &size = getImpl()->size;
  if (size)
    return ArrayRef<int64_t>(&size.value(), 1);
  return ArrayRef<int64_t>(&unknownSize, 1);
}
