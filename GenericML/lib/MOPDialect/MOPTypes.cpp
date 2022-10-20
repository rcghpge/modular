//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MOPDialect/MOPTypes.h"
#include "GenericML/MOPDialect/MOPDialect.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

#include "./MLIRFieldParsers.h"

using namespace M;
using namespace MOP;

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

/// Verifies the buffer size and align constraints are valid.
LogicalResult BufferType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 Optional<int64_t> size,
                                 Optional<int64_t> align) {
  if (size && *size < 0)
    return emitError() << "buffer size cannot be negative";
  if (align && *align < 0)
    return emitError() << "buffer align cannot be negative";
  if (align && !llvm::isPowerOf2_64(*align))
    return emitError() << "buffer align must be a power of two";
  return success();
}

bool BufferType::agreeOnSize(BufferType lhs, Type rhs) {
  if (auto rhsBuffer = mlir::dyn_cast<BufferType>(rhs))
    return lhs.getSize() == rhsBuffer.getSize();
  return false;
}

/// Implements ShapedType::cloneWith
ShapedType BufferType::cloneWith(Optional<ArrayRef<int64_t>> shape,
                                 Type dtype) const {
  assert(!shape && "cannot change shape in BufferType::cloneWith");
  assert(dtype.isUnsignedInteger(8) &&
         "invalid dtype for BufferType::cloneWith");
  return BufferType::get(getContext(), getSize(), getAlign());
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
// DType
//===----------------------------------------------------------------------===//

/// Verifies the given MLIR type can be represented as a 'raw' DType.
LogicalResult
DTypeType::verify(function_ref<::mlir::InFlightDiagnostic()> emitError,
                  Type mlirType) {
  DType rawDType = mlirTypeToRawDType(mlirType);
  if (rawDType.isInvalid())
    return emitError() << "unable to represent MLIR type " << mlirType
                       << " as a DType.";
  return success();
}

DType DTypeType::mlirTypeToRawDType(Type type) {
  if (auto intType = mlir::dyn_cast<IntegerType>(type))
    return getEquivalentDType(intType);
  if (auto fpType = mlir::dyn_cast<FloatType>(type))
    return getEquivalentDType(fpType);
  return {}; // invalid
}

Type DTypeType::rawDTypeToMLIRType(MLIRContext *ctxt, DType rawDType) {
  if (rawDType.isInvalid())
    return {}; // null
  if (rawDType.isInt())
    return getEquivalentIntegerType(ctxt, rawDType);
  if (rawDType.isFloat())
    return getEquivalentFloatType(ctxt, rawDType);
  return {}; // null
}

DType DTypeType::getRawDType() const {
  return mlirTypeToRawDType(getMlirType());
}

//===----------------------------------------------------------------------===//
// TensorShapeType
//===----------------------------------------------------------------------===//

/// Parses a list of dimensions, interpreting '?' as -1.
static ParseResult parseShapeDims(AsmParser &p,
                                  FailureOr<SmallVector<int64_t>> &dims) {
  dims.emplace();
  return p.parseDimensionList(*dims, /*allowDynamic=*/true,
                              /*withTrailingX=*/false);
}

/// Prints a list of dimensions, interpreting -1 as '?'.
static void printShapeDims(AsmPrinter &p, ArrayRef<int64_t> dims) {
  llvm::interleave(
      dims, p.getStream(),
      [&](int64_t dim) {
        p << (dim == ShapedType::kDynamicSize ? "?" : std::to_string(dim));
      },
      "x");
}

/// Verifies given dimensions represent a valid tensor shape.
LogicalResult
TensorShapeType::verify(function_ref<InFlightDiagnostic()> emitError,
                        ArrayRef<int64_t> dims) {
  if (dims.empty() || dims.size() > getMaximumRank())
    return emitError() << "shape must have rank between 1 and "
                       << getMaximumRank();
  for (auto dim : dims) {
    if (dim <= 0 && dim != ShapedType::kDynamicSize)
      return emitError() << "shape dimensions must be strictly positive or the "
                            "distinguished '?' unknown dimension";
  }
  return success();
}

int64_t TensorShapeType::getNumElements() const {
  int64_t prod = 1;
  for (auto dim : getDims()) {
    if (dim == ShapedType::kDynamicSize)
      return ShapedType::kDynamicSize;
    prod *= dim;
  }
  return prod;
}

//===----------------------------------------------------------------------===//
// TensorType
//===----------------------------------------------------------------------===//

/// Parses a compact tensor type description.
static ParseResult
parseCompactShapeAndDType(AsmParser &p, FailureOr<TensorShapeType> &tensorShape,
                          FailureOr<DTypeType> &dtype) {
  mlir::SMLoc dimsLoc;
  SmallVector<int64_t> dims;
  mlir::SMLoc typeLoc;
  Type mlirType;
  if (p.getCurrentLocation(&dimsLoc) ||
      p.parseDimensionList(dims, /*allowDynamic=*/true,
                           /*withTrailingX=*/true) ||
      p.getCurrentLocation(&typeLoc) || p.parseType(mlirType))
    return failure();
  // Though the following get calls will invoke verify, the resulting errors
  // will be locationless and trigger a core dump. Try to be friendlier.
  if (failed(TensorShapeType::verify([&]() { return p.emitError(dimsLoc); },
                                     dims)) ||
      failed(
          DTypeType::verify([&]() { return p.emitError(typeLoc); }, mlirType)))
    return failure();
  tensorShape.emplace(TensorShapeType::get(p.getContext(), dims));
  dtype.emplace(DTypeType::get(p.getContext(), mlirType));
  return success();
}

/// Prints a compact tensor type description.
static void printCompactShapeAndDType(AsmPrinter &p,
                                      TensorShapeType tensorShape,
                                      DTypeType dtype) {
  llvm::interleave(
      tensorShape.getDims(), p.getStream(),
      [&](int64_t dim) {
        p << (dim == ShapedType::kDynamicSize ? "?" : std::to_string(dim));
      },
      "x");
  p << "x";
  p << dtype.getMlirType();
}

int64_t TensorType::getSizeInBytes() const {
  return DTypeType::mlirTypeToRawDType(getDType().getMlirType())
      .getSizeInBytes(getTensorShape().getNumElements());
}

BufferType TensorType::getExpectedBufferType() const {
  int64_t size = getSizeInBytes();
  return BufferType::get(getContext(),
                         size == ShapedType::kDynamicSize ? Optional<int64_t>()
                                                          : size,
                         /*align=*/{});
}

/// Implements SubElementTypeInterface::walkImmediateSubElements.
void TensorType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkTypesFn(getTensorShape());
  walkTypesFn(getDType());
}

/// Implements SubElementTypeInterface::replaceImmediateSubElements.
Type TensorType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                             ArrayRef<Type> replTypes) const {
  assert(replAttrs.empty());
  assert(replTypes.size() == 2);
  TensorShapeType shape = mlir::dyn_cast<TensorShapeType>(replTypes[0]);
  assert(shape);
  DTypeType dtype = mlir::dyn_cast<DTypeType>(replTypes[1]);
  assert(dtype);
  return TensorType::get(getContext(), shape, dtype);
}

/// Implements ShapedTypeInterface::cloneWith.
ShapedType TensorType::cloneWith(Optional<ArrayRef<int64_t>> shape,
                                 Type elementType) const {
  TensorShapeType newShape = getTensorShape();
  if (shape)
    newShape = TensorShapeType::get(getContext(), *shape);
  DTypeType newDType = getDType();
  if (elementType)
    newDType = DTypeType::get(getContext(), elementType);
  return TensorType::get(getContext(), newShape, newDType);
}

/// Implements ShapedTypeInterface::getElementType.
Type TensorType::getElementType() const { return getDType().getMlirType(); }

/// Implements ShapedTypeInterface::hasRank.
bool TensorType::hasRank() const { return true; }

/// Implements ShapedTypeInterface::getShape.
ArrayRef<int64_t> TensorType::getShape() const {
  return getTensorShape().getDims();
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
