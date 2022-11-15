//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MOPDialect/MOPTypes.h"
#include "GenericML/MOP/Utils.h"
#include "GenericML/MOPDialect/MOPDialect.h"
#include "Support/Compiler/MLIRDType.h"
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

using namespace M;
using namespace MOP;

//===----------------------------------------------------------------------===//
// CpuBufferType
//===----------------------------------------------------------------------===//

/// Verifies the buffer size and align constraints are valid.
LogicalResult
CpuBufferType::verify(function_ref<InFlightDiagnostic()> emitError,
                      Optional<uint64_t> size) {
  if (!isValidSize(size))
    return emitError() << "buffer size must be positive.";
  return success();
}

CpuBufferType CpuBufferType::get(MLIRContext *context, uint64_t rawSize) {
  Optional<uint64_t> size =
      rawSize == kMissingSizeOrAlign ? Optional<uint64_t>() : rawSize;
  return CpuBufferType::get(context, size);
}

CpuBufferType
CpuBufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                          MLIRContext *context, uint64_t rawSize) {
  return CpuBufferType::get(context, rawSize);
}

CpuBufferType CpuBufferType::get(MLIRContext *context,
                                 IntegerAttr rawSizeOrNull) {
  Optional<uint64_t> size;
  if (rawSizeOrNull) {
    uint64_t value = rawSizeOrNull.getUInt();
    if (value != kMissingSizeOrAlign)
      size = value;
    // else: max value denotes unknown size.
  }
  // else: null denotes unknown size.
  return CpuBufferType::get(context, size);
}

CpuBufferType
CpuBufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                          MLIRContext *context, IntegerAttr rawSizeOrNull) {
  return CpuBufferType::get(context, rawSizeOrNull);
}

IntegerAttr CpuBufferType::getRawSize() const {
  Type ui64 = IntegerType::get(getContext(), 64, IntegerType::Unsigned);
  uint64_t sizeOrMax = getSize() ? *getSize() : kMissingSizeOrAlign;
  APInt value(64, sizeOrMax, /*isSigned=*/false);
  return IntegerAttr::get(ui64, value);
}

/// Implements ShapedType::cloneWith
ShapedType CpuBufferType::cloneWith(Optional<ArrayRef<int64_t>> shape,
                                    Type dtype) const {
  assert(!shape && "cannot change shape in CpuBufferType::cloneWith");
  assert(dtype.isUnsignedInteger(8) &&
         "invalid dtype for CpuBufferType::cloneWith");
  return CpuBufferType::get(getContext(), getSize());
}

/// Implements ShapedType::getElementType
Type CpuBufferType::getElementType() const {
  return IntegerType::get(getContext(), /*width=*/8, IntegerType::Unsigned);
}

/// Implements ShapedType::hasRank
bool CpuBufferType::hasRank() const { return true; }

// We defer CpuBufferType::getShape till after ODS import since depends on impl.

//===----------------------------------------------------------------------===//
// ContextType
//===----------------------------------------------------------------------===//

// Nothing to implement.

//===----------------------------------------------------------------------===//
// DTypeType
//===----------------------------------------------------------------------===//

/// The distinguished keyword for the unknown dtype.
constexpr const char *kUnknownDType = "unk";

/// Parses a DType, where 'unk' is represented as the none value, and
/// 'invalid' is considered illegal.
static ParseResult parseDType(AsmParser &p, FailureOr<Optional<DType>> &dtype) {
  StringRef keyword;
  llvm::SMLoc dtypeLoc;
  if (p.getCurrentLocation(&dtypeLoc) || p.parseKeyword(&keyword))
    return failure();

  if (keyword == kUnknownDType) {
    dtype.emplace(); // None denotes unknown.
    return success();
  }

  FailureOr<DType> failOrDType = DType::getFromString(keyword);
  if (failed(failOrDType))
    return p.emitError(dtypeLoc) << "unrecognized dtype '" << keyword << "'.";
  if (failOrDType->isInvalid())
    // Invalid is not legal.
    return p.emitError(dtypeLoc) << "invalid dtype.";

  dtype.emplace(*failOrDType);
  return success();
}

/// Prints a DType, where the none value is printed as 'unk'.
static void printDType(AsmPrinter &p, Optional<DType> dtype) {
  p << (dtype ? dtype->getAsString() : kUnknownDType);
}

ErrorOr<Optional<DType>> DTypeType::fromMlirType(Type mlirType) {
  if (auto unkType = mlir::dyn_cast<NoneType>(mlirType))
    return llvm::None; // None denotes unknown.
  if (auto intType = mlir::dyn_cast<IntegerType>(mlirType)) {
    FailureOr<DType> dtype = getEquivalentDType(intType);
    if (succeeded(dtype))
      return *dtype;
  }
  if (auto floatType = mlir::dyn_cast<FloatType>(mlirType)) {
    FailureOr<DType> dtype = getEquivalentDType(floatType);
    if (succeeded(dtype))
      return *dtype;
  }
  // TODO(#4717): Surely there's a better way?
  std::string str;
  llvm::raw_string_ostream os(str);
  os << "unable to represent MLIR type ";
  os << mlirType;
  os << " as a DType.";
  return Error(str);
}

ErrorOr<Type> DTypeType::toMLIRType(MLIRContext *context,
                                    Optional<DType> dtype) {
  if (!dtype)
    return NoneType::get(context); // None denotes unknown.
  if (dtype->isInt()) {
    Type type = getEquivalentIntegerType(context, *dtype);
    if (type != nullptr)
      return type;
  }
  if (dtype->isFloat()) {
    Type type = getEquivalentFloatType(context, *dtype);
    if (type != nullptr)
      return type;
  }
  return Error(Twine("unable to represent DType '") + dtype->getAsString() +
               "' as an MLIR type.");
}

std::string DTypeType::hasNoMLIRType(Optional<DType> dtype) {
  if (!dtype)
    return {}; // None denotes unknown.
  if (dtype->isInt() && hasEquivalentIntegerType(*dtype))
    return {};
  if (dtype->isFloat() && hasEquivalentFloatType(*dtype))
    return {};
  return (Twine("unable to represent DType '") + dtype->getAsString() +
          "' as an MLIR type.")
      .str();
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecType
//===----------------------------------------------------------------------===//

/// Parses a list of dimensions followed by a DType, interpreting '?' as the
/// -1 dimension and 'unk' as the missing DType.
static ParseResult parseShapeAndDType(AsmParser &p,
                                      FailureOr<SmallVector<int64_t>> &shape,
                                      FailureOr<Optional<DType>> &dtype) {
  shape.emplace();
  if (p.parseDimensionList(*shape, /*allowDynamic=*/true,
                           /*withTrailingX=*/true) ||
      parseDType(p, dtype))
    return failure();

  return success();
}

/// Prints a list of dimensions followed by a DType, interpreting -1 as the
/// '?' dimension and the missing DType as 'unk'.
static void printShapeAndDType(AsmPrinter &p, ArrayRef<int64_t> shape,
                               Optional<DType> dtype) {
  llvm::interleave(
      shape, p.getStream(),
      [&](int64_t dim) {
        p << (dim == ShapedType::kDynamicSize ? "?" : std::to_string(dim));
      },
      "x");
  p << "x";
  printDType(p, dtype);
}

/// Verifies that the given dimensions represent a valid tensor shape, and
/// that the element type in DType form is also representable as an MLIR type.
LogicalResult
CpuTensorSpecType::verify(function_ref<InFlightDiagnostic()> emitError,
                          ArrayRef<int64_t> shape, Optional<DType> dtype) {
  if (shape.empty() || shape.size() > getMaximumRank())
    return emitError() << "shape must have rank between 1 and "
                       << getMaximumRank() << ".";
  for (auto dim : shape) {
    if (dim <= 0 && dim != ShapedType::kDynamicSize)
      return emitError() << "shape dimensions must be strictly positive or the "
                            "distinguished '?' unknown dimension.";
  }

  if (auto err = DTypeType::hasNoMLIRType(dtype); !err.empty())
    return emitError() << err;

  return success();
}

CpuTensorSpecType CpuTensorSpecType::get(ArrayRef<int64_t> shape,
                                         Type elementType) {
  ErrorOr<Optional<DType>> errOrDType = DTypeType::fromMlirType(elementType);
  assert(errOrDType.getError() == nullptr &&
         "element type is not representable as a DType");
  return CpuTensorSpecType::get(elementType.getContext(), shape, *errOrDType);
}

CpuTensorSpecType CpuTensorSpecType::getChecked(
    function_ref<::mlir::InFlightDiagnostic()> emitError,
    ArrayRef<int64_t> shape, Type elementType) {
  ErrorOr<Optional<DType>> errOrDType = DTypeType::fromMlirType(elementType);
  if (const auto *err = errOrDType.getError()) {
    emitError() << err;
    return {};
  }
  return CpuTensorSpecType::get(elementType.getContext(), shape, *errOrDType);
}

CpuTensorSpecType CpuTensorSpecType::get(MLIRContext *context,
                                         ArrayRef<int64_t> shape,
                                         uint8_t rawDType) {
  // The invalid DType encodes unknown.
  DType dtypeOrInvalid(rawDType);
  Optional<DType> dtype =
      dtypeOrInvalid.isValid() ? Optional<DType>(dtypeOrInvalid) : llvm::None;
  return CpuTensorSpecType::get(context, shape, dtype);
}

CpuTensorSpecType CpuTensorSpecType::getChecked(
    function_ref<::mlir::InFlightDiagnostic()> emitError, MLIRContext *context,
    ArrayRef<int64_t> shape, uint8_t rawDType) {
  return CpuTensorSpecType::get(context, shape, rawDType);
}

CpuTensorSpecType CpuTensorSpecType::get(IntArrayElementsAttr rawShape,
                                         IntegerAttr rawDTypeOrNull) {
  // The null rawDType is a convenience for unknown,
  // and the invalid DType encodes unknown.
  uint8_t rawDTypeValue =
      rawDTypeOrNull ? rawDTypeOrNull.getUInt() : DType().getValue();
  return CpuTensorSpecType::get(rawShape.getContext(),
                                rawShape.asArrayRef<int64_t>(), rawDTypeValue);
}

CpuTensorSpecType CpuTensorSpecType::getChecked(
    function_ref<::mlir::InFlightDiagnostic()> emitError,
    IntArrayElementsAttr rawShape, IntegerAttr rawDTypeOrNull) {
  return CpuTensorSpecType::get(rawShape, rawDTypeOrNull);
}

size_t CpuTensorSpecType::getRank() const { return getShape().size(); }

size_t CpuTensorSpecType::getNumDynamicDims() const {
  return static_cast<size_t>(llvm::count(getShape(), ShapedType::kDynamicSize));
}

Optional<uint64_t> CpuTensorSpecType::operator[](size_t i) const {
  assert(i < getShape().size() && "shape index out of range");
  int64_t dim = getShape()[i];
  return dim == ShapedType::kDynamicSize ? Optional<uint64_t>()
                                         : static_cast<uint64_t>(dim);
}

Optional<uint64_t> CpuTensorSpecType::getNumElements() const {
  uint64_t prod = 1;
  for (auto dim : getShape()) {
    if (dim == ShapedType::kDynamicSize)
      return {}; // unknown dimension
    prod *= static_cast<size_t>(dim);
  }
  return prod;
}

Optional<uint64_t> CpuTensorSpecType::getSizeInBytes() const {
  Optional<DType> dtype = getDtype();
  if (!dtype)
    return {}; // unknown element type
  Optional<uint64_t> numElems = getNumElements();
  if (!numElems)
    return {}; // at least one unknown dimension
  ssize_t res = dtype->getSizeInBytes(*numElems);
  if (res < 0)
    return {}; // overflow
  return static_cast<uint64_t>(res);
}

/// ShapedTypeInterface: Returns true since statically ranked.
bool CpuTensorSpecType::hasRank() const { return true; }

/// ShapedTypeInterface: Returns clone with given element type, which
/// must be representable as a dtype.
ShapedType CpuTensorSpecType::cloneWith(Optional<ArrayRef<int64_t>> shape,
                                        Type elementType) const {
  ArrayRef<int64_t> newShape = shape.value_or(getShape());
  Optional<DType> newDType = getDtype();
  if (elementType != nullptr) {
    ErrorOr<Optional<DType>> errOrDtype = DTypeType::fromMlirType(elementType);
    assert(errOrDtype.getError() == nullptr &&
           "new element type is not representable as a DType");
    newDType = *errOrDtype;
  }
  return CpuTensorSpecType::get(getContext(), newShape, newDType);
}

/// ShapedTypeInterface: Returns dtype as an MLIR type. This is always well
/// defined thanks to verify above.
Type CpuTensorSpecType::getElementType() const {
  return *DTypeType::toMLIRType(getContext(), getDtype());
}

IntArrayElementsAttr CpuTensorSpecType::getRawShape() const {
  Type si64 = IntegerType::get(getContext(), 64, IntegerType::Signed);
  SmallVector<APInt> dims;
  for (int64_t dim : getShape())
    dims.emplace_back(si64.getIntOrFloatBitWidth(), dim,
                      si64.isSignedInteger());
  return IntArrayElementsAttr::get(
      ArrayType::get(static_cast<int64_t>(getRank()), si64), dims);
}

IntegerAttr CpuTensorSpecType::getRawDType() const {
  Type ui8 = IntegerType::get(getContext(), 8, IntegerType::Unsigned);
  Optional<DType> dtype = getDtype();
  uint8_t dtypeValue;
  if (dtype)
    dtypeValue = dtype->getValue();
  else
    dtypeValue = DType().getValue(); // invalid encodes unknown
  return IntegerAttr::get(ui8, dtypeValue);
}

CpuBufferType CpuTensorSpecType::getExpectedBufferType() const {
  return CpuBufferType::get(getContext(), getSizeInBytes());
}

//===----------------------------------------------------------------------===//
// CpuTensorType
//===----------------------------------------------------------------------===//

/// Parses a compact tensor type description.
static ParseResult parseTensorSpec(AsmParser &p,
                                   FailureOr<CpuTensorSpecType> &tensorSpec) {
  FailureOr<SmallVector<int64_t>> shape;
  FailureOr<Optional<DType>> dtype;
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || parseShapeAndDType(p, shape, dtype))
    return failure();

  // Though the following get calls will invoke verify, the resulting errors
  // will be locationless and trigger a core dump. Try to be friendlier.
  if (failed(CpuTensorSpecType::verify([&]() { return p.emitError(loc); },
                                       *shape, *dtype)))
    return failure();

  tensorSpec.emplace(CpuTensorSpecType::get(p.getContext(), *shape, *dtype));
  return success();
}

/// Prints a compact tensor type description.
static void printTensorSpec(AsmPrinter &p, CpuTensorSpecType tensorSpec) {
  printShapeAndDType(p, tensorSpec.getShape(), tensorSpec.getDtype());
}

/// ShapedTypeInterface: Returns clone of tensor type.
ShapedType CpuTensorType::cloneWith(Optional<ArrayRef<int64_t>> shape,
                                    Type elementType) const {
  auto tensorSpec = mlir::dyn_cast<CpuTensorSpecType>(
      getTensorSpec().cloneWith(shape, elementType));
  return CpuTensorType::get(getContext(), tensorSpec);
}

/// ShapedTypeInterface: Returns true since all tensors have statically
/// known rank.
bool CpuTensorType::hasRank() const { return true; }

/// ShapedTypeInterface: Returns tensor spec's shape
ArrayRef<int64_t> CpuTensorType::getShape() const {
  return getTensorSpec().getShape();
}

/// ShapedTypeInterface: Returns tensor spec's element type.
Type CpuTensorType::getElementType() const {
  return getTensorSpec().getElementType();
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
// CpuBufferType (cont.)
//===----------------------------------------------------------------------===//

const int64_t CpuBufferType::unknownSize = ShapedType::kDynamicSize;

/// Implements ShapedType::getShape
ArrayRef<int64_t> CpuBufferType::getShape() const {
  Optional<uint64_t> &size = getImpl()->size;
  if (!size)
    return ArrayRef<int64_t>(&unknownSize, 1);
  assert(*size <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
         "buffer size is too large to represent as an int64_t as required by "
         "ShapedType");
  // We'll just reinterpret the uint64_t size as an int64_t.
  return ArrayRef<int64_t>(reinterpret_cast<const int64_t *>(&size.value()), 1);
}
