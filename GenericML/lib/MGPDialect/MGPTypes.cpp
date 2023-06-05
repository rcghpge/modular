//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPTypes.h"
#include "GenericML/MGPDialect/MGPDialect.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ML/DType.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorShape.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MGP;

//===----------------------------------------------------------------------===//
// CpuBufferType
//===----------------------------------------------------------------------===//

/// Verifies the buffer size and align constraints are valid.
LogicalResult
CpuBufferType::verify(function_ref<InFlightDiagnostic()> emitError,
                      std::optional<uint64_t> optSize) {
  if (!isValidOptSize(optSize))
    return emitError() << "buffer size must be strictly positive.";
  return success();
}

CpuBufferType CpuBufferType::get(MLIRContext *context, uint64_t rawSize) {
  return CpuBufferType::get(context, rawSizeToOptSize(rawSize));
}

CpuBufferType
CpuBufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                          MLIRContext *context, uint64_t rawSize) {
  return CpuBufferType::get(context, rawSize);
}

CpuBufferType CpuBufferType::get(MLIRContext *context,
                                 IntegerAttr rawSizeOrNull) {
  std::optional<uint64_t> optSize;
  if (rawSizeOrNull)
    optSize = rawSizeToOptSize(rawSizeOrNull.getUInt());
  return CpuBufferType::get(context, optSize);
}

CpuBufferType
CpuBufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                          MLIRContext *context, IntegerAttr rawSizeOrNull) {
  return CpuBufferType::get(context, rawSizeOrNull);
}

IntegerAttr CpuBufferType::getRawSize() const {
  Type ui64 = IntegerType::get(getContext(), 64, IntegerType::Unsigned);
  APInt value(ui64.getIntOrFloatBitWidth(), optSizeToRawSize(getOptSize()),
              ui64.isSignedInteger());
  return IntegerAttr::get(ui64, value);
}

/// Implements ShapedType::cloneWith
ShapedType CpuBufferType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                    Type elementType) const {
  assert(!shape && "cannot change shape in CpuBufferType::cloneWith");
  assert(elementType.isUnsignedInteger(8) &&
         "invalid dtype for CpuBufferType::cloneWith");
  return CpuBufferType::get(getContext(), getOptSize());
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
static ParseResult parseDType(AsmParser &p, std::optional<DType> &optDType) {
  StringRef keyword;
  llvm::SMLoc dtypeLoc;
  if (p.getCurrentLocation(&dtypeLoc) || p.parseKeyword(&keyword))
    return failure();

  if (keyword == kUnknownDType)
    return success();

  FailureOr<DType> failOrDType = DType::getFromString(keyword);
  if (failed(failOrDType))
    return p.emitError(dtypeLoc) << "unrecognized dtype '" << keyword << "'.";
  if (failOrDType->isInvalid()) {
    // Invalid is not legal.
    return p.emitError(dtypeLoc) << "invalid dtype.";
  }

  optDType.emplace(*failOrDType);
  return success();
}

/// Prints a DType, where the none value is printed as 'unk'.
static void printDType(AsmPrinter &p, std::optional<DType> optDType) {
  p << (optDType ? optDType->getAsString() : kUnknownDType);
}

ErrorOr<std::optional<DType>> DTypeType::fromMlirType(Type mlirType) {
  if (auto unkType = mlir::dyn_cast<NoneType>(mlirType))
    return std::nullopt; // None denotes unknown.
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
                                    std::optional<DType> optDtype) {
  if (!optDtype)
    return NoneType::get(context); // None denotes unknown.
  if (optDtype->isInt() || optDtype->isBool()) {
    Type type = getEquivalentIntegerType(context, *optDtype);
    if (type != nullptr)
      return type;
  }
  if (optDtype->isFloat()) {
    Type type = getEquivalentFloatType(context, *optDtype);
    if (type != nullptr)
      return type;
  }
  return Error(Twine("unable to represent DType '") + optDtype->getAsString() +
               "' as an MLIR type.");
}

ErrorOrSuccess DTypeType::hasMLIRType(std::optional<DType> optDType) {
  if (!optDType)
    return {}; // None denotes unknown.
  if (optDType->isInt() && hasEquivalentIntegerType(*optDType))
    return {};
  if (optDType->isBool())
    return {};
  if (optDType->isFloat() && hasEquivalentFloatType(*optDType))
    return {};
  return Error(Twine("unable to represent DType '") + optDType->getAsString() +
               "' as an MLIR type.");
}

uint8_t DTypeType::asRawDType(std::optional<DType> optDType) {
  return optDType ? optDType->getValue() : DType().getValue();
}

std::optional<DType> DTypeType::fromRawDType(uint8_t rawDType) {
  DType dtype(rawDType);
  return dtype.isInvalid() ? std::optional<DType>() : dtype;
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecType
//===----------------------------------------------------------------------===//

/// Parses a list of dimensions followed by a DType, interpreting '?' as the
/// empty dimension and 'unk' as the empty DType. The shope is constructed
/// to match dims, but using the raw signed form.
static ParseResult parseDimsAndDType(AsmParser &p,
                                     SmallVector<std::optional<uint64_t>> &dims,
                                     std::optional<DType> &optDType,
                                     SmallVector<int64_t> &shape) {
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) ||
      p.parseDimensionList(shape, /*allowDynamic=*/true,
                           /*withTrailingX=*/true) ||
      parseDType(p, optDType))
    return failure();

  dims.reserve(shape.size());
  for (int64_t dim : shape) {
    if (!isValidRawSignedSize(dim))
      return p.emitError(loc) << "shape dimensions must be non-negative or the "
                                 "distinguished '?' unknown dimension.";
    dims.emplace_back(rawSignedSizeToOptSize(dim));
  }

  return success();
}

/// Prints a list of dimensions followed by a DType, printing an empty
/// dimension as the '?' and an empty DType as 'unk'. The shape is ignored.
static void printDimsAndDType(AsmPrinter &p,
                              ArrayRef<std::optional<uint64_t>> dims,
                              std::optional<DType> optDType,
                              ArrayRef<int64_t> shape) {
  llvm::interleave(
      dims, p.getStream(),
      [&](const std::optional<uint64_t> &dim) {
        p << (dim ? std::to_string(*dim) : "?");
      },
      "x");
  if (!dims.empty())
    p << "x";
  printDType(p, optDType);
}

/// Verifies that the given dimensions represent a valid tensor shape, and
/// that the element type in DType form is also representable as an MLIR type.
LogicalResult
CpuTensorSpecType::verify(function_ref<InFlightDiagnostic()> emitError,
                          ArrayRef<std::optional<uint64_t>> dims,
                          std::optional<DType> optDType,
                          ArrayRef<int64_t> shape) {
  if (shape.size() > kMaxRank)
    return emitError() << "shape must have rank between 0 and " << kMaxRank
                       << ".";
  for (const std::optional<uint64_t> &dim : dims) {
    if (!isValidOptSize(dim))
      return emitError() << "shape dimensions must be non-negative or the "
                            "distinguished '?' unknown dimension.";
  }

  // Just in case the type was constructed using the default method
  // with dims and shape set independently, check the two representations
  // coincide.
  if (dims.size() != shape.size())
    return emitError() << "dims and shape must correspond";
  for (size_t i = 0; i < dims.size(); ++i) {
    int64_t rawSignedDim = optSizeToRawSignedSize(dims[i]);
    if (shape[i] != rawSignedDim)
      return emitError() << "dims and shape must correspond";
  }

  // Caution: can't inline into if since ErrorOrSuccess is temporary.
  ErrorOrSuccess errOrSuc = DTypeType::hasMLIRType(optDType);
  if (const char *err = errOrSuc.getError())
    return emitError() << err;

  return success();
}

CpuTensorSpecType CpuTensorSpecType::get(MLIRContext *context,
                                         ArrayRef<std::optional<uint64_t>> dims,
                                         std::optional<DType> optDType) {
  SmallVector<int64_t> shape;
  shape.reserve(dims.size());
  for (const std::optional<uint64_t> &dim : dims)
    shape.emplace_back(optSizeToRawSignedSize(dim));
  return CpuTensorSpecType::get(context, dims, optDType, shape);
}

CpuTensorSpecType CpuTensorSpecType::getChecked(
    function_ref<::mlir::InFlightDiagnostic()> emitError, MLIRContext *context,
    ArrayRef<std::optional<uint64_t>> dims, std::optional<DType> optDType) {
  return CpuTensorSpecType::get(context, dims, optDType);
}

CpuTensorSpecType CpuTensorSpecType::get(ArrayRef<int64_t> shape,
                                         Type elementType) {
  ErrorOr<std::optional<DType>> errOrDType =
      DTypeType::fromMlirType(elementType);
  assert(errOrDType.getError() == nullptr &&
         "element type is not representable as a DType");

  SmallVector<std::optional<uint64_t>> dims;
  dims.reserve(shape.size());
  for (auto dim : shape)
    dims.emplace_back(rawSignedSizeToOptSize(dim));

  return CpuTensorSpecType::get(elementType.getContext(), dims, *errOrDType,
                                shape);
}

CpuTensorSpecType CpuTensorSpecType::getChecked(
    function_ref<::mlir::InFlightDiagnostic()> emitError,
    ArrayRef<int64_t> shape, Type elementType) {
  ErrorOr<std::optional<DType>> errOrDType =
      DTypeType::fromMlirType(elementType);
  if (const auto *err = errOrDType.getError()) {
    emitError() << err;
    return {};
  }
  return CpuTensorSpecType::get(shape, elementType);
}

CpuTensorSpecType CpuTensorSpecType::get(MLIRContext *context,
                                         ArrayRef<uint64_t> rawDims,
                                         uint8_t rawDType) {
  std::optional<DType> dtype = DTypeType::fromRawDType(rawDType);
  SmallVector<std::optional<uint64_t>> dims;
  dims.reserve(rawDims.size());
  for (auto dim : rawDims)
    dims.emplace_back(rawSizeToOptSize(dim));

  return CpuTensorSpecType::get(context, dims, dtype);
}

CpuTensorSpecType CpuTensorSpecType::getChecked(
    function_ref<::mlir::InFlightDiagnostic()> emitError, MLIRContext *context,
    ArrayRef<uint64_t> rawShape, uint8_t rawDType) {
  return CpuTensorSpecType::get(context, rawShape, rawDType);
}

CpuTensorSpecType CpuTensorSpecType::get(IntArrayElementsAttr rawShape,
                                         IntegerAttr rawDTypeOrNull) {
  // The null rawDType is a convenience for unknown,
  // and the invalid DType encodes unknown.
  uint8_t rawDTypeValue =
      rawDTypeOrNull ? rawDTypeOrNull.getUInt() : DType().getValue();
  return CpuTensorSpecType::get(rawShape.getContext(),
                                rawShape.asArrayRef<uint64_t>(), rawDTypeValue);
}

CpuTensorSpecType CpuTensorSpecType::getChecked(
    function_ref<::mlir::InFlightDiagnostic()> emitError,
    IntArrayElementsAttr rawShape, IntegerAttr rawDTypeOrNull) {
  return CpuTensorSpecType::get(rawShape, rawDTypeOrNull);
}

size_t CpuTensorSpecType::getRank() const { return getDims().size(); }

size_t CpuTensorSpecType::getNumDynamicDims() const {
  return static_cast<size_t>(llvm::count(getDims(), std::nullopt));
}

std::optional<uint64_t> CpuTensorSpecType::operator[](size_t i) const {
  assert(i < getDims().size() && "shape index out of range");
  return getDims()[i];
}

std::optional<uint64_t> CpuTensorSpecType::getNumElements() const {
  uint64_t prod = 1;
  for (const std::optional<uint64_t> &dim : getDims()) {
    if (!dim)
      return {}; // unknown dimension
    prod *= *dim;
  }
  return prod;
}

std::optional<uint64_t> CpuTensorSpecType::getSizeInBytes() const {
  std::optional<DType> optDType = getOptDType();
  if (!optDType)
    return {}; // unknown element type
  std::optional<uint64_t> numElems = getNumElements();
  if (!numElems)
    return {}; // at least one unknown dimension
  ssize_t bytes = optDType->getSizeInBytes(plainSizeToRuntimeSize(*numElems));
  assert(bytes >= 0 && "overflow in getSizeInBytes");
  return static_cast<uint64_t>(bytes);
}

/// ShapedTypeInterface: Returns true since statically ranked.
bool CpuTensorSpecType::hasRank() const { return true; }

/// ShapedTypeInterface: Returns clone with given element type, which
/// must be representable as a dtype.
ShapedType CpuTensorSpecType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                        Type elementType) const {
  ArrayRef<int64_t> newShape = shape.value_or(getShape());
  Type newElementType = elementType ? elementType : getElementType();
  return CpuTensorSpecType::get(newShape, newElementType);
}

/// ShapedTypeInterface: Returns dtype as an MLIR type. This is always well
/// defined thanks to verify above.
Type CpuTensorSpecType::getElementType() const {
  return *DTypeType::toMLIRType(getContext(), getOptDType());
}

IntArrayElementsAttr CpuTensorSpecType::getRawDims() const {
  Type ui64 = IntegerType::get(getContext(), 64, IntegerType::Unsigned);
  SmallVector<APInt> rawDims;
  for (const std::optional<uint64_t> &dim : getDims()) {
    uint64_t rawDim = optSizeToRawSize(dim);
    rawDims.emplace_back(ui64.getIntOrFloatBitWidth(), rawDim,
                         ui64.isSignedInteger());
  }
  return IntArrayElementsAttr::get(
      ArrayType::get(static_cast<int64_t>(getRank()), ui64), rawDims);
}

IntegerAttr CpuTensorSpecType::getRawDType() const {
  Type ui8 = IntegerType::get(getContext(), 8, IntegerType::Unsigned);
  return IntegerAttr::get(ui8, DTypeType::asRawDType(getOptDType()));
}

CpuBufferType CpuTensorSpecType::getExpectedBufferType() const {
  return CpuBufferType::get(getContext(), getSizeInBytes());
}

//===----------------------------------------------------------------------===//
// CpuTensorType
//===----------------------------------------------------------------------===//

/// Parses a compact tensor type description.
static ParseResult parseTensorSpec(AsmParser &p,
                                   CpuTensorSpecType &tensorSpec) {
  SmallVector<std::optional<uint64_t>> dims;
  std::optional<DType> optDType;
  SmallVector<int64_t> shape;
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || parseDimsAndDType(p, dims, optDType, shape))
    return failure();

  // Though the following get calls will invoke verify, the resulting errors
  // will be locationless and trigger a core dump. Try to be friendlier.
  if (failed(CpuTensorSpecType::verify([&]() { return p.emitError(loc); }, dims,
                                       optDType, shape)))
    return failure();

  tensorSpec = CpuTensorSpecType::get(p.getContext(), dims, optDType, shape);
  return success();
}

/// Prints a compact tensor type description.
static void printTensorSpec(AsmPrinter &p, CpuTensorSpecType tensorSpec) {
  printDimsAndDType(p, tensorSpec.getDims(), tensorSpec.getOptDType(),
                    tensorSpec.getShape());
}

/// ShapedTypeInterface: Returns clone of tensor type.
ShapedType CpuTensorType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
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
// MGPDialect type support
//===----------------------------------------------------------------------===//

void MGPDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "GenericML/MGPDialect/MGPTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "GenericML/MGPDialect/MGPTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// CpuBufferType (cont.)
//===----------------------------------------------------------------------===//

const int64_t CpuBufferType::unknownSize = ShapedType::kDynamic;

/// Implements ShapedType::getShape
ArrayRef<int64_t> CpuBufferType::getShape() const {
  std::optional<uint64_t> &optSize = getImpl()->optSize;
  if (!optSize)
    return ArrayRef<int64_t>(&unknownSize, 1);
  // Convert to raw signed size for assertion side effect.
  (void)optSizeToRawSignedSize(optSize);
  // We'll just reinterpret the uint64_t size as an int64_t since it's in range.
  return ArrayRef<int64_t>(reinterpret_cast<const int64_t *>(&*optSize), 1);
}
