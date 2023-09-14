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
// Utils
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

/// Parses (((<int> | '?') 'x')* | '*' 'x') (<dtype> | 'unk')
static ParseResult
parseDimsAndDType(AsmParser &p, bool &hasRank,
                  SmallVector<std::optional<uint64_t>> &dimsOrEmpty,
                  std::optional<DType> &optDType,
                  SmallVector<int64_t> &shapeOrEmpty) {
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc))
    return failure();

  if (succeeded(p.parseOptionalStar())) {
    if (p.parseXInDimensionList())
      return failure();
    hasRank = false;
  } else if (succeeded(p.parseDimensionList(shapeOrEmpty, /*allowDynamic=*/true,
                                            /*withTrailingX=*/true))) {
    hasRank = true;
    dimsOrEmpty.reserve(shapeOrEmpty.size());
    for (int64_t dim : shapeOrEmpty) {
      if (!isValidRawSignedSize(dim))
        return p.emitError(loc)
               << "shape dimensions must be non-negative or the "
                  "distinguished '?' unknown dimension.";
      dimsOrEmpty.emplace_back(rawSignedSizeToOptSize(dim));
    }
  } else {
    return failure();
  }

  if (p.getCurrentLocation(&loc) || parseDType(p, optDType))
    return failure();

  return success();
}

/// Prints a list of dimensions followed by a DType, printing an empty
/// dimension as the '?' and an empty DType as 'unk'. The shape is ignored.
static void printDimsAndDType(AsmPrinter &p, bool hasRank,
                              ArrayRef<std::optional<uint64_t>> dimsOrEmpty,
                              std::optional<DType> optDType,
                              ArrayRef<int64_t> shapeOrEmpty) {
  if (hasRank) {
    for (const auto &dim : dimsOrEmpty)
      p.getStream() << (dim ? std::to_string(*dim) : "?") << "x";
  } else {
    p.getStream() << "*x";
  }
  printDType(p, optDType);
}

ParseResult M::MGP::parseTensorSpec(AsmParser &p,
                                    CpuTensorSpecType &tensorSpec) {
  bool hasRank;
  SmallVector<std::optional<uint64_t>> dims;
  std::optional<DType> optDType;
  SmallVector<int64_t> shape;
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) ||
      parseDimsAndDType(p, hasRank, dims, optDType, shape))
    return failure();

  // Though the following get calls will invoke verify, the resulting errors
  // will be locationless and trigger a core dump. Try to be friendlier.
  if (failed(CpuTensorSpecType::verify([&]() { return p.emitError(loc); },
                                       hasRank, dims, optDType, shape)))
    return failure();

  tensorSpec =
      CpuTensorSpecType::get(p.getContext(), hasRank, dims, optDType, shape);
  return success();
}

void M::MGP::printTensorSpec(AsmPrinter &p, CpuTensorSpecType tensorSpec) {
  printDimsAndDType(p, tensorSpec.getHasRank(), tensorSpec.getDimsOrEmpty(),
                    tensorSpec.getOptDType(), tensorSpec.getShapeOrEmpty());
}

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

CpuBufferType CpuBufferType::get(M::MGP::CpuTensorSpecType specType) {
  return CpuBufferType::get(specType.getContext(), specType.getSizeInBytes());
}

CpuBufferType
CpuBufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                          M::MGP::CpuTensorSpecType specType) {
  return get(specType);
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

/// Verifies that the given dimensions represent a valid tensor shape, and
/// that the element type in DType form is also representable as an MLIR type.
LogicalResult CpuTensorSpecType::verify(
    function_ref<InFlightDiagnostic()> emitError, bool hasRank,
    ArrayRef<std::optional<uint64_t>> dimsOrEmpty,
    std::optional<DType> optDType, ArrayRef<int64_t> shapeOrEmpty) {
  if (hasRank) {
    if (dimsOrEmpty.size() > kMaxRank)
      return emitError() << "tensor spec must have a rank between 0 and "
                         << kMaxRank << ".";

    if (dimsOrEmpty.size() != shapeOrEmpty.size())
      return emitError()
             << "tensor spec dims and shape must have the same rank.";

    for (const std::optional<uint64_t> &dim : dimsOrEmpty) {
      if (!isValidOptSize(dim))
        return emitError()
               << "tensor spec dimensions must be non-negative or the "
                  "distinguished '?' unknown dimension.";
    }

    for (size_t i = 0; i < dimsOrEmpty.size(); ++i) {
      int64_t rawSignedDim = optSizeToRawSignedSize(dimsOrEmpty[i]);
      if (shapeOrEmpty[i] != rawSignedDim)
        return emitError()
               << "tensor spec dims and shape must have the same dimensions.";
    }
  } else {
    if (!dimsOrEmpty.empty() || !shapeOrEmpty.empty())
      return emitError() << "dynamically ranked tensor spec must have empty "
                            "dims and shape.";
  }

  // Caution: can't inline into if since ErrorOrSuccess is temporary.
  ErrorOrSuccess errOrSuc = DTypeType::hasMLIRType(optDType);
  if (const char *err = errOrSuc.getError())
    return emitError() << err;

  return success();
}

CpuTensorSpecType
CpuTensorSpecType::get(MLIRContext *context,
                       std::optional<ArrayRef<std::optional<uint64_t>>> optDims,
                       std::optional<DType> optDType) {
  if (optDims) {
    SmallVector<int64_t> shape;
    shape.reserve(optDims->size());
    for (const std::optional<uint64_t> &dim : *optDims)
      shape.emplace_back(optSizeToRawSignedSize(dim));
    return CpuTensorSpecType::get(context, /*hasRank=*/true, *optDims, optDType,
                                  shape);
  } else {
    return CpuTensorSpecType::get(context, /*hasRank=*/false, {}, optDType, {});
  }
}

CpuTensorSpecType CpuTensorSpecType::getChecked(
    function_ref<::mlir::InFlightDiagnostic()> emitError, MLIRContext *context,
    std::optional<ArrayRef<std::optional<uint64_t>>> optDims,
    std::optional<DType> optDType) {
  return CpuTensorSpecType::get(context, optDims, optDType);
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

  return CpuTensorSpecType::get(elementType.getContext(),
                                /*hasRank=*/true, dims, *errOrDType, shape);
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

CpuTensorSpecType CpuTensorSpecType::get(Type elementType) {
  ErrorOr<std::optional<DType>> errOrDType =
      DTypeType::fromMlirType(elementType);
  assert(errOrDType.getError() == nullptr &&
         "element type is not representable as a DType");
  return CpuTensorSpecType::get(elementType.getContext(),
                                /*hasRank=*/false, {}, *errOrDType, {});
}

CpuTensorSpecType CpuTensorSpecType::getChecked(
    function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType) {
  ErrorOr<std::optional<DType>> errOrDType =
      DTypeType::fromMlirType(elementType);
  if (const auto *err = errOrDType.getError()) {
    emitError() << err;
    return {};
  }
  return get(elementType);
}

CpuTensorSpecType CpuTensorSpecType::get(MLIRContext *context,
                                         ArrayRef<uint64_t> rawDims,
                                         uint8_t rawDType) {
  std::optional<DType> dtype = DTypeType::fromRawDType(rawDType);
  SmallVector<std::optional<uint64_t>> dims;
  dims.reserve(rawDims.size());
  for (auto dim : rawDims)
    dims.emplace_back(rawSizeToOptSize(dim));

  return CpuTensorSpecType::get(context, std::optional(dims), dtype);
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

/// ShapedTypeInterface::hasRank.
bool CpuTensorSpecType::hasRank() const { return getHasRank(); }

/// ShapedTypeInterface::cloneWith.
/// Returns clone with given element type, which must be representable as a
/// dtype.
ShapedType CpuTensorSpecType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                        Type elementType) const {
  ArrayRef<int64_t> newShape = shape.value_or(getShape());
  Type newElementType = elementType ? elementType : getElementType();
  return CpuTensorSpecType::get(newShape, newElementType);
}

/// ShapedTypeInterface::getElementType.
/// Returns dtype as an MLIR type. This is always well defined thanks to
/// verify above.
Type CpuTensorSpecType::getElementType() const {
  return *DTypeType::toMLIRType(getContext(), getOptDType());
}

/// ShapedTypeInterface::getShape.
ArrayRef<int64_t> CpuTensorSpecType::getShape() const {
  assert(getHasRank() && "tensor spec is dynamically ranked");
  return getShapeOrEmpty();
}

std::optional<ArrayRef<std::optional<uint64_t>>>
CpuTensorSpecType::getOptDims() const {
  if (getHasRank())
    return getDimsOrEmpty();
  else
    return std::nullopt;
}

ArrayRef<std::optional<uint64_t>> CpuTensorSpecType::getDims() const {
  assert(getHasRank() && "tensor spec is dynamically ranked");
  return getDimsOrEmpty();
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
  if (!getHasRank())
    return {}; // dynamic rank
  uint64_t prod = 1;
  for (const std::optional<uint64_t> &dim : getDims()) {
    if (!dim)
      return {}; // dynamic dimension
    prod *= *dim;
  }
  return prod;
}

uint64_t CpuTensorSpecType::getPreScaledNumElements() const {
  uint64_t prod = 1;
  for (const std::optional<uint64_t> &dim : getDims()) {
    if (!dim)
      continue;
    prod *= *dim;
  }
  return prod;
}

std::optional<uint64_t> CpuTensorSpecType::getSizeInBytes() const {
  std::optional<DType> optDType = getOptDType();
  if (!optDType)
    return {}; // dynamic dtype
  std::optional<uint64_t> numElems = getNumElements();
  if (!numElems)
    return {}; // dynamic rank or dimension
  ssize_t bytes = optDType->getSizeInBytes(plainSizeToRuntimeSize(*numElems));
  assert(bytes >= 0 && "overflow in getSizeInBytes");
  return static_cast<uint64_t>(bytes);
}

std::optional<uint64_t> CpuTensorSpecType::getPreScaledSizeInBytes() const {
  std::optional<DType> optDType = getOptDType();
  if (!optDType)
    return {}; // dynamic dtype
  uint64_t numElems = getPreScaledNumElements();
  ssize_t bytes = optDType->getSizeInBytes(plainSizeToRuntimeSize(numElems));
  assert(bytes >= 0 && "overflow in getPreScaledSizeInBytes");
  return static_cast<uint64_t>(bytes);
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

//===----------------------------------------------------------------------===//
// CpuTensorType
//===----------------------------------------------------------------------===//

/// ShapedTypeInterface::cloneWith.
ShapedType CpuTensorType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                    Type elementType) const {
  auto tensorSpec = mlir::dyn_cast<CpuTensorSpecType>(
      getTensorSpec().cloneWith(shape, elementType));
  return CpuTensorType::get(getContext(), tensorSpec);
}

/// ShapedTypeInterface::hasRank.
bool CpuTensorType::hasRank() const { return getTensorSpec().hasRank(); }

/// ShapedTypeInterface::getShape.
ArrayRef<int64_t> CpuTensorType::getShape() const {
  return getTensorSpec().getShape();
}

/// ShapedTypeInterface::getElementType.
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
