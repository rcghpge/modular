//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MO dialect types.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/ErrorOr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace M;
using namespace MO;

namespace {

ParseResult parseShapeParam(AsmParser &p, TypedAttr &shapeAttr) {
  auto shapeTy = p.getBuilder().getType<ShapeType>();
  OptionalParseResult res = MOSH::parseShapeAttr(p, shapeAttr);
  if (res.has_value() && succeeded(res.value()))
    return success();

  if (succeeded(KGEN::parseParamValue(p, shapeAttr, shapeTy))) {
    if (isa<KGEN::ParamDeclRefAttr, KGEN::ParamIndexRefAttr>(shapeAttr))
      return success();
    return p.emitError(p.getCurrentLocation())
           << "TensorType shape parameter must be ShapeAttr or "
              "ParamDeclRefAttr, got: "
           << shapeAttr;
  }
  return p.emitError(p.getCurrentLocation())
         << "couldn't parse shape parameter: expected `?`, shape reference, or "
            "literal shape reference (i.e. `[...]`)";
}

class TensorLikeTypeParser {
public:
  TensorLikeTypeParser(AsmParser &parser) : parser{parser} {}

  ParseResult parse();

  template <typename TensorLikeType>
  TensorLikeType getType() {
    mlir::MLIRContext *ctx = parser.getContext();

    if (!metadata)
      metadata = mlir::DictionaryAttr::get(ctx);

    return parser.getChecked<TensorLikeType>(startLoc, ctx, shape, dtype,
                                             deviceRef, metadata);
  }

private:
  /// Parse the optional fields of the tensor-like type (e.g. deviceRef and
  /// metadata).
  ParseResult parseOptionalFields();

  /// Try to parse the current token as the metadata field.
  ParseResult parseMetadata();

  /// Parse the last part of the tensor type, the closing '>'.
  ParseResult parseComplete();

private:
  AsmParser &parser;
  mlir::SMLoc startLoc;

  // Fields of tensor-like types
  TypedAttr shape;
  DType dtype;
  DeviceRefAttr deviceRef;
  mlir::DictionaryAttr metadata;
};

ParseResult TensorLikeTypeParser::parseComplete() {
  return parser.parseGreater();
}

ParseResult TensorLikeTypeParser::parseMetadata() {
  // Metadata is the last field of the tensor type, so there are no
  // alternatives to try.
  if (parser.parseAttribute(metadata))
    return failure();

  // There should be no more commas at this point, check anyway to provide
  // a better error message.
  if (parser.parseOptionalComma())
    return parseComplete();

  parser.emitError(
      parser.getCurrentLocation(),
      "attribute field order should be device followed by metadata");
  return failure();
}

ParseResult TensorLikeTypeParser::parseOptionalFields() {
  // No optional commas, so we must be at the end.
  if (parser.parseOptionalComma()) {
    // By default, if the device is not included, assume we are on host.
    deviceRef = DeviceRefAttr::getDefault(parser.getContext());
    return parseComplete();
  }

  StringRef deviceLabel;

  // The device section is expected to start with a keyword, if that fails,
  // check if the current token is for the metadata field.
  if (parser.parseOptionalKeyword(&deviceLabel)) {
    // By default, if the device is not included, assume we are on host.
    deviceRef = DeviceRefAttr::getDefault(parser.getContext());
    return parseMetadata();
  }

  DeviceId deviceId;

  // Parse the colon separator and integer device ID
  if (parser.parseColon() || parser.parseInteger(deviceId))
    return failure();

  // Successfully parsed a device ref
  deviceRef = DeviceRefAttr::get(parser.getContext(), deviceLabel, deviceId);

  if (parser.parseOptionalComma())
    return parseComplete();

  return parseMetadata();
}

ParseResult TensorLikeTypeParser::parse() {
  startLoc = parser.getCurrentLocation();

  // parse '<'
  if (parser.parseLess())
    return failure();

  // parse shape
  if (parseShapeParam(parser, shape))
    return failure();

  // parse dtype
  if (parser.parseComma())
    return failure();

  FailureOr<DType> dtypeOr = mlir::FieldParser<DType>::parse(parser);
  if (failed(dtypeOr)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "invalid dtype in MO::TensorType");
  }

  dtype = dtypeOr.value();

  return parseOptionalFields();
}

void printTensorLikeType(AsmPrinter &printer, TypedAttr shape, DType dtype,
                         DeviceRefAttr device, mlir::DictionaryAttr metadata) {
  printer << '<';
  MOSH::printShapeTypedAttr(printer, shape);
  printer << ", " << dtype.getAsString();

  // Only print the device if it is not the default/host device.
  if (device && device != DeviceRefAttr::getDefault(device.getContext()))
    printer << ", " << device.getLabel() << ":" << device.getId();
  else if (!device)
    printer << ", INVALID_STATE_NO_DEVICE";

  if (!metadata.empty())
    printer << ", " << metadata;

  printer << '>';
}

} // end namespace

FailureOr<DType> mlir::FieldParser<DType>::parse(AsmParser &parser) {
  StringRef value;
  if (parser.parseKeyword(&value))
    return failure();
  return DType::getFromString(value);
}

//===----------------------------------------------------------------------===//
// ODS Boilerplate
//===----------------------------------------------------------------------===//

static std::string asString(Type type) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << type;
  return str;
}

//===----------------------------------------------------------------------===//
// TensorType
//===----------------------------------------------------------------------===//

/// A layout attribute will always be stored in the metadata `DictionaryAttr` of
/// `MO::TensorType` with this name as the key.
static constexpr StringRef kLayoutMetadataName = "layout";

/// A iospec attribute will always be stored in the metadata `DictionaryAttr` of
/// `MO::TensorType` with this name as the key.
static constexpr StringRef kIOKindMetadataName = "io";

/// A mutability attribute will always be stored in the metadata
/// `DictionaryAttr` of `MO::TensorType` with this name as the key.
static constexpr StringRef kIsMutableMetadataName = "mut";

Type TensorType::parse(AsmParser &parser) {
  TensorLikeTypeParser tensorParser(parser);

  if (tensorParser.parse())
    return {};

  return tensorParser.getType<TensorType>();
}

void TensorType::print(AsmPrinter &printer) const {
  printTensorLikeType(printer, getShapeAttr(), getDtype(), getDeviceRef(),
                      getMetadata());
}

/// Verifies that the given dimensions represent a valid tensor shape, and
/// that the element type in DType form is also representable as an MLIR type.
LogicalResult TensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 TypedAttr shapeAttr, DType dtype,
                                 DeviceRefAttr deviceRef,
                                 DictionaryAttr metadata) {
  if (!deviceRef)
    return emitError() << "no device attribute was defined";

  auto shape = llvm::dyn_cast<ShapeAttr>(shapeAttr);
  if (!shape)
    return emitError() << "invalid shape attribute, only MOSH::ShapeAttr is "
                          "allowed for tensor types, got: "
                       << shapeAttr;

  if (!llvm::all_of(shape.getValues(), ShapeAttr::isDim))
    return emitError() << "tensor dimensions must be a getDimType()";

  for (TypedAttr dimAttr : shape.getValues()) {
    if (auto dimInt = ShapeAttr::getStaticDim(dimAttr)) {
      if (*dimInt < 0)
        return emitError()
               << "invalid dimension, expect all dimensions >= 0 got: "
               << *dimInt;
    }
  }

  // The device specification used to reside in the metadata dictionary,
  // sanity check that the tensor type is not using the old format.
  if (metadata && metadata.contains("device"))
    return emitError()
           << "device attribute no longer resides in the metadata dictionary";

  return success();
}

// TODO : These are shamelessly copied out of MGP. Need to align
ErrorOr<DType> M::MO::fromMLIRType(Type mlirType) {
  if (auto unkType = llvm::dyn_cast<NoneType>(mlirType))
    return DType(); // invalid denotes unknown.
  if (auto intType = llvm::dyn_cast<IntegerType>(mlirType)) {
    FailureOr<DType> dtype = getEquivalentDType(intType);
    if (succeeded(dtype) && !(dtype->isInvalid()))
      return *dtype;
  }
  if (auto floatType = llvm::dyn_cast<FloatType>(mlirType)) {
    FailureOr<DType> dtype = getEquivalentDType(floatType);
    if (succeeded(dtype) && !(dtype->isInvalid()))
      return *dtype;
  }
  return Error(Twine("unable to represent MLIR Type '") + asString(mlirType) +
               "' as an M::DType type.");
}

ErrorOr<Type> M::MO::toMLIRType(MLIRContext *context, DType dType) {
  if (!dType.isValid())
    return NoneType::get(context); // None denotes unknown.
  if (dType.isInt() || dType.isBool())
    if (Type type = getEquivalentIntegerType(context, dType))
      return type;
  if (dType.isFloat())
    if (Type type = getEquivalentFloatType(context, dType))
      return type;
  return Error(Twine("unable to represent DType '") + dType.getAsString() +
               "' as an MLIR type.");
}

DType M::MO::getDTypeOrInvalid(Type ty) {
  if (auto tensorTy = dyn_cast<TensorType>(ty))
    return tensorTy.getDtype();
  else if (auto bufferTy = dyn_cast<BufferType>(ty))
    return bufferTy.getDtype();
  ErrorOr<DType> errOr = fromMLIRType(ty);
  return errOr.isError() ? DType::invalid : *errOr;
}

TypedAttr M::MO::getShape(Type ty) {
  if (auto tensorTy = dyn_cast<TensorType>(ty))
    return tensorTy.getShapeAttr();
  else if (auto bufferTy = dyn_cast<BufferType>(ty))
    return bufferTy.getShapeAttr();

  llvm_unreachable(
      "M::MO::getShape can only take MO::BufferType or MO::TensorType");
}

// Base builder which all others call
TensorType TensorType::get(::mlir::MLIRContext *context, TypedAttr shapeAttr,
                           DType dtype, DeviceRefAttr deviceRef,
                           DictionaryAttr metadata) {
  // Ensure device ref is always populated
  if (!deviceRef)
    deviceRef = DeviceRefAttr::getDefault(context);

  return Base::get(context, std::move(shapeAttr), std::move(dtype),
                   std::move(deviceRef), std::move(metadata));
}

// Checked base builder
TensorType TensorType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  ::mlir::MLIRContext *context,
                                  TypedAttr shapeAttr, DType dtype,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  if (failed(verify(emitError, shapeAttr, dtype, deviceRef, metadata)))
    return {};
  return TensorType::get(context, shapeAttr, dtype, deviceRef, metadata);
}

TensorType TensorType::get(TypedAttr shapeAttr, DType dtype,
                           DeviceRefAttr deviceRef, DictionaryAttr metadata) {
  MLIRContext *ctx = shapeAttr.getContext();
  if (!metadata)
    metadata = DictionaryAttr::get(ctx);
  return get(ctx, shapeAttr, dtype, deviceRef, metadata);
}

TensorType TensorType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  TypedAttr shapeAttr, DType dtype,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  if (failed(verify(emitError, shapeAttr, dtype, deviceRef, metadata)))
    return {};
  return TensorType::get(shapeAttr, dtype, deviceRef, metadata);
}

TensorType TensorType::get(TypedAttr shapeAttr, Type elementType,
                           DeviceRefAttr deviceRef, DictionaryAttr metadata) {
  ErrorOr<DType> errOrDType = fromMLIRType(elementType);
  ASSERT_STREAM(!errOrDType.isError(),
                << "element type is not representable as a DType: "
                << errOrDType.takeError());
  return get(shapeAttr, *errOrDType, deviceRef, metadata);
}

TensorType TensorType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  TypedAttr shapeAttr, Type elementType,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  ErrorOr<DType> dtype = fromMLIRType(elementType);
  if (!dtype) {
    emitError() << dtype.takeError();
    return {};
  }
  if (failed(verify(emitError, shapeAttr, *dtype, deviceRef, metadata)))
    return {};
  return TensorType::get(shapeAttr, *dtype, deviceRef, metadata);
}

TensorType TensorType::get(ArrayRef<int64_t> shape, Type elementType,
                           DeviceRefAttr deviceRef, DictionaryAttr metadata) {
  auto ctx = elementType.getContext();
  auto newShape = ShapeAttr::get(shape, ShapeType::get(ctx));
  return get(newShape, elementType, deviceRef, metadata);
}

TensorType TensorType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<int64_t> shape, Type elementType,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  auto ctx = elementType.getContext();
  auto shapeAttr = ShapeAttr::getChecked(emitError, shape, ShapeType::get(ctx));
  if (!shapeAttr)
    return {};
  return TensorType::getChecked(emitError, shapeAttr, elementType, deviceRef,
                                metadata);
}

TensorType TensorType::get(ArrayRef<TypedAttr> shape, Type elementType,
                           DeviceRefAttr deviceRef, DictionaryAttr metadata) {
  auto ctx = elementType.getContext();
  auto newShape = ShapeAttr::get(shape, ShapeType::get(ctx));
  return get(newShape, elementType, deviceRef, metadata);
}

TensorType TensorType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<TypedAttr> shape, Type elementType,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  auto ctx = elementType.getContext();
  auto shapeAttr = ShapeAttr::getChecked(emitError, shape, ShapeType::get(ctx));
  if (!shapeAttr)
    return {};
  return TensorType::getChecked(emitError, shapeAttr, elementType, deviceRef,
                                metadata);
}

TensorType TensorType::get(MLIRContext *context, ArrayRef<int64_t> shape,
                           DType dtype, DeviceRefAttr deviceRef,
                           DictionaryAttr metadata) {
  auto newShape = ShapeAttr::get(shape, ShapeType::get(context));
  return get(newShape, dtype, deviceRef, metadata);
}

TensorType TensorType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  MLIRContext *context, ArrayRef<int64_t> shape,
                                  DType dtype, DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  auto shapeAttr =
      ShapeAttr::getChecked(emitError, shape, ShapeType::get(context));
  if (!shapeAttr)
    return {};
  return TensorType::getChecked(emitError, shapeAttr, dtype, deviceRef,
                                metadata);
}

TensorType TensorType::get(MLIRContext *context, ArrayRef<TypedAttr> shape,
                           DType dtype, DeviceRefAttr deviceRef,
                           DictionaryAttr metadata) {
  auto newShape = ShapeAttr::get(shape, ShapeType::get(context));
  return get(newShape, dtype, deviceRef, metadata);
}

TensorType TensorType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  MLIRContext *context,
                                  ArrayRef<TypedAttr> shape, DType dtype,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  auto shapeAttr =
      ShapeAttr::getChecked(emitError, shape, ShapeType::get(context));
  if (!shapeAttr)
    return {};
  return TensorType::getChecked(emitError, shapeAttr, dtype, deviceRef,
                                metadata);
}

TensorType TensorType::getScalarType(Type elementType, DeviceRefAttr deviceRef,
                                     DictionaryAttr metadata) {
  return get(ArrayRef<TypedAttr>{}, elementType, deviceRef, metadata);
}

TensorType TensorType::getVectorSI64Type(MLIRContext *ctx, int64_t size,
                                         DeviceRefAttr deviceRef,
                                         DictionaryAttr metadata) {
  auto si64Ty = IntegerType::get(ctx, 64, IntegerType::Signed);
  return get(ArrayRef<TypedAttr>{ShapeAttr::getDim(ctx, size)}, si64Ty,
             deviceRef, metadata);
}

LayoutAttr TensorType::getLayoutAttr() {
  Attribute layoutAttrOrNull = getMetadata().get(kLayoutMetadataName);
  return llvm::dyn_cast_or_null<LayoutAttr>(layoutAttrOrNull);
}

IOKindAttr TensorType::getIOKindAttr() {
  return llvm::dyn_cast_or_null<IOKindAttr>(
      getMetadata().get(kIOKindMetadataName));
}

// TODO: Delete this method.
TensorType TensorType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                 Type dtype) const {
  auto shapeTy = ShapeType::get(getContext());

  TypedAttr shapeRef;
  if (shape.has_value())
    shapeRef = ShapeAttr::get(shape.value(), shapeTy);
  else
    shapeRef = KGEN::UnknownAttr::get(shapeTy);

  ErrorOr<DType> errOrDType = fromMLIRType(dtype);
  ASSERT_STREAM(!errOrDType.isError(),
                << "element type is not representable as an M::DType: "
                << errOrDType.takeError());
  return TensorType::get(shapeRef, *errOrDType, getDeviceRef(), getMetadata());
}

TensorType TensorType::cloneWith(ArrayRef<TypedAttr> shapeDims) const {
  auto shapeTy = ShapeType::get(getContext());
  auto shapeAttr = ShapeAttr::get(shapeDims, shapeTy);
  return TensorType::get(shapeAttr, getDtype(), getDeviceRef(), getMetadata());
}

TensorType TensorType::cloneWith(TypedAttr shapeAttr) const {
  return TensorType::get(shapeAttr, getDtype(), getDeviceRef(), getMetadata());
}

TensorType TensorType::cloneWith(DType dtype) const {
  return TensorType::get(getShapeAttr(), dtype, getDeviceRef(), getMetadata());
}

TensorType TensorType::cloneWith(LayoutAttr layoutAttr) const {
  NamedAttrList namedAttrs(getMetadata());
  namedAttrs.set(kLayoutMetadataName, layoutAttr);
  DictionaryAttr newMetadata = namedAttrs.getDictionary(getContext());
  return TensorType::get(getShapeAttr(), getDtype(), getDeviceRef(),
                         newMetadata);
}

TensorType TensorType::cloneWithoutLayout() const {
  NamedAttrList namedAttrs(getMetadata());
  namedAttrs.erase(kLayoutMetadataName);
  DictionaryAttr newMetadata = namedAttrs.getDictionary(getContext());
  return TensorType::get(getShapeAttr(), getDtype(), getDeviceRef(),
                         newMetadata);
}

TensorType TensorType::cloneWith(IOKindAttr ioKindAttr) const {
  NamedAttrList namedAttrs(getMetadata());
  namedAttrs.set(kIOKindMetadataName, ioKindAttr);
  DictionaryAttr newMetadata = namedAttrs.getDictionary(getContext());
  return TensorType::get(getShapeAttr(), getDtype(), getDeviceRef(),
                         newMetadata);
}

TensorType TensorType::cloneWithoutIOKind() const {
  NamedAttrList namedAttrs(getMetadata());
  namedAttrs.erase(kIOKindMetadataName);
  DictionaryAttr newMetadata = namedAttrs.getDictionary(getContext());
  return TensorType::get(getShapeAttr(), getDtype(), getDeviceRef(),
                         newMetadata);
}

BoolAttr TensorType::getIsMutableAttr() {
  return llvm::dyn_cast_or_null<BoolAttr>(
      getMetadata().get(kIsMutableMetadataName));
}

TensorType TensorType::cloneWithMut(BoolAttr isMutable) const {
  NamedAttrList namedAttrs(getMetadata());
  namedAttrs.set(kIsMutableMetadataName, isMutable);
  DictionaryAttr newMetadata = namedAttrs.getDictionary(getContext());
  return TensorType::get(getShapeAttr(), getDtype(), getDeviceRef(),
                         newMetadata);
}

TensorType TensorType::cloneWith(DictionaryAttr metadataAttr) const {
  return TensorType::get(getShapeAttr(), getDtype(), getDeviceRef(),
                         metadataAttr);
}

TensorType TensorType::cloneWith(DeviceRefAttr deviceRef) const {
  return TensorType::get(getShapeAttr(), getDtype(), deviceRef, getMetadata());
}

bool TensorType::hasRank() const {
  // TensorType::verify guarantees the shape attr is a ShapeAttr, which is
  // always ranked. Kept for source-compatibility with callers and for parity
  // with ShapedTypeInterface.
  return true;
}

size_t TensorType::getRank() const { return getTypedShapeAttr().getRank(); }

size_t TensorType::getRankNonZero() const {
  return std::max<size_t>(1, getRank());
}

bool TensorType::hasStaticShape() const {
  return hasRank() && getTypedShapeAttr().isStatic();
}

Type TensorType::getElementType() const {
  ErrorOr<Type> errOrType = toMLIRType(getContext(), getDtype());
  ASSERT_STREAM(!errOrType.isError(),
                << "element type is not representable as an mlir Type: "
                << errOrType.takeError());
  return *errOrType;
}

ShapeAttr TensorType::getTypedShapeAttr() const {
  return llvm::cast<ShapeAttr>(getShapeAttr());
}

SmallPtrSet<StringAttr, 4> TensorType::getUsedParamNames() const {
  if (auto shapeRef = llvm::dyn_cast<KGEN::ParamDeclRefAttr>(getShapeAttr()))
    return {shapeRef.getName()};
  SmallPtrSet<StringAttr, 4> res;
  mlir::AttrTypeWalker walker;
  walker.addWalk(
      [&](KGEN::ParamDeclRefAttr attr) { res.insert(attr.getName()); });
  walker.walk(getTypedShapeAttr());
  return res;
}

ArrayRef<TypedAttr> TensorType::getDims() const {
  return getTypedShapeAttr().getValues();
}

TypedAttr TensorType::getDim(size_t n) const { return getDims()[n]; }

bool TensorType::isParameterizedDim(size_t n) const {
  return isParameterizedDimParam(getDim(n));
}

bool TensorType::isIntegerDim(size_t n) const {
  return isIntegerDimParam(getDim(n));
}

bool TensorType::isParameterizedDimParam(TypedAttr attr) {
  bool isParamOrParamExpr = llvm::isa<KGEN::ParamDeclRefAttr>(attr) ||
                            llvm::isa<KGEN::ParamOperatorAttr>(attr);
  return isParamOrParamExpr && ShapeAttr::isDim(attr);
}

bool TensorType::isIntegerDimParam(TypedAttr attr) {
  return ShapeAttr::getStaticDim(attr).has_value();
}

SmallVector<int64_t> TensorType::getIntDims() const {
  return getTypedShapeAttr().getIntDims();
}

int64_t TensorType::getIntDim(size_t n) const {
  return getTypedShapeAttr().getIntDim(n);
}

TypedAttr TensorType::getSymbolicNumElements() const {
  return getTypedShapeAttr().getSymbolicNumElements();
}

std::optional<int64_t> TensorType::getNumElements() const {
  if (!hasRank())
    return {};

  TypedAttr attr = getSymbolicNumElements();
  auto numElements = ShapeAttr::getStaticDim(attr);
  if (!numElements.has_value())
    return {};

  // We allow one -1 dimension for now, which is a fancy parameter.
  // TODO(15523): maybe remove once -1 is handled more gracefully.
  if (*numElements < 0)
    return {};

  return *numElements;
}

TensorType TensorType::cloneWithNewDim(uint64_t dim, uint64_t size) {
  SmallVector<TypedAttr> shape(getDims());
  shape[dim] = ShapeAttr::getDim(shape.front().getContext(), size);
  auto shapeAttr = ShapeAttr::get(shape, ShapeType::get(getContext()));

  return cloneWith(shapeAttr);
}

//===----------------------------------------------------------------------===//
// ListType
//===----------------------------------------------------------------------===//

LogicalResult ListType::verify(function_ref<InFlightDiagnostic()> emitError,
                               Type elementType) {
  // check against things like `!mo.list<f32>`
  auto tensorTy = llvm::dyn_cast<TensorType>(elementType);
  if (!tensorTy) {
    return emitError() << "list can only contain tensor elements, but got "
                       << elementType;
  }

  // check against things like `!mo.list<!mo.tensor<Sh0, f32>>`
  auto shapeAttr = tensorTy.getShapeAttr();
  if (llvm::isa<KGEN::ParamDeclRefAttr>(shapeAttr))
    return emitError() << "tensor element shape cannot be a parameter";

  // check against things like `!mo.list<!mo.tensor<[20, D0], f32>>`
  if (auto dimsAttr = llvm::dyn_cast<MOSH::ShapeAttr>(shapeAttr)) {
    for (TypedAttr dimAttr : dimsAttr.getValues()) {
      if (llvm::isa<KGEN::ParamDeclRefAttr>(dimAttr))
        return emitError() << "tensor element dimensions cannot have parameter";
    }
  }

  return success();
}

DeviceRefAttr ListType::getDeviceRef() const {
  auto tensorType = llvm::cast<TensorType>(getElementType());
  return tensorType.getDeviceRef();
}

Type ListType::cloneWithNewDevice(DeviceRefAttr device) const {
  auto tensorType = llvm::cast<TensorType>(getElementType());
  return ListType::get(getContext(), tensorType.cloneWithNewDevice(device));
}

//===----------------------------------------------------------------------===//
// BundleType
//===----------------------------------------------------------------------===//

namespace {

/// Description of a bundle whose element tensor types can all be reconstructed
/// from a single shape/dtype/device-range tuple. Used by the BundleType
/// printer and parser to share the compact-form representation.
struct CompactBundleInfo {
  TypedAttr shapeAttr;
  DType dtype;
  // Shared device of every element. Non-null.
  DeviceRefAttr deviceRef;
  DictionaryAttr metadata;
  // ID of element 0 and element N-1. These are equal for single-element
  // bundles.
  DeviceId firstId;
  DeviceId lastId;
};

/// Returns a CompactBundleInfo if `elementTypes` can be pretty-printed in the
/// compact form. A bundle is compact-eligible iff all elements are TensorTypes
/// with identical shape, dtype, device label, and metadata, and their device
/// IDs are ordered and contiguous (`ids[i] == ids[0] + i`).
std::optional<CompactBundleInfo>
computeCompactBundleInfo(ArrayRef<Type> elementTypes) {
  if (elementTypes.empty())
    return std::nullopt;

  auto first = llvm::dyn_cast<TensorType>(elementTypes.front());
  if (!first)
    return std::nullopt;

  DeviceRefAttr firstDevice = first.getDeviceRef();
  if (!firstDevice)
    return std::nullopt;

  DeviceId firstId = firstDevice.getId();
  for (auto [idx, ty] : llvm::enumerate(elementTypes)) {
    auto tensor = llvm::dyn_cast<TensorType>(ty);
    if (!tensor)
      return std::nullopt;
    if (tensor.getShapeAttr() != first.getShapeAttr())
      return std::nullopt;
    if (tensor.getDtype() != first.getDtype())
      return std::nullopt;
    if (tensor.getMetadata() != first.getMetadata())
      return std::nullopt;
    DeviceRefAttr devRef = tensor.getDeviceRef();
    if (!devRef || devRef.getLabel() != firstDevice.getLabel())
      return std::nullopt;
    if (devRef.getId() != firstId + static_cast<DeviceId>(idx))
      return std::nullopt;
  }

  DeviceId lastId = firstId + static_cast<DeviceId>(elementTypes.size() - 1);
  return CompactBundleInfo{first.getShapeAttr(), first.getDtype(), firstDevice,
                           first.getMetadata(),  firstId,          lastId};
}

} // namespace

Type BundleType::parse(AsmParser &parser) {
  if (parser.parseLess() || parser.parseLSquare())
    return {};

  // Lookahead: the verbose form has a tensor type at this point; the compact
  // form has a shape `[...]` dim list (integers / `?` / param refs), which is
  // not a type.
  Type firstType;
  OptionalParseResult typeRes = parser.parseOptionalType(firstType);
  if (typeRes.has_value()) {
    if (failed(*typeRes))
      return {};

    // Verbose form: finish parsing the remaining tensor types.
    SmallVector<Type> elementTypes{firstType};
    while (succeeded(parser.parseOptionalComma())) {
      Type ty;
      if (parser.parseType(ty))
        return {};
      elementTypes.push_back(ty);
    }
    if (parser.parseRSquare() || parser.parseGreater())
      return {};
    return BundleType::get(parser.getContext(), elementTypes);
  }

  // Compact form: the outer `[` has been consumed; parse the shape dims that
  // would normally live inside it.
  MLIRContext *ctx = parser.getContext();
  auto shapeTy = parser.getBuilder().getType<ShapeType>();
  SmallVector<TypedAttr> shapeDims;
  if (MOSH::parseShapeValue(parser, shapeDims) || parser.parseRSquare())
    return {};
  auto shapeAttr = parser.getChecked<ShapeAttr>(shapeDims, shapeTy);
  if (!shapeAttr)
    return {};

  if (parser.parseComma())
    return {};

  FailureOr<DType> dtypeOr = mlir::FieldParser<DType>::parse(parser);
  if (failed(dtypeOr))
    return {};
  DType dtype = dtypeOr.value();

  // Optional device section: `, <label>:<id>[-<lastId>]`.
  DeviceRefAttr firstDevice;
  DeviceId firstId = 0;
  DeviceId lastId = 0;
  bool hasExplicitDevice = false;
  DictionaryAttr metadata;

  if (succeeded(parser.parseOptionalComma())) {
    StringRef deviceLabel;
    if (succeeded(parser.parseOptionalKeyword(&deviceLabel))) {
      hasExplicitDevice = true;
      if (parser.parseColon() || parser.parseInteger(firstId))
        return {};
      lastId = firstId;

      // MLIR's AsmParser lacks a parseMinus primitive. Use the fact that the
      // lexer tokenizes `-1` as a signed integer literal: try to consume an
      // optional signed integer; if present it must be negative, and its
      // absolute value is the range end.
      int64_t rangeEnd = 0;
      OptionalParseResult rangeRes = parser.parseOptionalInteger(rangeEnd);
      if (rangeRes.has_value()) {
        if (failed(*rangeRes))
          return {};
        if (rangeEnd >= 0) {
          parser.emitError(parser.getCurrentLocation())
              << "expected '-' before compact bundle device range end";
          return {};
        }
        lastId = static_cast<DeviceId>(-rangeEnd);
        if (lastId < firstId) {
          parser.emitError(parser.getCurrentLocation())
              << "compact bundle device range must be ordered "
              << "(firstId <= lastId)";
          return {};
        }
      }

      firstDevice = DeviceRefAttr::get(ctx, deviceLabel, firstId);

      // Optional metadata after the device section.
      if (succeeded(parser.parseOptionalComma())) {
        if (parser.parseAttribute(metadata))
          return {};
      }
    } else {
      // No device keyword; the comma introduces a metadata attribute.
      if (parser.parseAttribute(metadata))
        return {};
    }
  }

  if (parser.parseGreater())
    return {};

  if (!hasExplicitDevice)
    firstDevice = DeviceRefAttr::getDefault(ctx);
  if (!metadata)
    metadata = DictionaryAttr::get(ctx);

  // Materialize one TensorType per device ID in the range.
  size_t numElements =
      hasExplicitDevice ? static_cast<size_t>(lastId - firstId + 1) : 1;
  SmallVector<Type> elementTypes;
  elementTypes.reserve(numElements);
  for (size_t i = 0; i < numElements; ++i) {
    DeviceRefAttr devRef =
        hasExplicitDevice
            ? DeviceRefAttr::get(ctx, firstDevice.getLabel(), firstId + i)
            : firstDevice;
    auto tensor =
        parser.getChecked<TensorType>(ctx, shapeAttr, dtype, devRef, metadata);
    if (!tensor)
      return {};
    elementTypes.push_back(tensor);
  }
  return BundleType::get(ctx, elementTypes);
}

void BundleType::print(AsmPrinter &printer) const {
  ArrayRef<Type> elementTypes = getElementTypes();

  if (auto info = computeCompactBundleInfo(elementTypes)) {
    printer << '<';
    MOSH::printShapeTypedAttr(printer, info->shapeAttr);
    printer << ", " << info->dtype.getAsString();

    // Elide the device section only when the parser would re-derive the
    // correct bundle from its absence: a single element on the default host
    // device.
    auto defaultDev = DeviceRefAttr::getDefault(getContext());
    bool singleDefault =
        elementTypes.size() == 1 && info->deviceRef == defaultDev;
    if (!singleDefault) {
      printer << ", " << info->deviceRef.getLabel() << ':' << info->firstId;
      if (info->firstId != info->lastId)
        printer << '-' << info->lastId;
    }

    if (info->metadata && !info->metadata.empty())
      printer << ", " << info->metadata;

    printer << '>';
    return;
  }

  // Verbose fallback: print the full list of element tensor types.
  printer << "<[";
  llvm::interleaveComma(elementTypes, printer,
                        [&](Type ty) { printer.printType(ty); });
  printer << "]>";
}

LogicalResult BundleType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<Type> elementTypes) {
  if (elementTypes.empty())
    return emitError() << "bundle must have at least one element";
  bool allTensors = llvm::all_of(
      elementTypes, [](Type t) { return llvm::isa<TensorType>(t); });
  if (!allTensors)
    return emitError() << "bundle elements must all be !mo.tensor types";
  return success();
}

//===----------------------------------------------------------------------===//
// Device list printing / parsing
//===----------------------------------------------------------------------===//

namespace M::MO {

void printDeviceList(AsmPrinter &printer, ArrayRef<DeviceRefAttr> devices) {
  if (devices.empty())
    return;

  // Try the compact `label:first[-last]` form when all devices share a label
  // and have consecutive IDs starting at firstId.
  StringRef label = devices.front().getLabel();
  DeviceId firstId = devices.front().getId();
  bool isContiguous = true;
  for (auto [i, d] : llvm::enumerate(devices)) {
    if (d.getLabel() != label ||
        d.getId() != firstId + static_cast<DeviceId>(i)) {
      isContiguous = false;
      break;
    }
  }
  if (isContiguous) {
    printer << label << ':' << firstId;
    if (devices.size() > 1)
      printer << '-' << (firstId + devices.size() - 1);
    return;
  }

  // Fallback: comma-separated `label:id` pairs.
  llvm::interleaveComma(devices, printer, [&](DeviceRefAttr d) {
    printer << d.getLabel() << ':' << d.getId();
  });
}

ParseResult parseDeviceList(AsmParser &parser,
                            SmallVectorImpl<DeviceRefAttr> &devices) {
  MLIRContext *ctx = parser.getContext();

  // Parse one entry: `label:id[-lastId]` (compact range) or just `label:id`.
  // We accept a comma-separated list of these.  The first entry may be a
  // range, e.g. `gpu:0-3, cpu:0` -- ranges and single entries can mix.
  auto parseOneEntry = [&]() -> ParseResult {
    StringRef label;
    if (parser.parseKeyword(&label))
      return failure();
    DeviceId firstId = 0;
    if (parser.parseColon() || parser.parseInteger(firstId))
      return failure();
    DeviceId lastId = firstId;

    // Lookahead for `-<lastId>` as a signed integer literal (lexer
    // tokenizes `-1` as a single signed integer).
    int64_t rangeEnd = 0;
    OptionalParseResult rangeRes = parser.parseOptionalInteger(rangeEnd);
    if (rangeRes.has_value()) {
      if (failed(*rangeRes))
        return failure();
      if (rangeEnd >= 0)
        return parser.emitError(parser.getCurrentLocation(),
                                "expected '-' before compact device range end");
      lastId = static_cast<DeviceId>(-rangeEnd);
      if (lastId < firstId)
        return parser.emitError(
            parser.getCurrentLocation(),
            "compact device range must be ordered (firstId <= lastId)");
    }

    for (DeviceId id = firstId; id <= lastId; ++id)
      devices.push_back(DeviceRefAttr::get(ctx, label, id));
    return success();
  };

  if (parseOneEntry())
    return failure();
  while (succeeded(parser.parseOptionalComma())) {
    if (parseOneEntry())
      return failure();
  }
  return success();
}

} // namespace M::MO

//===----------------------------------------------------------------------===//
// ScalarType
//===----------------------------------------------------------------------===//

Type ScalarType::getDType() const {
  ErrorOr<Type> errOrType = toMLIRType(getContext(), getDtype());
  ASSERT_STREAM(!errOrType.isError(),
                << "scalar type is not representable as an mlir Type: "
                << errOrType.takeError());
  return *errOrType;
}

//===----------------------------------------------------------------------===//
// MODialect type support
//===----------------------------------------------------------------------===//

void MODialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "GraphCompiler/GraphCompiler/MODialect/MOTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

BufferType BufferType::cloneWith(DeviceRefAttr deviceRef) const {
  return BufferType::get(getContext(), getShapeAttr(), getDtype(), deviceRef,
                         getMetadata());
}

Type BufferType::parse(AsmParser &parser) {
  TensorLikeTypeParser bufferParser(parser);

  if (bufferParser.parse())
    return {};

  return bufferParser.getType<BufferType>();
}

void BufferType::print(AsmPrinter &printer) const {
  printTensorLikeType(printer, getShapeAttr(), getDtype(), getDeviceRef(),
                      getMetadata());
}

/// Verifies that the given dimensions represent a valid tensor shape, and
/// that the element type in DType form is also representable as an MLIR type.
LogicalResult BufferType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 TypedAttr shapeAttr, DType dtype,
                                 DeviceRefAttr deviceRef,
                                 DictionaryAttr metadata) {
  if (!deviceRef)
    return emitError() << "no device attribute was defined";

  auto shape = llvm::dyn_cast<ShapeAttr>(shapeAttr);
  if (!shape)
    return emitError() << "invalid shape attribute, only MOSH::ShapeAttr is "
                          "allowed for buffer types, got: "
                       << shapeAttr;

  for (TypedAttr dimAttr : shape.getValues()) {
    if (auto dimInt = ShapeAttr::getStaticDim(dimAttr)) {
      if (*dimInt < 0)
        return emitError()
               << "invalid dimension, expect all dimensions >= 0 got: "
               << *dimInt;
    }
  }

  if (!llvm::all_of(shape.getValues(), ShapeAttr::isDim))
    return emitError() << "buffer dimensions must be a getDimType()";

  // The device specification used to reside in the metadata dictionary,
  // sanity check that the buffer type is not using the old format.
  if (metadata && metadata.contains("device"))
    return emitError()
           << "device attribute no longer resides in the metadata dictionary";

  return success();
}

// Base builder which all others call
BufferType BufferType::get(::mlir::MLIRContext *context, TypedAttr shapeAttr,
                           DType dtype, DeviceRefAttr deviceRef,
                           DictionaryAttr metadata) {
  // Ensure device ref is always populated
  if (!deviceRef)
    deviceRef = DeviceRefAttr::getDefault(context);

  return Base::get(context, std::move(shapeAttr), std::move(dtype),
                   std::move(deviceRef), std::move(metadata));
}

// Checked base builder
BufferType BufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  ::mlir::MLIRContext *context,
                                  TypedAttr shapeAttr, DType dtype,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  if (failed(verify(emitError, shapeAttr, dtype, deviceRef, metadata)))
    return {};
  return BufferType::get(context, shapeAttr, dtype, deviceRef, metadata);
}

BufferType BufferType::get(TypedAttr shapeAttr, DType dtype,
                           DeviceRefAttr deviceRef, DictionaryAttr metadata) {
  MLIRContext *ctx = shapeAttr.getContext();
  if (!metadata)
    metadata = DictionaryAttr::get(ctx);
  return get(ctx, shapeAttr, dtype, deviceRef, metadata);
}

BufferType BufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  TypedAttr shapeAttr, DType dtype,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  if (failed(verify(emitError, shapeAttr, dtype, deviceRef, metadata)))
    return {};
  return BufferType::get(shapeAttr, dtype, deviceRef, metadata);
}

BufferType BufferType::get(TypedAttr shapeAttr, Type elementType,
                           DeviceRefAttr deviceRef, DictionaryAttr metadata) {
  ErrorOr<DType> errOrDType = fromMLIRType(elementType);
  ASSERT_STREAM(!errOrDType.isError(),
                << "element type is not representable as a DType: "
                << errOrDType.takeError());
  return get(shapeAttr, *errOrDType, deviceRef, metadata);
}

BufferType BufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  TypedAttr shapeAttr, Type elementType,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  ErrorOr<DType> dtype = fromMLIRType(elementType);
  if (!dtype) {
    emitError() << dtype.takeError();
    return {};
  }
  if (failed(verify(emitError, shapeAttr, *dtype, deviceRef, metadata)))
    return {};
  return BufferType::get(shapeAttr, *dtype, deviceRef, metadata);
}

BufferType BufferType::get(ArrayRef<TypedAttr> shape, Type elementType,
                           DeviceRefAttr deviceRef, DictionaryAttr metadata) {
  auto ctx = elementType.getContext();
  auto newShape = ShapeAttr::get(shape, ShapeType::get(ctx));
  return get(newShape, elementType, deviceRef, metadata);
}

BufferType BufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<TypedAttr> shape, Type elementType,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  auto ctx = elementType.getContext();
  auto shapeAttr = ShapeAttr::getChecked(emitError, shape, ShapeType::get(ctx));
  if (!shapeAttr)
    return {};
  return BufferType::getChecked(emitError, shapeAttr, elementType, deviceRef,
                                metadata);
}

BufferType BufferType::get(TensorType tensorType) {
  return get(tensorType.getShapeAttr(), tensorType.getElementType(),
             tensorType.getDeviceRef(), tensorType.getMetadata());
}

BufferType BufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  TensorType tensorType) {
  if (failed(verify(emitError, tensorType.getShapeAttr(), tensorType.getDtype(),
                    tensorType.getDeviceRef(), tensorType.getMetadata())))
    return {};
  return BufferType::get(tensorType);
}

TensorType BufferType::toTensorType(BufferType ty) {
  auto shapeAttr = ty.getShapeAttr();
  auto dtype = ty.getDtype();
  auto deviceRef = ty.getDeviceRef();
  auto metadata = ty.getMetadata();

  return TensorType::get(shapeAttr, dtype, deviceRef, metadata);
}

TensorType BufferType::toTensorType() {
  return BufferType::toTensorType(*this);
}

TensorType M::MO::castFromTensorLike(Type ty) {
  if (auto tensorTy = dyn_cast<TensorType>(ty))
    return tensorTy;
  return BufferType::toTensorType(cast<MO::BufferType>(ty));
}

TensorType M::MO::dynCastFromTensorLike(Type ty) {
  if (auto tensorTy = dyn_cast<TensorType>(ty))
    return tensorTy;
  if (auto bufferTy = dyn_cast<BufferType>(ty))
    return BufferType::toTensorType(bufferTy);
  return nullptr;
}

size_t BufferType::getRank() const {
  return cast<MOSH::ShapeAttr>(getShapeAttr()).getRank();
}

size_t BufferType::getRankNonZero() const {
  size_t rank = getRank();
  return rank == 0 ? rank + 1 : rank;
}

llvm::SmallVector<int64_t> M::MO::getStridesForTensor(MO::TensorType tensor,
                                                      bool unknown) {
  size_t rank = std::max(tensor.getRank(), static_cast<size_t>(1));
  llvm::SmallVector<int64_t> strides(rank, mlir::ShapedType::kDynamic);

  if (unknown)
    return strides;

  strides.back() = 1;
  for (int64_t i = rank - 2; i >= 0; --i) {
    if (tensor.getIntDim(i + 1) == mlir::ShapedType::kDynamic)
      break;
    strides[i] = strides[i + 1] * tensor.getIntDim(i + 1);
  }

  return strides;
}

//===----------------------------------------------------------------------===//
// OpaqueType
//===----------------------------------------------------------------------===//

OpaqueType OpaqueType::get(MLIRContext *ctx, StringAttr symbol,
                           DictionaryAttr parameters) {
  if (!parameters)
    parameters = DictionaryAttr::get(ctx);
  return Base::get(ctx, symbol, parameters);
}

OpaqueType OpaqueType::get(StringAttr symbol) {
  auto ctx = symbol.getContext();
  return get(ctx, symbol, DictionaryAttr::get(ctx));
}

OpaqueType OpaqueType::get(StringAttr symbol, DictionaryAttr parameters) {
  return get(symbol.getContext(), symbol, parameters);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "GraphCompiler/GraphCompiler/MODialect/MOTypes.cpp.inc"
