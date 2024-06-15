//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MO dialect types.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MODialect/MOAttrs.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/ErrorOr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MO;

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

/// Name of optional attribute holding the #M.device_ref on which a MO
/// tensor should be stored.
constexpr StringRef kTypeDeviceRefAttrName = "device";

static ParseResult parseShapeParam(AsmParser &p, TypedAttr &shapeAttr) {
  auto shapeTy = p.getBuilder().getType<ShapeType>();
  OptionalParseResult res = MOSH::parseShapeAttrOrUnknown(p, shapeAttr);
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

Type TensorType::parse(AsmParser &p) {
  MLIRContext *ctx = p.getContext();
  mlir::SMLoc loc = p.getCurrentLocation();

  // parse '<'
  if (p.parseLess())
    return {};

  // parse shape
  TypedAttr shapeAttr;
  if (parseShapeParam(p, shapeAttr))
    return {};

  // parse dtype
  if (p.parseComma())
    return {};
  FailureOr<DType> dtypeOr = mlir::FieldParser<DType>::parse(p);
  if (failed(dtypeOr)) {
    p.emitError(p.getCurrentLocation(), "invalid dtype in MO::TensorType");
    return {};
  }
  DType dtype = dtypeOr.value();

  // parse optional metadata
  DictionaryAttr metadata;
  if (succeeded(p.parseOptionalComma()) && p.parseAttribute(metadata))
    return {};
  if (!metadata)
    metadata = DictionaryAttr::get(ctx);

  // parse '>'
  if (p.parseGreater())
    return {};

  return p.getChecked<TensorType>(loc, ctx, shapeAttr, dtype, metadata);
}

void TensorType::print(AsmPrinter &p) const {
  p << '<';
  MOSH::printShapeTypedAttr(p, getShapeAttr());
  p << ", " << getDtype().getAsString();

  DictionaryAttr metadata = getMetadata();
  if (!metadata.empty())
    p << ", " << metadata;

  p << '>';
}

/// Verifies that the given dimensions represent a valid tensor shape, and
/// that the element type in DType form is also representable as an MLIR type.
LogicalResult TensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 TypedAttr shapeAttr, DType dtype,
                                 DictionaryAttr metadata) {
  if (auto unkAttr = llvm::dyn_cast<KGEN::UnknownAttr>(shapeAttr)) {
    if (Type ty = unkAttr.getType(); !llvm::isa<ShapeType>(ty)) {
      return emitError() << "unknown shape attribute must have ShapeType, got "
                         << ty;
    }
    return success();
  }
  if (auto declRef = llvm::dyn_cast<KGEN::ParamDeclRefAttr>(shapeAttr)) {
    if (Type ty = declRef.getType(); !llvm::isa<ShapeType>(ty)) {
      return emitError()
             << "shape parameter reference must have ShapeType, got " << ty;
    }
    return success();
  }
  if (auto indexRef = ::dyn_cast<KGEN::ParamIndexRefAttr>(shapeAttr)) {
    if (Type ty = indexRef.getType(); !::isa<ShapeType>(ty)) {
      return emitError()
             << "shape parameter reference must have ShapeType, got " << ty;
    }
    return success();
  }

  auto attr = llvm::dyn_cast<ShapeAttr>(shapeAttr);
  if (!attr)
    return emitError() << "invalid shape attribute " << shapeAttr;

  for (TypedAttr dimAttr : attr.getValues()) {
    if (auto dimIntAttr = llvm::dyn_cast<IntegerAttr>(dimAttr)) {
      if (dimIntAttr.getInt() < 0)
        return emitError()
               << "invalid dimension, expect all dimensions >= 0 got: "
               << dimIntAttr.getInt();
    }
  }
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
  ErrorOr<DType> errOr = fromMLIRType(ty);
  return errOr.isError() ? DType::invalid : *errOr;
}

TensorType TensorType::get(TypedAttr shapeAttr, DType dtype,
                           DictionaryAttr metadata) {
  MLIRContext *ctx = shapeAttr.getContext();
  if (!metadata)
    metadata = DictionaryAttr::get(ctx);
  return get(ctx, shapeAttr, dtype, metadata);
}

TensorType TensorType::get(TypedAttr shapeAttr, Type elementType,
                           DictionaryAttr metadata) {
  ErrorOr<DType> errOrDType = fromMLIRType(elementType);
  ASSERT_STREAM(!errOrDType.isError(),
                << "element type is not representable as a DType: "
                << errOrDType.takeError());
  return get(shapeAttr, *errOrDType, metadata);
}

TensorType TensorType::get(ArrayRef<int64_t> shape, Type elementType,
                           DictionaryAttr metadata) {
  auto ctx = elementType.getContext();
  auto newShape = ShapeAttr::get(shape, ShapeType::get(ctx));
  return get(newShape, elementType, metadata);
}

TensorType TensorType::get(ArrayRef<TypedAttr> shape, Type elementType,
                           DictionaryAttr metadata) {
  auto ctx = elementType.getContext();
  auto newShape = ShapeAttr::get(shape, ShapeType::get(ctx));
  return get(newShape, elementType, metadata);
}

TensorType TensorType::get(MLIRContext *context, ArrayRef<int64_t> shape,
                           DType dtype, DictionaryAttr metadata) {
  auto newShape = ShapeAttr::get(shape, ShapeType::get(context));
  return get(newShape, dtype, metadata);
}

TensorType TensorType::get(MLIRContext *context, ArrayRef<TypedAttr> shape,
                           DType dtype, DictionaryAttr metadata) {
  auto newShape = ShapeAttr::get(shape, ShapeType::get(context));
  return get(newShape, dtype, metadata);
}

TensorType TensorType::getScalarF32Type(MLIRContext *ctx,
                                        DictionaryAttr metadata) {
  return get(SmallVector<TypedAttr>{}, Float32Type::get(ctx), metadata);
}

TensorType TensorType::getScalarF64Type(MLIRContext *ctx,
                                        DictionaryAttr metadata) {
  return get(SmallVector<TypedAttr>{}, Float64Type::get(ctx), metadata);
}

TensorType TensorType::getScalarSI32Type(MLIRContext *ctx,
                                         DictionaryAttr metadata) {
  auto si32Ty =
      IntegerType::get(ctx, /*width=*/32, /*signedness=*/IntegerType::Signed);
  return get(SmallVector<TypedAttr>{}, si32Ty, metadata);
}

TensorType TensorType::getScalarSI64Type(MLIRContext *ctx,
                                         DictionaryAttr metadata) {
  auto si64Ty =
      IntegerType::get(ctx, /*width=*/64, /*signedness=*/IntegerType::Signed);
  return get(SmallVector<TypedAttr>{}, si64Ty, metadata);
}

TensorType TensorType::getVectorSI64Type(MLIRContext *ctx, int64_t size,
                                         DictionaryAttr metadata) {
  auto si64Ty = IntegerType::get(ctx, 64, IntegerType::Signed);
  auto idxTy = IndexType::get(ctx);
  return get(SmallVector<TypedAttr>(1, IntegerAttr::get(idxTy, size)), si64Ty,
             metadata);
}

LayoutAttr TensorType::getLayoutAttr() {
  Attribute layoutAttrOrNull = getMetadata().get(kLayoutMetadataName);
  return llvm::dyn_cast_or_null<LayoutAttr>(layoutAttrOrNull);
}

TensorType TensorType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                 Type dtype, DictionaryAttr metadata) const {
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
  return TensorType::get(shapeRef, *errOrDType, metadata);
}

TensorType TensorType::cloneWith(ArrayRef<TypedAttr> shapeDims) const {
  auto shapeTy = ShapeType::get(getContext());
  auto shapeAttr = ShapeAttr::get(shapeDims, shapeTy);
  return TensorType::get(shapeAttr.getContext(), shapeAttr, getDtype(),
                         getMetadata());
}

TensorType TensorType::cloneWith(TypedAttr shapeAttr) const {
  return TensorType::get(shapeAttr.getContext(), shapeAttr, getDtype(),
                         getMetadata());
}

TensorType TensorType::cloneWith(DType dtype) const {
  return TensorType::get(getContext(), getShapeAttr(), dtype, getMetadata());
}

TensorType TensorType::cloneWith(LayoutAttr layoutAttr) const {
  NamedAttrList namedAttrs(getMetadata());
  namedAttrs.set(kLayoutMetadataName, layoutAttr);
  DictionaryAttr newMetadata = namedAttrs.getDictionary(getContext());
  return TensorType::get(getContext(), getShapeAttr(), getDtype(), newMetadata);
}

TensorType TensorType::cloneWith(DictionaryAttr metadataAttr) const {
  return TensorType::get(getContext(), getShapeAttr(), getDtype(),
                         metadataAttr);
}

TensorType TensorType::cloneWithUnknownRank() const {
  return TensorType::get(getContext(),
                         KGEN::UnknownAttr::get(ShapeType::get(getContext())),
                         getDtype(), getMetadata());
}

TensorType TensorType::cloneWithUnknownDims(int64_t rank) const {
  SmallVector<TypedAttr> shapes(
      rank, KGEN::UnknownAttr::get(IndexType::get(getContext())));
  return TensorType::get(
      getContext(), MOSH::ShapeAttr::get(shapes, ShapeType::get(getContext())),
      getDtype(), getMetadata());
}

TensorType TensorType::cloneWith(DeviceRefAttr device) const {
  NamedAttrList namedAttrs(getMetadata());
  namedAttrs.set(kTypeDeviceRefAttrName, device);
  DictionaryAttr newMetadata = namedAttrs.getDictionary(getContext());
  return TensorType::get(getContext(), getShapeAttr(), getDtype(), newMetadata);
}

bool TensorType::isDynamicallyRanked() const { return !hasRank(); }

bool TensorType::hasRank() const {
  // Rank is only known if we have a concrete attribute
  if (llvm::isa<KGEN::UnknownAttr, KGEN::ParamDeclRefAttr>(getShapeAttr()))
    return false;
  return true;
}

size_t TensorType::getRank() const {
  ASSERT_STREAM(hasRank(), << "cannot get rank if shape is not known");
  return getTypedShapeAttr().getRank();
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
  if (llvm::isa<KGEN::UnknownAttr>(getShapeAttr()))
    return {};
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

bool TensorType::isUnknownDim(size_t n) const {
  return isUnknownDimParam(getDim(n));
}

bool TensorType::isParameterizedDim(size_t n) const {
  return isParameterizedDimParam(getDim(n));
}

bool TensorType::isIntegerDim(size_t n) const {
  return isIntegerDimParam(getDim(n));
}

bool TensorType::isUnknownDimParam(TypedAttr attr) {
  return llvm::isa<KGEN::UnknownAttr>(attr) && attr.getType().isIndex();
}

bool TensorType::isParameterizedDimParam(TypedAttr attr) {
  bool isParamOrParamExpr = llvm::isa<KGEN::ParamDeclRefAttr>(attr) ||
                            llvm::isa<KGEN::ParamOperatorAttr>(attr);
  return isParamOrParamExpr && attr.getType().isIndex();
}

bool TensorType::isIntegerDimParam(TypedAttr attr) {
  return llvm::isa<IntegerAttr>(attr) && attr.getType().isIndex();
}

SmallVector<int64_t> TensorType::getIntDims() const {
  return getTypedShapeAttr().getIntDims();
}

int64_t TensorType::getIntDim(size_t n) const {
  return getTypedShapeAttr().getIntDim(n);
}

bool TensorType::hasUnknownShape() const {
  if (llvm::isa<KGEN::UnknownAttr>(getShapeAttr()))
    return true;
  if (llvm::isa<KGEN::ParamDeclRefAttr>(getShapeAttr()))
    return false;
  bool res = false;
  mlir::AttrTypeWalker walker;
  walker.addWalk([&](KGEN::UnknownAttr attr) {
    res = true;
    return WalkResult::interrupt();
  });
  walker.walk(getTypedShapeAttr());
  return res;
};

TypedAttr TensorType::getSymbolicNumElements() const {
  return getTypedShapeAttr().getSymbolicNumElements();
}

std::optional<int64_t> TensorType::getNumElements() const {
  if (!hasRank())
    return {};

  TypedAttr attr = getSymbolicNumElements();
  auto resultElements = mlir::dyn_cast_or_null<IntegerAttr>(attr);
  if (!resultElements)
    return {};

  // We allow one -1 dimension for now, which is a fancy parameter.
  // TODO(15523): maybe remove once -1 is handled more gracefully.
  if (resultElements.getInt() < 0)
    return {};

  return resultElements.getInt();
}

ErrorOr<TensorType> TensorType::broadcastShapes(TensorType x, TensorType y) {
  if (!x.hasRank() || !y.hasRank())
    return Error("Guards should have rejected unranked inputs.");
  if (x.getElementType() != y.getElementType())
    return Error("Tensors must have the same element type for "
                 "MO::TensorType::broadcastShapes.");

  auto resultOr =
      ShapeAttr::broadcastShapes(x.getTypedShapeAttr(), y.getTypedShapeAttr());
  if (resultOr.isError())
    return resultOr.takeError();

  return MO::TensorType::get(std::move(*resultOr), x.getElementType());
}

DeviceRefAttr TensorType::getDeviceRef() const {
  return dyn_cast_if_present<DeviceRefAttr>(
      getMetadata().get(kTypeDeviceRefAttrName));
}

TensorType TensorType::cloneWithNewDim(uint64_t dim, uint64_t size) {
  SmallVector<TypedAttr> shape(getDims());
  shape[dim] = mlir::IntegerAttr::get(shape.front().getType(), size);
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
#include "GenericML/GraphCompiler/MODialect/MOTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "GenericML/GraphCompiler/MODialect/MOTypes.cpp.inc"
