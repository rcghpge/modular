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
#include "Support/Compiler/MLIRDType.h"
#include "Support/ErrorOr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// ODS Boilerplate
//===----------------------------------------------------------------------===//
namespace mlir {
/// Parse a dtype.
template <>
struct FieldParser<DType> {
  static FailureOr<DType> parse(AsmParser &parser) {
    StringRef value;
    if (parser.parseKeyword(&value))
      return failure();
    return DType::getFromString(value);
  }
};

} // namespace mlir

static std::string asString(Type type) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << type;
  return str;
}

//===----------------------------------------------------------------------===//
// Tensor
//===----------------------------------------------------------------------===//

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
              "ParamDeclRefAttr";
  }
  return p.emitError(p.getCurrentLocation())
         << "couldn't parse shape parameter: expected `?`, shape reference, or "
            "literal shape reference (i.e. `[...]`)";
}

static auto printShapeParam = MOSH::printShapeTypedAttr;

/// Verifies that the given dimensions represent a valid tensor shape, and
/// that the element type in DType form is also representable as an MLIR type.
LogicalResult TensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 TypedAttr shapeAttr, DType dtype,
                                 std::optional<Attribute> metadata) {
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

  for (TypedAttr dimAttr: attr.getValues()) {
    if (auto dimIntAttr = dimAttr.dyn_cast<IntegerAttr>()) {
      if (dimIntAttr.getInt() < 0)
        return emitError() << "invalid dimension, expect all dimensions >= 0 got: " << dimIntAttr.getInt();
    }
  }
  return success();
}

// TODO : These are shamelessly copied out of MGP. Need to align
ErrorOr<DType> static fromMLIRType(Type mlirType) {
  if (auto unkType = mlir::dyn_cast<NoneType>(mlirType))
    return DType(); // invalid denotes unknown.
  if (auto intType = mlir::dyn_cast<IntegerType>(mlirType)) {
    FailureOr<DType> dtype = getEquivalentDType(intType);
    if (succeeded(dtype) && !(dtype->isInvalid()))
      return *dtype;
  }
  if (auto floatType = mlir::dyn_cast<FloatType>(mlirType)) {
    FailureOr<DType> dtype = getEquivalentDType(floatType);
    if (succeeded(dtype) && !(dtype->isInvalid()))
      return *dtype;
  }
  return Error(Twine("unable to represent MLIR Type '") + asString(mlirType) +
               "' as an M::DType type.");
}

ErrorOr<Type> static toMLIRType(MLIRContext *context, DType dType) {
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

TensorType TensorType::get(TypedAttr shapeAttr, DType dtype,
                           std::optional<Attribute> metadata) {
  return get(shapeAttr.getContext(), shapeAttr, dtype, metadata);
}

TensorType TensorType::get(TypedAttr shapeAttr, Type elementType,
                           std::optional<Attribute> metadata) {
  ErrorOr<DType> errOrDType = fromMLIRType(elementType);
  assert(!errOrDType.isError() &&
         "element type is not representable as a DType");
  return get(shapeAttr, *errOrDType, metadata);
}

TensorType TensorType::get(ArrayRef<int64_t> shape, Type elementType,
                           std::optional<Attribute> metadata) {
  auto ctx = elementType.getContext();
  auto newShape = ShapeAttr::get(shape, ShapeType::get(ctx));
  return get(newShape, elementType, metadata);
}

TensorType TensorType::get(ArrayRef<TypedAttr> shape, Type elementType,
                           std::optional<Attribute> metadata) {
  auto ctx = elementType.getContext();
  auto newShape = ShapeAttr::get(shape, ShapeType::get(ctx));
  return get(newShape, elementType, metadata);
}

TensorType TensorType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                 Type dtype,
                                 std::optional<Attribute> metadata) const {
  auto shapeTy = ShapeType::get(getContext());

  TypedAttr shapeRef;
  if (shape.has_value())
    shapeRef = ShapeAttr::get(shape.value(), shapeTy);
  else
    shapeRef = KGEN::UnknownAttr::get(shapeTy);

  ErrorOr<DType> errOrDType = fromMLIRType(dtype);
  assert(!errOrDType.isError() &&
         "element type is not representable as an M::DType");
  if (!metadata.has_value())
    metadata = getMetadata();
  return TensorType::get(shapeRef, *errOrDType, metadata);
}

TensorType TensorType::cloneWith(TypedAttr shapeAttr) const {
  return TensorType::get(shapeAttr.getContext(), shapeAttr, getDtype(),
                         getMetadata());
}

bool TensorType::hasRank() const {
  // Rank is only known if we have a concrete attribute
  if (llvm::isa<KGEN::UnknownAttr, KGEN::ParamDeclRefAttr>(getShapeAttr()))
    return false;
  return true;
}

size_t TensorType::getRank() const {
  assert(hasRank() && "cannot get rank if shape is not known");
  return getTypedShapeAttr().getRank();
}

bool TensorType::hasStaticShape() const {
  return hasRank() && getTypedShapeAttr().isStatic();
}

Type TensorType::getElementType() const {
  ErrorOr<Type> errOrType = toMLIRType(getContext(), getDtype());
  assert(!errOrType.isError() &&
         "element type is not representable as an mlir Type");
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

ErrorOr<TensorType> TensorType::broadcastShapes(TensorType x, TensorType y) {
  if (!x.hasRank() || !y.hasRank())
    return Error(Twine("Guards should have rejected unranked inputs."));

  UNWRAP_ERROR(result, ShapeAttr::broadcastShapes(x.getTypedShapeAttr(),
                                                  y.getTypedShapeAttr()));

  return MO::TensorType::get(result, x.getElementType());
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
