//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MODialect/MOTypes.h"
#include "GenericML/MODialect/MOAttrs.h"
#include "GenericML/MODialect/MODialect.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
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

std::string asString(Type type) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << type;
  return str;
}

//===----------------------------------------------------------------------===//
// ShapeParameter
//===----------------------------------------------------------------------===//

static void printShapeParam(AsmPrinter &p, TypedAttr shapeAttr) {
  if (isa<KGEN::UnknownAttr>(shapeAttr)) {
    p << "?";
  } else if (auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(shapeAttr)) {
    KGEN::printParamName(p, declRef.getName());
  } else {
    p << "[";
    printShapeValue(p, cast<MO::ShapeAttr>(shapeAttr).getValues());
    p << "]";
  }
}

/// This version is required by `custom<ShapeParam>`
static void printShapeParam(AsmPrinter &p, TypedAttr shapeAttr,
                            ArrayRef<int64_t> shape) {
  printShapeParam(p, shapeAttr);
}

static ParseResult parseShapeParam(AsmParser &p,
                                   FailureOr<TypedAttr> &shapeAttr) {
  auto shapeTy = p.getBuilder().getType<MO::ShapeType>();
  if (succeeded(p.parseOptionalQuestion())) {
    shapeAttr = {KGEN::UnknownAttr::get(shapeTy)};
    return success();
  }

  if (succeeded(p.parseOptionalLSquare())) {
    FailureOr<llvm::SmallVector<TypedAttr>> values;
    if (failed(p.parseOptionalRSquare())) {
      if (parseShapeValue(p, values) || p.parseRSquare())
        return failure();
    } else {
      values.emplace();
    }

    auto tmpAttr = p.getChecked<MO::ShapeAttr>(values.value(), shapeTy);
    if (!tmpAttr)
      return failure();
    shapeAttr = {tmpAttr};
    return success();
  }

  if (StringAttr name; succeeded(KGEN::parseParamName(p, name))) {
    shapeAttr = {KGEN::ParamDeclRefAttr::get(name, shapeTy)};
    return success();
  }
  return p.emitError(p.getCurrentLocation())
         << "couldn't parse shape parameter: expected `?`, shape reference, or "
            "literal shape reference (i.e. `[...]`)";
}

static ParseResult parseShapeParam(AsmParser &p,
                                   FailureOr<TypedAttr> &shapeAttr,
                                   FailureOr<SmallVector<int64_t>> &shape) {
  auto parseResult = parseShapeParam(p, shapeAttr);
  if (succeeded(parseResult)) {
    if (auto attr = dyn_cast<MO::ShapeAttr>(shapeAttr.value()))
      shape = attr.getIntDims();
    else
      shape.emplace();
  }
  return parseResult;
}

//===----------------------------------------------------------------------===//
// Tensor
//===----------------------------------------------------------------------===//

/// Verifies that the given dimensions represent a valid tensor shape, and
/// that the element type in DType form is also representable as an MLIR type.
LogicalResult TensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 TypedAttr shapeAttr, DType dtype,
                                 ArrayRef<int64_t> shape,
                                 std::optional<Attribute> metadata) {
  if (auto unkAttr = llvm::dyn_cast<KGEN::UnknownAttr>(shapeAttr)) {
    if (Type ty = unkAttr.getType(); !llvm::isa<MO::ShapeType>(ty)) {
      return emitError() << "unknown shape attribute must have ShapeType, got "
                         << ty;
    }
    if (!shape.empty()) {
      return emitError()
             << "hidden shape not empty when shape attribute is unknown";
    }
    return mlir::success();
  }
  if (auto declRef = llvm::dyn_cast<KGEN::ParamDeclRefAttr>(shapeAttr)) {
    if (Type ty = declRef.getType(); !llvm::isa<MO::ShapeType>(ty)) {
      return emitError()
             << "shape parameter reference must have ShapeType, got " << ty;
    }
    if (!shape.empty()) {
      return emitError() << "hidden shape not empty when shape attribute is "
                            "parameter reference";
    }
    return success();
  }

  auto attr = llvm::dyn_cast<MO::ShapeAttr>(shapeAttr);
  if (!attr)
    return emitError() << "invalid shape attribute " << shapeAttr;
  if (!shape.equals(attr.getIntDims()))
    return emitError() << "hidden shape does not match shape attribute";
  return success();
}

// TODO : These are shamelessly copied out of MGP. Need to align
ErrorOr<DType> static fromMLIRType(Type mlirType) {
  if (auto unkType = mlir::dyn_cast<NoneType>(mlirType))
    return DType(); // invalid denotes unknown.
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
  return Error(Twine("unable to represent MLIR Type '") + asString(mlirType) +
               "' as an M::DType type.");
}

ErrorOr<Type> static toMLIRType(MLIRContext *context, DType dType) {
  if (!dType.isValid())
    return NoneType::get(context); // None denotes unknown.
  if (dType.isInt()) {
    if (Type type = getEquivalentIntegerType(context, dType))
      return type;
  }
  if (dType.isFloat()) {
    if (Type type = getEquivalentFloatType(context, dType))
      return type;
  }
  return Error(Twine("unable to represent DType '") + dType.getAsString() +
               "' as an MLIR type.");
}

TensorType TensorType::get(TypedAttr shapeAttr, DType dtype,
                           std::optional<Attribute> metadata) {
  SmallVector<int64_t> hiddenShape{};
  if (auto attr = llvm::dyn_cast<MO::ShapeAttr>(shapeAttr))
    hiddenShape = attr.getIntDims();
  return get(shapeAttr.getContext(), shapeAttr, dtype, hiddenShape, metadata);
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
  auto newShape = MO::ShapeAttr::get(shape, MO::ShapeType::get(ctx));
  return get(newShape, elementType, metadata);
}

/// Returns clone with given element type, which must be representable as a
/// dtype. Required by ShapedTypeInference
ShapedType TensorType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                 Type dtype,
                                 std::optional<Attribute> metadata) const {
  auto shapeTy = MO::ShapeType::get(getContext());

  TypedAttr shapeRef;
  if (shape.has_value())
    shapeRef = MO::ShapeAttr::get(shape.value(), shapeTy);
  else
    shapeRef = KGEN::UnknownAttr::get(shapeTy);

  ErrorOr<DType> errOrDType = fromMLIRType(dtype);
  assert(!errOrDType.isError() &&
         "element type is not representable as an M::DType");
  return TensorType::get(shapeRef, *errOrDType, metadata);
}

/// Returns clone with a new shape attribute.
TensorType TensorType::cloneWith(TypedAttr shapeAttr) const {
  return TensorType::get(shapeAttr.getContext(), shapeAttr, getDtype(),
                         getShape(), getMetadata());
}

/// Does the Tensor have a static Rank?  Required by ShapedTypeInferece
bool TensorType::hasRank() const {
  // Rank is only known if we have a concrete attribute
  if (llvm::isa<KGEN::UnknownAttr, KGEN::ParamDeclRefAttr>(getShapeAttr()))
    return false;
  return true;
}

// Get the Rank of the Tensor.  Required by ShapedTypeInference
size_t TensorType::getRank() const {
  assert(hasRank() && "cannot get rank if shape is not known");
  return getTypedShapeAttr().getRank();
}

/// Returns if this type has a static shape, i.e. if the type is ranked and
/// all dimensions have known size (>= 0).
bool TensorType::hasStaticShape() const {
  return hasRank() && getTypedShapeAttr().isStatic();
}

// Get the mlir element type of the Tensor.  Required by ShapedTypeInference
Type TensorType::getElementType() const {
  ErrorOr<Type> errOrType = toMLIRType(getContext(), getDtype());
  assert(!errOrType.isError() &&
         "element type is not representable as an mlir Type");
  return *errOrType;
}

MO::ShapeAttr TensorType::getTypedShapeAttr() const {
  return llvm::cast<MO::ShapeAttr>(getShapeAttr());
}

LogicalResult ShapeType::printValue(AsmPrinter &p, TypedAttr value) const {
  if (!mlir::isa<KGEN::ParamDeclRefAttr, MO::ShapeAttr>(value))
    return failure();
  printShapeParam(p, value);
  return success();
}

OptionalParseResult ShapeType::parseValue(AsmParser &p,
                                          TypedAttr &value) const {
  FailureOr<TypedAttr> shapeAttr;
  auto parseResult = parseShapeParam(p, shapeAttr);
  if (succeeded(parseResult))
    value = shapeAttr.value();
  return parseResult;
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
