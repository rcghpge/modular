//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MODialect/MOTypes.h"
#include "GenericML/MODialect/MODialect.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
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

template <typename AttributeT>
struct FieldParser<
    Optional<AttributeT>,
    std::enable_if_t<std::is_base_of<Attribute, AttributeT>::value,
                     Optional<AttributeT>>> {
  static FailureOr<Optional<AttributeT>> parse(AsmParser &parser) {
    AttributeT attr;
    OptionalParseResult result = parser.parseOptionalAttribute(attr);
    if (result.has_value()) {
      if (succeeded(*result))
        return {Optional<AttributeT>(attr)};
      return failure();
    }
    return {std::nullopt};
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
                                 ArrayRef<int64_t> shape, DType dtype,
                                 Optional<Attribute> metadata) {
  for (auto dim : shape) {
    if (dim <= 0 && dim != ShapedType::kDynamic && dim != kDynamicRank) {
      return emitError()
             << "shape dimensions must be strictly positive or the "
                "distinguished '?' unknown dimension or '*' unknown rank.";
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

/// Returns clone with given element type, which must be representable as a
/// dtype. Required by ShapedTypeInference
ShapedType TensorType::cloneWith(Optional<ArrayRef<int64_t>> shape, Type dtype,
                                 Optional<Attribute> metadata) const {
  ArrayRef<int64_t> newShape =
      shape.has_value() ? ArrayRef<int64_t>(shape.value()) : getShape();

  ErrorOr<DType> errOrDType = fromMLIRType(dtype);
  assert(!errOrDType.isError() &&
         "element type is not representable as an M::DType");
  return TensorType::get(getContext(), newShape, *errOrDType, metadata);
}

/// Does the Tensor have a static Rank?  Required by ShapedTypeInferece
bool TensorType::hasRank() const {
  auto shape = getShape();
  return std::find(shape.begin(), shape.end(), kDynamicRank) != shape.end();
}

// Get the Rank of the Tensor.  Required by ShapedTypeInference
size_t TensorType::getRank() const { return getShape().size(); }

// Get the mlir element type of the Tensor.  Required by ShapedTypeInference
Type TensorType::getElementType() const {
  ErrorOr<Type> errOrType = toMLIRType(getContext(), getDtype());
  assert(!errOrType.isError() &&
         "element type is not representable as an mlir Type");
  return *errOrType;
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
