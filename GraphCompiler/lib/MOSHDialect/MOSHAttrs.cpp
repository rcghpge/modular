//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MOSH dialect attributes.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/AssertStream.h"
#include "Support/MDialect/ParserUtils.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MOSH;

//===----------------------------------------------------------------------===//
// MOSHDialect attribute support
//===----------------------------------------------------------------------===//

void MOSHDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "GraphCompiler/MOSHDialect/MOSHAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ShapeAttr
//===----------------------------------------------------------------------===//

OptionalParseResult M::MOSH::parseShapeAttr(AsmParser &p,
                                            TypedAttr &shapeAttr) {
  auto shapeTy = p.getBuilder().getType<ShapeType>();

  if (succeeded(p.parseOptionalLSquare())) {
    SmallVector<TypedAttr> values;
    if (failed(p.parseOptionalRSquare())) {
      if (parseShapeValue(p, values) || p.parseRSquare())
        return failure();
    }

    shapeAttr = p.getChecked<ShapeAttr>(values, shapeTy);
    return failure(!shapeAttr);
  }

  return std::nullopt;
}

void M::MOSH::printShapeTypedAttr(AsmPrinter &p, TypedAttr shapeAttr) {
  assert(isa<ShapeType>(shapeAttr.getType()));
  if (auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(shapeAttr)) {
    KGEN::printParamName(p, declRef.getName());
  } else if (auto declRef = dyn_cast<KGEN::ParamIndexRefAttr>(shapeAttr)) {
    KGEN::printParamValue(p, declRef, declRef.getType());
  } else {
    p << "[";
    printShapeValue(p, cast<ShapeAttr>(shapeAttr).getValues());
    p << "]";
  }
}

std::string M::MOSH::shapeTypedAttrToStr(TypedAttr attr) {
  std::string str;
  llvm::raw_string_ostream os(str);
  StreamAsmPrinter printer(os);
  MOSH::printShapeTypedAttr(printer, attr);
  return str;
}

int64_t M::MOSH::findDeclIdx(ShapeAttr shape, KGEN::ParamDeclAttr decl) {
  StringAttr declName = decl.getName();
  for (auto [idx, dim] : llvm::enumerate(shape.getValues())) {
    if (auto dimRef = dyn_cast<KGEN::ParamDeclRefAttr>(dim))
      if (dimRef.getName() == declName)
        return idx;
  }
  return -1;
}

static ParseResult parseColonTypeOrDimension(AsmParser &parser, Type &type) {
  auto dimType = ShapeAttr::getDimType(parser.getContext());
  return KGEN::parseColonTypeOrDefault(parser, type, dimType);
}

static ParseResult parseShapeValueElement(AsmParser &p, TypedAttr &value) {
  Type type;
  if (parseColonTypeOrDimension(p, type) ||
      KGEN::parseParamValue(p, value, type))
    return failure();

  return success();
}

ParseResult M::MOSH::parseShapeValue(AsmParser &p,
                                     SmallVectorImpl<TypedAttr> &values) {
  values.clear();

  if (p.parseCommaSeparatedList([&]() -> ParseResult {
        return parseShapeValueElement(p, values.emplace_back());
      }))
    return failure();

  return success();
}

ParseResult M::MOSH::parseShapeValue(AsmParser &p,
                                     SmallVectorImpl<TypedAttr> &values,
                                     ShapeType type) {
  return parseShapeValue(p, values);
}

/// print `:<type> ` or elide it entirely if type is the dim type.
static void printColonTypeOrDimensionPrefix(AsmPrinter &p, Type type) {
  if (type == ShapeAttr::getDimType(type.getContext()))
    return;
  p << ':';
  KGEN::printKGENType(p, type);
  p << ' ';
}

void M::MOSH::printDimensionValue(AsmPrinter &p, TypedAttr value) {
  printColonTypeOrDimensionPrefix(p, value.getType());
  KGEN::printParamValue(p, value);
}

void M::MOSH::printShapeValue(AsmPrinter &p, ArrayRef<TypedAttr> values) {
  llvm::interleaveComma(
      values, p, [&](TypedAttr value) { printDimensionValue(p, value); });
}

void M::MOSH::printShapeValue(AsmPrinter &p, ArrayRef<TypedAttr> values,
                              ShapeType type) {
  printShapeValue(p, values);
}

KGEN::SIMDType ShapeAttr::getDimType(MLIRContext *ctx) {
  return KGEN::SIMDType::get(ctx, 1, KGEN::KGENDType::si64);
}

KGEN::SIMDAttr ShapeAttr::getDim(MLIRContext *ctx, int64_t value) {
  return KGEN::SIMDAttr::get(value, getDimType(ctx));
}

/// Is the given type a legal dimension (scalar `si64` SIMD lane)?
bool ShapeAttr::isDimType(Type type) {
  if (type == getDimType(type.getContext()))
    return true;

  if (auto simdType = dyn_cast<KGEN::SIMDType>(type))
    return simdType.getResolvedSize() == 1 &&
           simdType.getResolvedDType()
               .value_or(KGEN::KGENDType::invalid)
               .isIntLike(); // it seems that we do allow bool/index.

  return false;
}

std::optional<int64_t> ShapeAttr::getStaticDim(Attribute attr) {
  auto simdAttr = dyn_cast_or_null<KGEN::SIMDAttr>(attr);
  if (!simdAttr || !isDim(simdAttr))
    return {};

  // must be a scalar SIMD if reach this point.
  return simdAttr.getValues().front().getIntVal().getExtValue();
}

SmallVector<TypedAttr> dimAttrs(MLIRContext *context,
                                ArrayRef<int64_t> values) {
  SmallVector<TypedAttr> attrs;
  attrs.reserve(values.size());

  for (auto v : values) {
    if (mlir::ShapedType::isDynamic(v)) {
      ASSERT_STREAM(false, "ShapeAttr does not support dynamic dimensions");
    } else {
      attrs.push_back(ShapeAttr::getDim(context, v));
    }
  }
  return attrs;
}

ShapeAttr ShapeAttr::get(ArrayRef<int64_t> values, ShapeType type) {
  auto ctx = type.getContext();
  return get(ctx, dimAttrs(ctx, values), type);
}

ShapeAttr ShapeAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                ArrayRef<int64_t> values, ShapeType type) {
  auto ctx = type.getContext();
  auto dims = dimAttrs(ctx, values);
  return ShapeAttr::getChecked(emitError, dims, type);
}

ShapeAttr ShapeAttr::get(IntArrayElementsAttr intDims, ShapeType type) {
  SmallVector<int64_t> rawDims = llvm::map_to_vector(
      intDims.getValues(), [](const APInt &dim) { return dim.getSExtValue(); });
  return get(rawDims, type);
}

ShapeAttr ShapeAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                IntArrayElementsAttr intDims, ShapeType type) {
  auto ctx = type.getContext();
  auto dims = dimAttrs(ctx, intDims.asArrayRef<int64_t>());
  return ShapeAttr::getChecked(emitError, dims, type);
}

bool ShapeAttr::isConstant() const {
  return llvm::all_of(getValues(), [](TypedAttr attr) {
    return getStaticDim(attr).has_value();
  });
}

LogicalResult ShapeAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                ArrayRef<TypedAttr> values, ShapeType type) {
  for (auto [idx, value] : llvm::enumerate(values)) {
    // Must be a scalar SIMD with int-like dtype.
    if (!ShapeAttr::isDimType(value.getType())) {
      return emitError() << "expected all list elements to have "
                            "scalar<integer> type "
                         << value.getType() << " at position " << idx;
    }
  }

  auto elementTypes =
      llvm::map_range(values, [](TypedAttr value) { return value.getType(); });

  // Ensure all elements of the ShapeAttr have the same type.
  if (!llvm::all_equal(elementTypes))
    return emitError() << "expected all elements to have the same type";

  return success();
}

/// Convert a symbolic dimension attribute to an integer value.
static int64_t symbolicDimToInt(TypedAttr dim) {
  std::optional<int64_t> v = ShapeAttr::getStaticDim(dim);
  return v.has_value() ? *v : mlir::ShapedType::kDynamic;
}

int64_t ShapeAttr::getIntDim(size_t n) const {
  return symbolicDimToInt(getValues()[n]);
}

SmallVector<int64_t> ShapeAttr::getIntDims() const {
  return llvm::map_to_vector(getValues(), symbolicDimToInt);
}

TypedAttr ShapeAttr::getSymbolicNumElements() const {
  if (getRank() == 0)
    return ShapeAttr::getDim(getContext(), 1);
  return KGEN::ParamOperatorAttr::get(KGEN::POC::MulNoWrap, getValues());
}

ErrorOr<ShapeAttr> ShapeAttr::broadcastShapes(ShapeAttr xSh, ShapeAttr ySh) {
  // This largely should follow the semantics described here:
  // https://numpy.org/doc/stable/user/basics.broadcasting.html
  // TODO (#7331): Merge with Support/ML/BroadcastShape.

  if (xSh == ySh) // Short circuit.
    return ySh;

  auto intDim1 = getDim(xSh.getContext(), 1ll);

  SmallVector<TypedAttr> result;
  ArrayRef<TypedAttr> xShDims = xSh.getValues();
  ArrayRef<TypedAttr> yShDims = ySh.getValues();
  for (auto [xShDim, yShDim] :
       llvm::zip_longest(llvm::reverse(xShDims), llvm::reverse(yShDims))) {
    if (!yShDim.has_value()) {
      result.push_back(xShDim.value());
      continue;
    }

    if (!xShDim.has_value()) {
      result.push_back(yShDim.value());
      continue;
    }

    TypedAttr xShDimVal = xShDim.value();
    TypedAttr yShDimVal = yShDim.value();

    if (xShDimVal == yShDimVal) {
      result.push_back(xShDimVal);
      continue;
    }

    if (xShDimVal == intDim1) {
      result.push_back(yShDimVal);
      continue;
    }

    if (yShDimVal == intDim1) {
      result.push_back(xShDimVal);
      continue;
    }

    // Not 1, not dynamic. Then the other shape necessarily matches this.
    if (xShDimVal != yShDimVal) {
      return Error(Twine(
          "upstream dialect should have rejected non-broadcastable inputs"));
    }
    result.push_back(xShDimVal);
  }
  return ShapeAttr::get(SmallVector<TypedAttr>(llvm::reverse(result)),
                        xSh.getType());
}

TypedAttr ShapeAttr::getCommonBroadcastShape(ArrayRef<TypedAttr> shapeParams) {
  // We collect all the statically ranked shapes first.
  SmallVector<ShapeAttr> staticRankShapes;
  for (TypedAttr p : shapeParams) {
    if (auto shape = llvm::dyn_cast<ShapeAttr>(p)) {
      staticRankShapes.emplace_back(shape);
    }
  }

  // If all the shapes are dynamically ranked, we can only determine a common
  // shape if they are all the same.
  if (staticRankShapes.empty()) {
    TypedAttr res = shapeParams.front();
    auto notSameAsFirst = [res](TypedAttr other) { return res != other; };
    if (llvm::any_of(shapeParams.drop_front(), notSameAsFirst))
      return {};
    return res;
  }

  // If we have both static and dynamic ranks, we can't do anything.
  if (staticRankShapes.size() != shapeParams.size())
    return {};

  // At this point, we are dealing with all static ranks. We will go through all
  // the shapes and successively reduce the common shape. If the rank of the
  // common dimension drops below the maximum of the ranks, we terminate early.
  ShapeAttr commonShape = staticRankShapes.pop_back_val();
  while (!staticRankShapes.empty()) {
    ShapeAttr nextShape = staticRankShapes.pop_back_val();
    ErrorOr<ShapeAttr> maybeCommonShape =
        ShapeAttr::broadcastShapes(commonShape, nextShape);
    if (maybeCommonShape.isError())
      return {};
    commonShape = maybeCommonShape.takeValue();
  }

  return commonShape;
}

ErrorOr<SmallVector<int64_t>>
ShapeAttr::getBroadcastedAxes(ShapeAttr beforeShape, ShapeAttr bcastedShape) {
  ArrayRef<TypedAttr> beforeDims = beforeShape.getValues();
  ArrayRef<TypedAttr> bcastedDims = bcastedShape.getValues();

  ASSERT_STREAM(
      beforeDims.size() <= bcastedDims.size(),
      << "Broadcasting can only increase rank but it is trying to decrease "
         "rank from "
      << beforeDims.size() << " to " << bcastedDims.size());

  // we will initially add in reverse sorted order since it's easier
  // as bcasting rules follow from the innermost dimensions
  SmallVector<int64_t> answer;
  for (size_t iter = 0; iter < beforeDims.size(); ++iter) {
    int beforeI = beforeDims.size() - iter - 1;
    int bcastedI = bcastedDims.size() - iter - 1;

    const TypedAttr &beforeDim = beforeDims[beforeI];
    const TypedAttr &bcastedDim = bcastedDims[bcastedI];

    // We cannot necessarily prove things are broadcast in all scenarios.
    // For example A --> B. A and B are runtime parameters so in some cases
    // the axis will be a result of broadcasting and other times they will be
    // not.
    //
    // Therefore, we can only return a correct answer if the only differing
    // dimensions are integral dimensions going from 1 --> not 1.
    std::optional<int64_t> beforeDimInt = ShapeAttr::getStaticDim(beforeDim);
    std::optional<int64_t> bcastedDimInt = ShapeAttr::getStaticDim(bcastedDim);
    if (beforeDimInt && bcastedDimInt && *beforeDimInt == 1 &&
        *bcastedDimInt != 1) {
      answer.push_back(bcastedI);
    } else if (beforeDim != bcastedDim) {
      return Error("Cannot statically prove which dimension broadcast.");
    }
  }

  // add all the newly introduced dims too
  size_t extraDims = bcastedDims.size() - beforeDims.size();
  for (size_t iter = 0; iter < extraDims; ++iter) {
    // remember to return in reverse order
    answer.push_back(extraDims - 1 - iter);
  }

  std::reverse(answer.begin(), answer.end());
  return answer;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GraphCompiler/MOSHDialect/MOSHAttrs.cpp.inc"
