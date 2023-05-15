//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// KGENDialect attribute support
//===----------------------------------------------------------------------===//

void MODialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "GenericML/GraphCompiler/MODialect/MOAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ChainAttr
//===----------------------------------------------------------------------===//

bool ChainAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// ShapeAttr
//===----------------------------------------------------------------------===//

int64_t M::MO::findDeclIdx(ShapeAttr shape, KGEN::ParamDeclAttr decl) {
  StringAttr declName = decl.getName();
  for (auto [idx, dim] : llvm::enumerate(shape.getValues())) {
    if (auto dimRef = dyn_cast<KGEN::ParamDeclRefAttr>(dim))
      if (dimRef.getName() == declName)
        return idx;
  }
  return -1;
}

ParseResult M::MO::parseShapeValue(AsmParser &p,
                                   SmallVector<TypedAttr> &values) {
  values = SmallVector<TypedAttr>();
  IndexType idxTy = p.getBuilder().getIndexType();
  return p.parseCommaSeparatedList(
      [&] { return KGEN::parseParamValue(p, values.emplace_back(), idxTy); });
}

ParseResult M::MO::parseShapeValue(AsmParser &p, SmallVector<TypedAttr> &values,
                                   MO::ShapeType type) {
  return parseShapeValue(p, values);
}

void M::MO::printShapeValue(AsmPrinter &p, ArrayRef<TypedAttr> values) {
  llvm::interleaveComma(
      values, p, [&](TypedAttr value) { KGEN::printParamValue(p, value); });
}

void M::MO::printShapeValue(AsmPrinter &p, ArrayRef<TypedAttr> values,
                            MO::ShapeType type) {
  printShapeValue(p, values);
}

MO::ShapeAttr MO::ShapeAttr::get(ArrayRef<int64_t> values, MO::ShapeType type) {
  auto idxTy = IndexType::get(type.getContext());
  SmallVector<TypedAttr> newValues;
  for (auto v : values) {
    if (mlir::ShapedType::isDynamic(v))
      newValues.emplace_back(KGEN::UnknownAttr::get(idxTy));
    else
      newValues.emplace_back(IntegerAttr::get(idxTy, v));
  }
  return get(type.getContext(), newValues, type);
}

MO::ShapeAttr MO::ShapeAttr::get(IntArrayElementsAttr intDims,
                                 MO::ShapeType type) {
  SmallVector<int64_t> rawDims = map_to_vector(
      intDims.getValues(), [](const APInt &dim) { return dim.getSExtValue(); });
  return get(rawDims, type);
}

bool MO::ShapeAttr::isConstant() const {
  // A shape is constant if the dimensions are concrete integers.
  return llvm::all_of(
      getValues(), [](TypedAttr attr) { return llvm::isa<IntegerAttr>(attr); });
}

bool MO::ShapeAttr::hasUnknownDims() const {
  return llvm::any_of(getValues(), [](TypedAttr attr) {
    return llvm::isa<KGEN::UnknownAttr>(attr);
  });
};

LogicalResult
MO::ShapeAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<TypedAttr> values, MO::ShapeType type) {
  bool hasEncounteredNegOne = false;
  for (auto [idx, value] : llvm::enumerate(values)) {
    if (!value.getType().isIndex()) {
      return emitError() << "expected all list elements to have type "
                         << IndexType::get(type.getContext()) << " but got "
                         << value.getType() << " at position " << idx;
    }
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(value)) {
      auto intVal = intAttr.getInt();
      if (intVal == -1) {
        if (hasEncounteredNegOne)
          return emitError() << "at most one -1 dimension is allowed";
        hasEncounteredNegOne = true;
      } else if (intVal < 0 && !mlir::ShapedType::isDynamic(intVal)) {
        return emitError() << "invalid shape dimension: " << intVal
                           << " at position " << idx;
      }
    }
  }
  return success();
}

/// Convert a symbolic dimension attribute to an integer value.
static int64_t symbolicDimToInt(TypedAttr dim) {
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(dim))
    return intAttr.getInt();
  return mlir::ShapedType::kDynamic;
}

int64_t MO::ShapeAttr::getIntDim(size_t n) const {
  return symbolicDimToInt(getValues()[n]);
}

SmallVector<int64_t> MO::ShapeAttr::getIntDims() const {
  return map_to_vector(getValues(), symbolicDimToInt);
}

TypedAttr MO::ShapeAttr::getSymbolicNumElements() const {
  if (getRank() == 0)
    return OpBuilder(getContext()).getIndexAttr(1);
  return KGEN::ParamOperatorAttr::get(KGEN::POC::MulNuw, getValues());
}

ErrorOr<ShapeAttr> MO::ShapeAttr::broadcastShapes(ShapeAttr xSh,
                                                  ShapeAttr ySh) {
  // This largely should follow the semantics described here:
  // https://numpy.org/doc/stable/user/basics.broadcasting.html
  // TODO (#7331): Merge with Support/ML/BroadcastShape.

  if (xSh == ySh) // Short circuit.
    return ySh;

  auto idxTy = IndexType::get(xSh.getContext());
  auto intDim1 = IntegerAttr::get(idxTy, 1ll);

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

    if (xShDim->isa<KGEN::ParamDeclRefAttr>()) {
      if (yShDim->isa<KGEN::UnknownAttr>()) {
        return Error(Twine("xShDim and yShDim are incompatible"));
      }
    }

    if (yShDim->isa<KGEN::ParamDeclRefAttr>()) {
      if (xShDim->isa<KGEN::UnknownAttr>()) {
        return Error(Twine("xShDim and yShDim are incompatible"));
      }
    }

    if (xShDim->isa<KGEN::UnknownAttr>()) {
      result.push_back(yShDimVal);
      continue;
    }

    if (yShDim->isa<KGEN::UnknownAttr>()) {
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
  return MO::ShapeAttr::get(SmallVector<TypedAttr>(llvm::reverse(result)),
                            xSh.getType());
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GenericML/GraphCompiler/MODialect/MOAttrs.cpp.inc"
