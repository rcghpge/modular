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

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GenericML/GraphCompiler/MODialect/MOAttrs.cpp.inc"
