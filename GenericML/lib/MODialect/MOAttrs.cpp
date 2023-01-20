//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MODialect/MODialect.h"
#include "GenericML/MODialect/MOTypes.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/LLVMCompilerForwardDecls.h"
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
#include "GenericML/MODialect/MOAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ShapeAttr
//===----------------------------------------------------------------------===//

ParseResult M::MO::parseShapeValue(AsmParser &p,
                                   FailureOr<SmallVector<TypedAttr>> &values,
                                   MO::ShapeType type) {
  values.emplace();
  IndexType idxType = p.getBuilder().getIndexType();
  return p.parseCommaSeparatedList([&] {
    return KGEN::parseParamValue(p, values->emplace_back(), idxType);
  });
}

void M::MO::printShapeValue(AsmPrinter &p, ArrayRef<TypedAttr> values,
                            MO::ShapeType type) {
  llvm::interleaveComma(
      values, p, [&](TypedAttr value) { KGEN::printParamValue(p, value); });
}

MO::ShapeAttr MO::ShapeAttr::get(ArrayRef<int64_t> values, MO::ShapeType type) {
  auto t = IndexType::get(type.getContext());
  SmallVector<TypedAttr> newValues;
  for (auto v : values) {
    if (mlir::ShapedType::isDynamic(v))
      newValues.emplace_back(KGEN::UnknownAttr::get(t));
    else
      newValues.emplace_back(IntegerAttr::get(t, v));
  }
  return get(type.getContext(), newValues, type);
}

/// A shape constant is simple if its type and elements are fully-resolved.
bool MO::ShapeAttr::isConstant() const {
  return llvm::all_of(getValues(), [](TypedAttr attr) {
    // Only IntegerAttr elements can be constant.
    return llvm::isa<IntegerAttr>(attr);
  });
}

LogicalResult
MO::ShapeAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<TypedAttr> values, MO::ShapeType type) {

  auto indexTy = IndexType::get(type.getContext());
  for (TypedAttr value : values) {
    if (value.getType() != indexTy) {
      return emitError() << "expected all list elements to have type "
                         << indexTy;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GenericML/MODialect/MOAttrs.cpp.inc"
