//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MOSH dialect types.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MOSH;

//===----------------------------------------------------------------------===//
// ShapeType
//===----------------------------------------------------------------------===//

LogicalResult ShapeType::printValue(AsmPrinter &p, TypedAttr value) const {
  if (!llvm::isa<KGEN::ParamDeclRefAttr, ShapeAttr>(value))
    return failure();
  printShapeTypedAttr(p, value);
  return success();
}

OptionalParseResult ShapeType::parseValue(AsmParser &p,
                                          TypedAttr &value) const {
  return parseShapeAttr(p, value);
}

//===----------------------------------------------------------------------===//
// MOSHDialect type support
//===----------------------------------------------------------------------===//

void MOSHDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "GraphCompiler/MOSHDialect/MOSHTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "GraphCompiler/MOSHDialect/MOSHTypes.cpp.inc"
