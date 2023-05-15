//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MOSH dialect types.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MOSH;

//===----------------------------------------------------------------------===//
// ShapeType
//===----------------------------------------------------------------------===//

LogicalResult ShapeType::printValue(AsmPrinter &p, TypedAttr value) const {
  if (!mlir::isa<KGEN::ParamDeclRefAttr, ShapeAttr>(value))
    return failure();
  printShapeTypedAttr(p, value);
  return success();
}

OptionalParseResult ShapeType::parseValue(AsmParser &p,
                                          TypedAttr &value) const {
  return parseShapeAttrOrUnknown(p, value);
}

//===----------------------------------------------------------------------===//
// MOSHDialect type support
//===----------------------------------------------------------------------===//

void MOSHDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.cpp.inc"
