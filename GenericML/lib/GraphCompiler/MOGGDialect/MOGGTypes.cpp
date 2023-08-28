//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MOGG;

// Printer and parser for shape attributes.
static ParseResult parseShapeParam(AsmParser &p, TypedAttr &shapeAttr) {
  auto shapeTy = p.getBuilder().getType<MOSH::ShapeType>();
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

//===----------------------------------------------------------------------===//
// MOGGDialect type support
//===----------------------------------------------------------------------===//

void MOGGDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "GenericML/GraphCompiler/MOGGDialect/MOGGTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "GenericML/GraphCompiler/MOGGDialect/MOGGTypes.cpp.inc"
