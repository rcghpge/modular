//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"

using namespace M;
using namespace MOSH;

//===----------------------------------------------------------------------===//
// MOSHDialect
//===----------------------------------------------------------------------===//

void MOSHDialect::initialize() {
  registerAttributes();
  registerTypes();
  registerOperations();
}

Operation *MOSHDialect::materializeConstant(OpBuilder &b, Attribute value,
                                            Type type, Location loc) {
  // Materialize index constants.
  auto indexDialect =
      getContext()->getOrLoadDialect<mlir::index::IndexDialect>();
  if (Operation *res = indexDialect->materializeConstant(b, value, type, loc))
    return res;

  // Materialize symbolic shapes.
  if (isa<KGEN::ParamDeclRefAttr, ShapeAttr, KGEN::ParamOperatorAttr>(value))
    return b.create<KGEN::ParamConstantOp>(loc, type, cast<TypedAttr>(value));

  return nullptr;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOSHDialect/MOSHDialect.cpp.inc"
