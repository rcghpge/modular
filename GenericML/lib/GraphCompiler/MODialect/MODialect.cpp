//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GraphRT/MIPDialect/MIPDialect.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// MODialect
//===----------------------------------------------------------------------===//

void MODialect::initialize() {
  registerAttributes();
  registerTypes();
  registerOperations();
}

Operation *MODialect::materializeConstant(OpBuilder &b, Attribute value,
                                          Type type, Location loc) {
  // Materialize index constants in shape op outputs
  auto indexDialect =
      getContext()->getOrLoadDialect<mlir::index::IndexDialect>();
  if (Operation *res = indexDialect->materializeConstant(b, value, type, loc))
    return res;

  // Materialize symbolic shapes in shape op outputs
  if (isa<KGEN::ParamDeclRefAttr, ShapeAttr, KGEN::ParamOperatorAttr>(value))
    return b.create<KGEN::ParamConstantOp>(loc, type, cast<TypedAttr>(value));

  if (auto elements = dyn_cast_or_null<ElementsAttr>(value))
    return b.create<ConstantOp>(loc, type, value);

  return nullptr;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MODialect.cpp.inc"
