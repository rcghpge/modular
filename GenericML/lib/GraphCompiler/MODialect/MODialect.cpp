//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GraphRT/MIPDialect/MIPDialect.h"

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
  // Materialize shape constants and symbolic shapes.
  auto moshDialect = getContext()->getOrLoadDialect<MOSH::MOSHDialect>();
  if (Operation *res = moshDialect->materializeConstant(b, value, type, loc))
    return res;

  if (auto elements = dyn_cast_or_null<ElementsAttr>(value))
    return b.create<ConstantOp>(loc, type, elements);

  return nullptr;
}

ModuleOp MODialect::getShapeFuncModule() { return shapeFuncModule; }

void MODialect::setShapeFuncModule(ModuleOp module) {
  shapeFuncModule = module;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MODialect.cpp.inc"
