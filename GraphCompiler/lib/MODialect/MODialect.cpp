//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "Support/AssertStream.h"
#include "mlir/Parser/Parser.h"

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
    return ConstantOp::create(b, loc, type, elements);

  return nullptr;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MODialect.cpp.inc"
