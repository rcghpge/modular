//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GraphRT/MIPDialect/MIPDialect.h"
#include <mlir/Parser/Parser.h>

// Directly embed the mojo file rather than loading dynamically.
#include MOSH_MOJO_BINDING_H

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

ModuleOp MODialect::getShapeFuncModule() {
  if (!shapeFuncModule) {
    StringRef source{reinterpret_cast<const char *>(MojoMoshBindings::data),
                     MojoMoshBindings::size};
    shapeFuncModule = mlir::parseSourceString<ModuleOp>(source, getContext());
    assert(shapeFuncModule && "could not load the embedded shape functions");
  }
  return *shapeFuncModule;
}

/// TODO: remove this when #14744 is fixed
MODialect::~MODialect() {
  if (shapeFuncModule)
    shapeFuncModule.release();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MODialect.cpp.inc"
