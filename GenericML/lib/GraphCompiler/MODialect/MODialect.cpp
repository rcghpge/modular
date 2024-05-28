//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GenericML/Interfaces/GraphInterfaces.h"
#include "GraphRT/MIPDialect/MIPDialect.h"
#include "Support/AssertStream.h"
#include "mlir/Parser/Parser.h"

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// MOGraphVisualizationInterface
//===----------------------------------------------------------------------===//
namespace {
struct MOGraphVisualizationInterface : public GraphVisualizationInterface {

  using GraphVisualizationInterface::GraphVisualizationInterface;

  VisualizationInfo populateVisualizationInfo(Operation *op) override {
    return VisualizationInfo{op->getName().getStringRef(),
                             /* isModularOp = */ true};
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// MODialect
//===----------------------------------------------------------------------===//

void MODialect::initialize() {
  registerAttributes();
  registerTypes();
  registerOperations();

  addInterfaces<MOGraphVisualizationInterface>();
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

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MODialect.cpp.inc"
