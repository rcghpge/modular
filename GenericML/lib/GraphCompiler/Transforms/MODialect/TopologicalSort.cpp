//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Analysis/MODialect/ParametricTopologyAnalysis.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"

namespace M::MO {

#define GEN_PASS_DEF_TOPOLOGICALSORT
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

struct TopologicalSort : public impl::TopologicalSortBase<TopologicalSort> {

  TopologicalSort() : TopologicalSortBase() {}

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    ParametricTopologyAnalysis analysis(moduleOp);
    analysis.topologicallySortAllOpsWithin(&moduleOp.getRegion());
  }
};

} // namespace M::MO
