//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"

namespace M::MO {

#define GEN_PASS_DEF_TOPOLOGICALSORT
#include "GenericML/GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

struct TopologicalSort : public impl::TopologicalSortBase<TopologicalSort> {

  TopologicalSort() : TopologicalSortBase() {}

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    ParametricTopologyAnalysis analysis(moduleOp);
    analysis.topologicallySortAllOpsWithin(&moduleOp.getRegion());
  }
};

} // namespace M::MO
