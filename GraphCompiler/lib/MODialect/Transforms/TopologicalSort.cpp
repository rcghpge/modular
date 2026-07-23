//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHTypes.h"

namespace M::MO {

#define GEN_PASS_DEF_TOPOLOGICALSORT
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

struct TopologicalSort : public impl::TopologicalSortBase<TopologicalSort> {

  TopologicalSort() : TopologicalSortBase() {}

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    ParametricTopologyAnalysis analysis(moduleOp);
    analysis.topologicallySortAllOpsWithin(&moduleOp.getRegion());
  }
};

} // namespace M::MO
