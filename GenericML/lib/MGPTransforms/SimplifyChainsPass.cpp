//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "GenericML/MGPDialect/MGPOps.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

#include "./AsyncDataflowDependencies.h"
#include "./Utils.h"

using namespace mlir;

namespace M::MGP {

// Generate support code for SimplifyChainsPass
#define GEN_PASS_DEF_SIMPLIFYCHAINSPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// SimplifyChainsPass
//===----------------------------------------------------------------------===//

class SimplifyChainsPass
    : public impl::SimplifyChainsPassBase<SimplifyChainsPass> {
  using SimplifyChainsPassBase::SimplifyChainsPassBase;

  void runOnOperation() override {
    ModelOp modelOp = getOperation();
    // Construct the dependencies from the model. The transforms we
    // do below will preserve these dependencies since we never introduce
    // new bound values or add new dependencies.
    IndexSet<Value> index = createValueIndexSet(modelOp);
    AsyncDataflowDependencies dependencies(&index);

    modelOp.walk([&](GRT::ChainMergeOp mergeOp) {
      ValueSetVector valuesSet;
      accumulatePrimitiveChains(valuesSet, mergeOp.getResult());
      if (valuesSet.empty())
        return;
      SmallVector<Value> values = valuesSet.takeVector();
      dependencies.simplify(values);
      if (values.size() == 1) {
        mergeOp.getResult().replaceAllUsesWith(values.front());
        mergeOp.erase();
      } else {
        // Ok for values to now be empty.
        mergeOp->setOperands(values);
      }
    });
  }
};

} // namespace

} // namespace M::MGP
