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

  /// Returns the 'root' block containing op, which is either the 'init' or
  /// 'execute' block of the model.
  Block *getRootBlock(ModelOp modelOp, Operation *op) {
    while (op->getBlock() != modelOp.getInitBlock() &&
           op->getBlock() != modelOp.getExecuteBlock())
      op = op->getParentOp();
    return op->getBlock();
  }

  /// Returns a block argument in block which is equivalent to value, or null
  /// if no such argument exists.
  Value getEquivArgument(const AsyncDataflowDependencies &dependencies,
                         Block *block, Value value) {
    for (Value arg : block->getArguments()) {
      if (dependencies.sameAs(arg, value))
        return arg;
    }
    return {};
  }

  // Returns a value representing the always ready 'root chain', ie %arg0 of
  // the 'execute' block or its local equivalent, provided we have
  // dependencies.isAlwaysReady for that value. Returns null if no such value
  // can be found.
  Value getAlwaysReadyRootChain(const AsyncDataflowDependencies &dependencies,
                                ModelOp modelOp, Operation *op) {
    Block *rootBlock = getRootBlock(modelOp, op);
    if (rootBlock == modelOp.getExecuteBlock()) {
      Value rootChain = rootBlock->getArgument(0);
      if (dependencies.isAlwaysReady(rootChain))
        return getEquivArgument(dependencies, op->getBlock(), rootChain);
    }
    return {};
  }

  void runOnOperation() override {
    ModelOp modelOp = getOperation();
    // Construct the dependencies from the model. The transforms we
    // do below will preserve these dependencies since we never introduce
    // new bound values or add new dependencies.
    IndexSet<Value> index = createValueIndexSet(modelOp);
    AsyncDataflowDependencies dependencies(&index);

    // CAUTION: We'll be making in-place updates as we go. However, since
    // we don't update the dependencies analysis results, we must be careful
    // to only do rewrites which are invariant w.r.t. that.

    modelOp.walk([&](GRT::ChainCreateOp chainCreateOp) {
      ValueSetVector valuesSet;
      accumulatePrimitiveChains(valuesSet, chainCreateOp.getResult());
      SmallVector<Value> values = valuesSet.takeVector();
      dependencies.simplify(values);
      if (values.empty()) {
        // Instead of creating a fresh chain, try to reuse the 'execute'
        // block's %arg0 if it is also always ready.
        Value replacementValue =
            getAlwaysReadyRootChain(dependencies, modelOp, chainCreateOp);
        if (replacementValue) {
          chainCreateOp.getResult().replaceAllUsesWith(replacementValue);
          chainCreateOp.erase();
        } else {
          chainCreateOp->setOperands(values);
        }
      } else if (values.size() == 1) {
        chainCreateOp.getResult().replaceAllUsesWith(values.front());
        chainCreateOp.erase();
      } else {
        chainCreateOp->setOperands(values);
      }
    });
  }
};

} // namespace

} // namespace M::MGP
