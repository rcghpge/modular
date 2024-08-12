//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "./AsyncDataflowDependencies.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPTransforms/Utils.h"
#include "GraphRT/GRTDialect/GRTChainInterface.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "GraphRT/MEFDialect/MEFTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

#define DEBUG_TYPE "mgp-simplify-chains-pass"

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

  /// If chain is defined by a grt.chain.create operation then accumulate
  /// all of its operands into primitiveChains. Otherwise adds chain directly
  /// to primitiveChains.
  void accumulatePrimitiveChains(IndexSet<Value> &valueIndex,
                                 IndexSet<Value>::Set &primitiveChains,
                                 Value chain) {
    // Only use chains where the input is settable. These ops assume that the
    // input and output chains have the same type.
    if (auto chainCreateOp =
            chain.getDefiningOp<GRT::GRTGatedOnChainOpInterface>()) {
      for (auto operand : dyn_cast<GRT::GRTChainCreateOpInterface>(
                              chainCreateOp.getOperation())
                              .getInputChains())
        accumulatePrimitiveChains(valueIndex, primitiveChains, operand);
    } else {
      valueIndex.addToSet(primitiveChains, chain);
    }
  }

  void runOnOperation() override {
    ModelOp modelOp = getOperation();

    // Construct the dependencies from the model. The transforms we
    // do below will preserve these dependencies since we never introduce
    // new bound values or add new dependencies.
    IndexSet<Value> valueIndex = createValueIndexSet(modelOp);
    IndexSet<Operation *> operationIndex = createOperationIndexSet(modelOp);

    LLVM_DEBUG(llvm::dbgs()
               << "SimplifyChainsPass: analyzing " << valueIndex.size()
               << " values and " << operationIndex.size() << " operations\n");

    AsyncDataflowDependencies dependencies(
        &valueIndex, &operationIndex, AsyncDataflowDependencies::kImpliesReady);

    // CAUTION: We'll be making in-place updates as we go. However, since
    // we don't update the dependencies analysis results, we must be careful
    // to only do rewrites which are invariant w.r.t. that.

    LLVM_DEBUG(llvm::dbgs() << "SimplifyChainsPass: rewriting\n");

    modelOp.walk([&](GRT::GRTChainCreateOpInterface chainCreateOp) {
      IndexSet<Value>::Set valuesSet = valueIndex.emptySet();
      accumulatePrimitiveChains(valueIndex, valuesSet,
                                chainCreateOp.getChainResult());
      dependencies.simplify(valuesSet);
      SmallVector<Value> values = valueIndex.setToVector(valuesSet);
      if (values.empty() &&
          dyn_cast<GRT::ChainCreateOp>(chainCreateOp.getOperation())) {
        // Instead of creating a fresh chain, try to reuse the 'execute'
        // block's %arg0 if it is also always ready.
        Value replacementValue =
            getAlwaysReadyRootChain(dependencies, modelOp, chainCreateOp);
        if (replacementValue) {
          chainCreateOp.getChainResult().replaceAllUsesWith(replacementValue);
          chainCreateOp.erase();
        } else {
          chainCreateOp->setOperands({});
        }
      } else if (values.size() == 1 &&
                 values[0].getType() ==
                     chainCreateOp.getChainResult().getType()) {
        // We can only replace %x = <chain create op>(%y) with %y if %y and
        // %x have the same type. This is because chain create ops may accept
        // arbitrary values and/or chains of different types.
        chainCreateOp.getChainResult().replaceAllUsesWith(values.front());
        chainCreateOp.erase();
      } else {
        if (auto chainSetOp = dyn_cast<GRT::GRTGatedOnChainOpInterface>(
                chainCreateOp.getOperation()))
          chainSetOp.setInputChains(values);
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "SimplifyChainsPass: done\n");
  }
};

} // namespace

} // namespace M::MGP
