//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTensorInterface.h"
#include "GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "Support/AssertStream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

#define DEBUG_TYPE "mgp-defer-allocs"

namespace M::MGP {

// Generate support code for DeferAllocsPass
#define GEN_PASS_DEF_DEFERALLOCSPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// DeferAllocsPass
//===----------------------------------------------------------------------===//

/// This pass defers buffer allocations until the operations which write to
/// them are ready to execute.
class DeferAllocsPass : public impl::DeferAllocsPassBase<DeferAllocsPass> {
  void runOnOperation() override {
    ModelOp model = getOperation();
    ExecuteState execState(model.getContext(), model);
    // Recover type and buffer builders appropriate for the model.
    BufferBuilderContext builderContext =
        TensorBufferDialectInterface::getAllBufferBuilders(model);

    // This helps prevent infinite loop in case an op write to multiple buffers
    DenseSet<Operation *> movedAllocOps;

    auto defer = [&](Operation *op) {
      if (movedAllocOps.contains(op))
        return;

      auto allocOp = dyn_cast<MGPBufferAllocOpInterface>(op);
      if (!allocOp)
        return;

      OpOperand *mutator = getUniqueMutator(allocOp.getOutBuffer());
      if (!mutator)
        return;

      // Move the allocation to just before the mutating op.
      // (Note that this may mean we revisit the same allocation again.)
      allocOp->moveBefore(mutator->getOwner());
      movedAllocOps.insert(allocOp);

      // Make the allocation depend on the same chains as the mutator.
      SmallVector<Value> additionalChainsToWaitOn;
      for (OpOperand *mutatorChainOpOperand :
           getInputChainOpOperands(mutator->getOwner())) {
        additionalChainsToWaitOn.push_back(mutatorChainOpOperand->get());
      }

      execState.builder.setInsertionPoint(allocOp);
      [[maybe_unused]] auto res = alsoWaitForChains(
          execState.builder, builderContext, allocOp, additionalChainsToWaitOn);
      ASSERT_STREAM(succeeded(res),
                    << "all allocation ops must accept an input "
                       "chain to control their timing");
    };

    // For each mgp.buffer.alloc.*...
    model.getExecute().walk(defer);
    model.getInit().walk(defer);
  }
};

} // namespace
} // namespace M::MGP
