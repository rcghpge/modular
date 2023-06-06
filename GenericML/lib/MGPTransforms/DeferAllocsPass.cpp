//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "Cache/CacheDialect/CacheAttrs.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

#include "./Utils.h"

using namespace mlir;

namespace M::MGP {

// Generate support code for DeferAllocsPass
#define GEN_PASS_DEF_DEFERALLOCSPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// DeferAllocsPass
//===----------------------------------------------------------------------===//

/// This pass defers buffer allocations until the operations which write to
/// them are ready to execute.
class DeferAllocsPass : public impl::DeferAllocsPassBase<DeferAllocsPass> {
  void runOnOperation() override {
    ModelOp model = getOperation();
    ExecuteState execState(model.getContext(), model.getExecuteBlock());
    // This helps prevent infinite loop in case an op write to multiple buffers
    DenseSet<Operation *> movedAllocOps;

    // For each mgp.cpu.buffer.alloc.*...
    model.getExecute().walk([&](Operation *op) {
      if (movedAllocOps.contains(op))
        return;
      if (auto allocOp = dyn_cast<MGPBufferAllocInterface>(op)) {
        // For the unique writer to the allocated buffer...
        if (Operation *writerOp = getUniqueWriter(allocOp.getOutBuffer())) {
          // For the input chain(s) to that writer....
          for (OpOperand *writerChainOpOperand :
               getInputChainOpOperands(writerOp)) {
            // Move the allocation to just before the writing op.
            // (Note that this may mean we revisit the same allocation again.)
            allocOp->moveBefore(writerOp);
            movedAllocOps.insert(allocOp);
            // The allocation must only be triggered once the writing op's
            // input chain is ready.
            Value writerInChain = writerChainOpOperand->get();
            auto res =
                alsoWaitForChains(execState.builder, allocOp, {writerInChain});
            assert(succeeded(res) && "All buffer allocation ops are assumed to "
                                     "accept an input chain");
          }
        }
      }
    });
    execState.erasePending();
  }
};

} // namespace
} // namespace M::MGP
