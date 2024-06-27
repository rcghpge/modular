//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "GenericML/MGPDialect/MGPBufferLifetimeAnalysis.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTensorInterface.h"
#include "GenericML/MGPTransforms/Utils.h"
#include "Support/AssertStream.h"

#define DEBUG_TYPE "mgp-explicit-buffer-lifetimes"

using namespace mlir;

namespace M::MGP {

// Generate support code for ExplicitBufferLifetimesPass
#define GEN_PASS_DEF_EXPLICITBUFFERLIFETIMESPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

class ExplicitBufferLifetimesPass
    : public impl::ExplicitBufferLifetimesPassBase<
          ExplicitBufferLifetimesPass> {
  using ExplicitBufferLifetimesPassBase::ExplicitBufferLifetimesPassBase;

  void runOnOperation() override {
    ModelOp modelOp = getOperation();

    int numAllocs = 0;
    int numFreesAdded = 0;

    auto analysis = getAnalysis<BufferLifetimeAnalysisPass>();
    markAnalysesPreserved<BufferLifetimeAnalysisPass>();

    modelOp->walk([&](Block *block) {
      // Walk each block, then for each alloc.dynamic in each block, insert a
      // buffer free at the op that postdominates all of the alloc's uses, which
      // means it is guaranteed to free memory only after all uses of the buffer
      // have finished executing

      for (Operation &op : block->getOperations()) {
        auto allocOp = dyn_cast<MGP::BufferAllocDynamicOp>(op);
        if (!allocOp)
          continue;
        numAllocs++;

        auto allocUses = getAllocUses(allocOp);
        if (allocUses.isError())
          continue;
        auto ipdomNode = analysis.getDomTree(block).findLCA(*allocUses);

        // Traverse up the postdominator tree until we find a node that has an
        // MGPDeviceChain output, meaning we can insert our free after it
        while (ipdomNode) {
          bool hasMGPChainOutput = false;
          for (auto result : ipdomNode->op->getResults())
            if (isa<MGP::DeviceChainType>(result.getType()))
              hasMGPChainOutput = true;
          if (hasMGPChainOutput)
            break;
          ipdomNode = ipdomNode->parent;
        }

        if (!ipdomNode) {
          LLVM_DEBUG(llvm::dbgs()
                     << "could not find IPDOM for " << allocOp << "\n");
          continue;
        }

        auto ipdom = ipdomNode->op;

        OpBuilder builder(ipdom);
        auto loc = ipdom->getLoc();

        OpResult chainOutput;
        for (auto result : ipdom->getResults()) {
          if (isa<MGP::DeviceChainType>(result.getType())) {
            chainOutput = result;
            break;
          }
        }
        ASSERT_STREAM(chainOutput, "op should have chain output");

        auto bufferFree = builder.create<MGP::BufferFreeOp>(
            loc, chainOutput, allocOp.getOutBuffer());

        chainOutput.replaceAllUsesExcept(bufferFree, bufferFree);
        numFreesAdded++;
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "added frees for " << numFreesAdded << " out of "
                            << numAllocs << " allocs\n");
  }
};
} // namespace M::MGP
