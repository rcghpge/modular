//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "GenericML/MGPDialect/MGPBufferLifetimeAnalysis.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTensorInterface.h"
#include "Support/AssertStream.h"
#include "Utils.h"

#define DEBUG_TYPE "mgp-explicit-buffer-lifetimes"

using namespace mlir;

namespace M::MGP {

// Generate support code for ExplicitBufferLifetimesPass
#define GEN_PASS_DEF_EXPLICITBUFFERLIFETIMESPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

namespace {

Operation *
findImmediatePostdominator(Block *block,
                           const BufferDominationRange &dominationRange,
                           const BufferLifetimeInfo &lifetimeInfo) {
  // To find IPDOM, first find all ops in the block that postdominate the given
  // range. Then, find the one op that appears first in the postorder traversal
  // in order to find the immediate postdominator
  SmallVector<Operation *> postDominators;
  for (auto &op : block->getOperations()) {
    if (!lifetimeInfo.hasDominationRange(&op))
      continue;

    const auto &info = lifetimeInfo.getDominationRange(&op);
    if (BufferLifetimeInfo::postdominates(info, dominationRange)) {
      postDominators.push_back(&op);
      LLVM_DEBUG(llvm::dbgs() << "\t postdom found: " << op << "\n");
    }
  }

  Operation *ipdom = nullptr;
  int minPost = std::numeric_limits<int>::max();

  for (auto *postDom : postDominators) {
    // Only consider postdominators that return a !mgp.chain so
    // the buffer.free can depend on it
    bool hasMGPChainOutput = false;
    for (auto result : postDom->getResults())
      if (isa<MGP::DeviceChainType>(result.getType()))
        hasMGPChainOutput = true;

    if (!hasMGPChainOutput)
      continue;

    // Immediate postdominator is the one that appears first in the
    // post-order walk
    if (!lifetimeInfo.hasDominationRange(postDom))
      continue;
    const auto &info = lifetimeInfo.getDominationRange(postDom);
    if (info.postIndex < minPost) {
      ipdom = postDom;
      minPost = info.postIndex;
    }
  }

  return ipdom;
}

} // namespace

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

        auto allocUseRange = analysis.lifetimeInfo.getAllocUseRange(allocOp);
        if (allocUseRange.isError()) {
          LLVM_DEBUG(llvm::dbgs() << "invalid allocUseRange for " << allocOp
                                  << ": " << allocUseRange.getError() << "\n");
          continue;
        }

        auto ipdom = findImmediatePostdominator(block, allocUseRange.get(),
                                                analysis.lifetimeInfo);

        if (!ipdom) {
          LLVM_DEBUG(llvm::dbgs()
                     << "could not find IPDOM for " << allocOp << "\n");
          continue;
        }

        OpBuilder builder(ipdom);
        auto loc = ipdom->getLoc();
        BufferFreeOp bufferFree;

        // ipdom is guaranteed to have a chain output by
        // findImmediatePostdominator
        OpResult chainOutput;
        for (auto result : ipdom->getResults()) {
          if (isa<MGP::DeviceChainType>(result.getType())) {
            chainOutput = result;
          }
        }
        ASSERT_STREAM(chainOutput, "op should have chain output");

        bufferFree = builder.create<MGP::BufferFreeOp>(loc, chainOutput,
                                                       allocOp.getOutBuffer());

        chainOutput.replaceUsesWithIf(bufferFree, [&](auto &opOperand) {
          return opOperand.getOwner() != bufferFree;
        });

        numFreesAdded++;
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "added frees for " << numFreesAdded << " out of "
                            << numAllocs << " allocs\n");
  }
};
} // namespace M::MGP
