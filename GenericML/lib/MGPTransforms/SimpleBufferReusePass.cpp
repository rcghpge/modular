//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "GenericML/MGPDialect/MGPBufferLifetimeAnalysis.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "Support/AssertStream.h"
#include "Utils.h"

#define DEBUG_TYPE "mgp-simple-buffer-reuse"

using namespace mlir;

namespace M::MGP {

// Generate support code for SimpleBufferReusePass
#define GEN_PASS_DEF_SIMPLEBUFFERREUSEPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

Operation *getFirstUse(const PostDominatorTree &tree,
                       const SmallVector<Operation *> &uses) {
  int maxDepth = -1;
  Operation *firstUse{nullptr};
  for (Operation *use : uses) {
    if (use->getBlock() != tree.getBlock())
      continue;
    assert(tree.contains(use));
    int depth = tree.get(use)->depth;
    if (depth > maxDepth) {
      maxDepth = depth;
      firstUse = use;
    }
  }
  return firstUse;
}

class SimpleBufferReusePass
    : public impl::SimpleBufferReusePassBase<SimpleBufferReusePass> {
  using SimpleBufferReusePassBase::SimpleBufferReusePassBase;

  void runOnOperation() override {
    ModelOp modelOp = getOperation();

    auto analysis = getAnalysis<BufferLifetimeAnalysisPass>();

    modelOp->walk([&](Block *block) {
      SmallVector<MGP::BufferAllocDynamicOp> allocs;

      // Map from an alloc to the node in the dominator tree corresponding to
      // its first use
      DenseMap<MGP::BufferAllocDynamicOp, PostDominatorTreeNode *>
          firstUseDomNode;

      // Map from an alloc to the node in the dominator tree corresponding to
      // the lowest common ancestor of all of its uses
      DenseMap<MGP::BufferAllocDynamicOp, PostDominatorTreeNode *>
          allUsesLCADomNode;

      // Populate the firstUseDomNode and allUsesLCADomNode maps with
      // information from the BufferLifetimeAnalysisPass
      for (auto allocOp : block->getOps<MGP::BufferAllocDynamicOp>()) {
        auto allocUses = getAllocUses(allocOp);

        if (allocUses.isError() || allocUses->empty())
          continue;
        auto firstUse = getFirstUse(analysis.getDomTree(block), *allocUses);
        if (!firstUse)
          continue;

        firstUseDomNode[allocOp] = analysis.getDomTree(block).get(firstUse);

        allUsesLCADomNode[allocOp] =
            analysis.getDomTree(block).findLCA(*allocUses);
      }

      SmallVector<MGP::BufferAllocDynamicOp> liveAllocs;
      SmallVector<MGP::BufferAllocDynamicOp> replaced;

      // In dataflow order, go through every alloc A in the block while
      // maintaining a set of live allocations

      // For each alloc A in the block, go through each alloc B in the set of
      // live allocations and try to find a candidate C for reuse. If such a
      // candidate C exists, use alloc C as a replacement for alloc A
      // can reuse, add it to the list of rewrites that will later be applied
      for (auto op : getDataflowOrder(block)) {
        auto curAlloc = dyn_cast<MGP::BufferAllocDynamicOp>(*op);
        if (!curAlloc || !firstUseDomNode.contains(curAlloc))
          continue;

        auto curFirstUseNode = firstUseDomNode[curAlloc];

        auto hasBeenFreed = [&](auto alloc) -> bool {
          // An alloc can be considered freed if the first use of the current
          // alloc postdominates all of its uses and if the first use of the
          // current alloc doesn't use it directly in order to prevent the case
          // where a kernel execute uses the same buffer for both input and
          // output
          bool freed = analysis.getDomTree(block).postdominates(
              curFirstUseNode, allUsesLCADomNode[alloc]);
          auto firstUse = curFirstUseNode->op;

          if (llvm::any_of(firstUse->getOperands(), [&](Value operand) {
                return operand.getDefiningOp() == alloc;
              }))
            freed = false;

          return freed;
        };

        // Go through all live allocations to see if there are any that the
        // current allocation can reuse
        // MGP::BufferAllocDynamicOp toReuse;
        auto toReuse =
            llvm::find_if(liveAllocs, [&](MGP::BufferAllocDynamicOp candidate) {
              return hasBeenFreed(candidate) &&
                     candidate.getRuntimeSize() == curAlloc.getRuntimeSize() &&
                     candidate.getDevice() == curAlloc.getDevice() &&
                     candidate.getInChain() == curAlloc.getInChain() &&
                     candidate.getRawAlign() == curAlloc.getRawAlign();
            });

        if (toReuse != liveAllocs.end()) {
          // If we found a valid reuse candidate, move it out of freed and back
          // into liveAllocs, and update the map for the current alloc to point
          // to the reused alloc's IPDOM
          allUsesLCADomNode[*toReuse] = allUsesLCADomNode[curAlloc];

          // Replace curAlloc and add to vector for future deletion
          replaced.push_back(curAlloc);
          curAlloc->replaceAllUsesWith(*toReuse);
        } else {
          // Otherwise, just add the current allocation to liveAllocs
          liveAllocs.push_back(curAlloc);
        }
      }

      // Erase all the replaced allocs
      for (auto &toErase : replaced) {
        toErase.erase();
      }

      LLVM_DEBUG(llvm::dbgs() << "reused " << replaced.size() << " allocs\n");
    });
  }
};
} // namespace M::MGP
