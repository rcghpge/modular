//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "GenericML/MGPDialect/MGPBufferLifetimeAnalysis.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPTransforms/Utils.h"
#include "Support/AssertStream.h"

#define DEBUG_TYPE "mgp-simple-buffer-reuse"

using namespace mlir;

namespace M::MGP {

// Generate support code for SimpleBufferReusePass
#define GEN_PASS_DEF_SIMPLEBUFFERREUSEPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

class SimpleBufferReusePass
    : public impl::SimpleBufferReusePassBase<SimpleBufferReusePass> {
  using SimpleBufferReusePassBase::SimpleBufferReusePassBase;
  using PostDominationRange = std::pair<int, int>;

  void runOnOperation() override {
    ModelOp modelOp = getOperation();

    auto analysis = getAnalysis<BufferLifetimeAnalysisPass>();

    modelOp->walk([&](Block *block) {
      // Get postdominator tree for the current block
      auto &tree = analysis.getDomTree(block);

      // Map from alloc to its postdomination range in the tree
      DenseMap<MGP::BufferAllocDynamicOp, PostDominationRange> allocRanges;

      // Populate allocRanges with information from BufferLifetimeAnalysisPass
      for (auto allocOp : block->getOps<MGP::BufferAllocDynamicOp>()) {
        auto allocUses = getAllocUses(allocOp);

        if (allocUses.isError() || allocUses->empty())
          continue;

        int maxPre = std::numeric_limits<int>::min();
        int minPost = std::numeric_limits<int>::max();

        bool ineligibleAlloc = false;
        for (auto use : allocUses.get()) {
          if (!tree.contains(use)) {
            ineligibleAlloc = true;
            break;
          }
          maxPre = std::max(maxPre, tree.get(use)->preIndex);
          minPost = std::min(minPost, tree.get(use)->postIndex);
        }

        // If the tree didn't have information on every use of this alloc, skip
        // it to be safe
        if (ineligibleAlloc)
          continue;

        allocRanges[allocOp] = {maxPre, minPost};
      }

      SmallVector<MGP::BufferAllocDynamicOp> liveAllocs;
      SmallVector<MGP::BufferAllocDynamicOp> replaced;

      // In dataflow order, go through every alloc A in the block while
      // maintaining a set of live allocations

      // For each alloc A in the block, go through each alloc B in the set of
      // live allocations and try to find a candidate C for reuse. If such a
      // candidate C exists, use alloc C as a replacement for alloc A
      for (auto op : getDataflowOrder(block)) {
        auto curAlloc = dyn_cast<MGP::BufferAllocDynamicOp>(*op);
        if (!curAlloc || !allocRanges.contains(curAlloc))
          continue;

        auto &curRange = allocRanges[curAlloc];

        auto hasBeenFreed = [&](auto alloc) -> bool {
          // An alloc can be considered freed if all the uses of the current
          // alloc strictly postdominates all of its uses
          auto allocUses = getAllocUses(alloc).get();
          bool freed = true;
          for (auto use : allocUses) {
            assert(tree.contains(use)); // if allocRanges contains alloc, all
                                        // uses should be in tree
            auto node = tree.get(use);
            freed &= curRange.first < node->preIndex &&
                     curRange.second > node->postIndex;
          }
          return freed;
        };

        // Go through all live allocations to see if there are any that the
        // current allocation can reuse
        auto toReuse =
            llvm::find_if(liveAllocs, [&](MGP::BufferAllocDynamicOp candidate) {
              return candidate.getRuntimeSize() == curAlloc.getRuntimeSize() &&
                     candidate.getDevice() == curAlloc.getDevice() &&
                     candidate.getInChain() == curAlloc.getInChain() &&
                     candidate.getRawAlign() == curAlloc.getRawAlign() &&
                     hasBeenFreed(candidate);
            });

        if (toReuse != liveAllocs.end()) {
          // If we found a valid reuse candidate update the map for the current
          // alloc to point to the reused alloc's range
          allocRanges[*toReuse] = allocRanges[curAlloc];

          // Replace curAlloc and add to vector for future deletion
          replaced.push_back(curAlloc);
          curAlloc->replaceAllUsesWith(*toReuse);
        } else {
          // Otherwise, just add the current allocation to liveAllocs
          liveAllocs.push_back(curAlloc);
        }
      }

      // Erase all the replaced allocs
      for (auto &toErase : replaced)
        toErase.erase();

      LLVM_DEBUG(llvm::dbgs() << "reused " << replaced.size() << " allocs\n");
    });
  }
};
} // namespace M::MGP
