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
  using LifetimeRange = std::pair<int, int>;

  void runOnOperation() override {
    ModelOp modelOp = getOperation();

    auto analysis = getAnalysis<BufferLifetimeAnalysisPass>();

    modelOp->walk([&](Block *block) {
      // Get dependency tree for the current block
      auto &tree = analysis.getDepTree(block);

      // Map from alloc to its uses' lifetime range in the tree
      // narrow represents max(x.pre for x in uses), min(x.post for x in uses)
      // wide represents min(x.pre for x in uses), max(x.post for x in uses)
      DenseMap<MGP::BufferAllocDynamicOp, LifetimeRange> narrowAllocRanges;
      DenseMap<MGP::BufferAllocDynamicOp, LifetimeRange> wideAllocRanges;

      // Populate alloc ranges with information from BufferLifetimeAnalysisPass
      for (auto allocOp : block->getOps<MGP::BufferAllocDynamicOp>()) {
        auto allocUses = getAllocUses(allocOp);

        if (allocUses.isError() || allocUses->empty())
          continue;

        int maxPre = std::numeric_limits<int>::min();
        int minPre = std::numeric_limits<int>::max();
        int minPost = std::numeric_limits<int>::max();
        int maxPost = std::numeric_limits<int>::min();

        bool ineligibleAlloc = false;
        for (auto use : allocUses.get()) {
          if (!tree.contains(use)) {
            ineligibleAlloc = true;
            break;
          }
          maxPre = std::max(maxPre, tree.get(use)->preIndex);
          minPost = std::min(minPost, tree.get(use)->postIndex);
          minPre = std::min(minPre, tree.get(use)->preIndex);
          maxPost = std::max(maxPost, tree.get(use)->postIndex);
        }

        // If the tree didn't have information on every use of this alloc, skip
        // it to be safe
        if (ineligibleAlloc)
          continue;

        narrowAllocRanges[allocOp] = {maxPre, minPost};
        wideAllocRanges[allocOp] = {minPre, maxPost};
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
        if (!curAlloc || !narrowAllocRanges.contains(curAlloc))
          continue;

        auto &curRange = narrowAllocRanges[curAlloc];

        auto hasBeenFreed = [&](auto alloc) -> bool {
          // An alloc can be considered freed if all the uses of the current
          // alloc are ancestors of all of its uses in the dependency tree
          const auto &usedRange = wideAllocRanges[alloc];
          return curRange.first < usedRange.first &&
                 curRange.second > usedRange.second;
        };

        // Go through all live allocations to see if there are any that the
        // current allocation can reuse
        auto toReuse =
            llvm::find_if(liveAllocs, [&](MGP::BufferAllocDynamicOp candidate) {
              return candidate.getRuntimeSize() == curAlloc.getRuntimeSize() &&
                     candidate.getDevice() == curAlloc.getDevice() &&
                     candidate.getRawAlign() == curAlloc.getRawAlign() &&
                     hasBeenFreed(candidate);
            });

        if (toReuse != liveAllocs.end()) {
          // If we found a valid reuse candidate update the map for the current
          // alloc to point to the reused alloc's range
          narrowAllocRanges[*toReuse] = narrowAllocRanges[curAlloc];
          wideAllocRanges[*toReuse] = wideAllocRanges[curAlloc];

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
