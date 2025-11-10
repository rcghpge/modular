//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/MGPDialect/MGPBufferLifetimeAnalysis.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPTransforms/MGPTransforms.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "mgp-simple-buffer-reuse"

using namespace mlir;

namespace M {
namespace MGP {

struct ReuseKey {
  mlir::TypedValue<mlir::IndexType> size;
  Attribute device;
  uint64_t alignment;

  bool operator==(const ReuseKey &other) const {
    return size == other.size && device == other.device &&
           alignment == other.alignment;
  }
};

} // namespace MGP
} // namespace M

namespace llvm {
template <>
struct DenseMapInfo<M::MGP::ReuseKey> {
  using KeyTy = M::MGP::ReuseKey;

  static inline KeyTy getEmptyKey() {
    return {mlir::TypedValue<mlir::IndexType>(),
            DenseMapInfo<Attribute>::getEmptyKey(),
            DenseMapInfo<uint64_t>::getEmptyKey()};
  }

  static inline KeyTy getTombstoneKey() {
    return {mlir::TypedValue<mlir::IndexType>(),
            DenseMapInfo<Attribute>::getTombstoneKey(),
            DenseMapInfo<uint64_t>::getTombstoneKey()};
  }

  static unsigned getHashValue(const KeyTy &k) {
    return hash_combine(DenseMapInfo<mlir::Value>::getHashValue(k.size),
                        DenseMapInfo<Attribute>::getHashValue(k.device),
                        DenseMapInfo<uint64_t>::getHashValue(k.alignment));
  }

  static bool isEqual(const KeyTy &a, const KeyTy &b) { return a == b; }
};
} // namespace llvm

namespace M {
namespace MGP {

// Generate support code for SimpleBufferReusePass
#define GEN_PASS_DEF_SIMPLEBUFFERREUSEPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

class SimpleBufferReusePass
    : public impl::SimpleBufferReusePassBase<SimpleBufferReusePass> {
  using SimpleBufferReusePassBase::SimpleBufferReusePassBase;
  using LifetimeRange = std::pair<int, int>;

  struct AllocInfo {
    MGP::BufferAllocOp alloc;
    int firstUseIndex;
    int lastUseIndex;
    SmallVector<Operation *> uses;
  };

  // Check if buffer uses are compatible for reuse using existing runsAfter
  // bitsets.
  // This leverages the precomputed dependency information in
  // DependencyTree
  bool canSafelyReuseBuffer(ArrayRef<Operation *> currentUses,
                            ArrayRef<Operation *> candidateUses,
                            const DependencyTree &depTree) {
    // Early termination for empty sets
    if (currentUses.empty() || candidateUses.empty())
      return true;

    // For each use in current allocation, verify it executes after all
    // candidate uses
    for (Operation *currentUse : currentUses) {
      for (Operation *candidateUse : candidateUses) {
        // Check if current executes after candidate using the executesAfter
        // method
        if (!depTree.executesAfter(currentUse, candidateUse)) {
          return false;
        }
      }
    }
    return true;
  }

public:
  void runOnOperation() override {
    ModelOp modelOp = getOperation();

    auto &analysis = getAnalysis<BufferLifetimeAnalysisPass>();
    int reuseCount = 0;

    // Mark the analysis as preserved so subsequent passes can reuse it
    // Note: since the IR is modified (by reusing and erasing allocs), the
    // reused analysis is not valid. However, since the next pass to run (which
    // will reuse this analysis) checks the uses of the allocs and not the
    // allocs themselves, this is safe to do.
    markAnalysesPreserved<BufferLifetimeAnalysisPass>();

    modelOp->walk<mlir::WalkOrder::PreOrder>([&](Block *block) {
      if (isa_and_present<MOGG::GenericCompiledRegionOp>(block->getParentOp()))
        return WalkResult::skip();

      DenseMap<ReuseKey, SmallVector<MGP::BufferAllocOp>> allocGroups;

      // First pass: Group allocs by reuse key
      for (auto allocOp : block->getOps<MGP::BufferAllocOp>()) {
        ReuseKey key = {allocOp.getRuntimeSize(), allocOp.getDevice(),
                        allocOp.getRawAlign()};
        allocGroups[key].push_back(allocOp);
      }

      if (allocGroups.empty())
        return WalkResult::advance();

      auto &depTree = analysis.getDepTree(block);

      // Second pass: Try to reuse buffers
      for (auto &[key, allocs] : allocGroups) {
        // Skip any groups with no reuse potential
        if (allocs.size() < 2)
          continue;

        // Collect all valid allocs with their use information first
        SmallVector<std::pair<MGP::BufferAllocOp, SmallVector<Operation *>>>
            validAllocs;
        validAllocs.reserve(allocs.size());

        for (auto alloc : allocs) {
          ErrorOr<SmallVector<Operation *>> uses = getAllocUses(alloc);
          if (!uses.isError())
            validAllocs.emplace_back(alloc, std::move(*uses));
        }

        // Sort allocations to prioritize those with more uses
        llvm::stable_sort(validAllocs, [](const auto &a, const auto &b) {
          return a.second.size() > b.second.size();
        });

        // Check all pairs of current and candidate allocations for reuse
        // opportunities
        for (unsigned i = 0; i < validAllocs.size(); ++i) {
          auto &pair = validAllocs[i];
          MGP::BufferAllocOp currentAlloc = pair.first;
          auto &currentUses = pair.second;
          if (!currentAlloc)
            continue; // Skip already erased allocs

          for (unsigned j = 0; j < validAllocs.size(); ++j) {
            if (i == j)
              continue;
            auto &candidatePair = validAllocs[j];
            MGP::BufferAllocOp candidateAlloc = candidatePair.first;
            auto &candidateUses = candidatePair.second;
            if (!candidateAlloc)
              continue;

            // Skip if there are any shared operations between uses.
            // This is for the case where two buffers are used in the same
            // operation, which we can't do anything about since reusing would
            // create a conflicting read/write scenario
            bool sharesOperation = false;
            for (Operation *op : currentUses) {
              if (llvm::is_contained(candidateUses, op)) {
                sharesOperation = true;
                break;
              }
            }
            if (sharesOperation)
              continue;

            // Check if the candidate can be reused (if all uses of the current
            // buffer execute after all uses of the candidate buffer)
            if (canSafelyReuseBuffer(currentUses, candidateUses, depTree)) {
              currentAlloc->replaceAllUsesWith(candidateAlloc);
              currentAlloc->erase();
              validAllocs[i].first = nullptr; // Mark as erased

              // Update uses of candidateAlloc
              SmallVector<Operation *> newUses;
              newUses.reserve(candidateUses.size() + currentUses.size());
              newUses.append(candidateUses.begin(), candidateUses.end());
              newUses.append(currentUses.begin(), currentUses.end());
              validAllocs[j].second = std::move(newUses);

              reuseCount++;
              break; // Move to next alloc after successful reuse
            }
          }
        }
      }

      return WalkResult::advance();
    });

    LLVM_DEBUG(llvm::dbgs()
               << "Reused " << reuseCount << " buffer allocations\n");
  }
};

} // namespace MGP
} // namespace M
