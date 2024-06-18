//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTensorInterface.h"
#include "Support/AssertStream.h"

namespace M::MGP {

struct BufferDominationRange {
  int preIndex;
  int postIndex;

  BufferDominationRange overlap(const BufferDominationRange &other) {
    return BufferDominationRange{std::min(preIndex, other.preIndex),
                                 std::max(postIndex, other.postIndex)};
  }
};

class BufferLifetimeInfo {
private:
  using DominationAnalysisMap = DenseMap<Operation *, BufferDominationRange>;
  DominationAnalysisMap dominationMap;

public:
  void calculateDataflowPostdominance(Operation *terminator);

  // Returns true if a postdominates b, meaning that all paths to the block
  // terminator from a must pass through b
  bool postdominates(Operation *a, Operation *b) const;

  // Same as above, but accepts ranges directly. Assumes ranges are from the
  // same block.
  static bool postdominates(const BufferDominationRange &a,
                            const BufferDominationRange &b);

  // Check if an op has domination range information (if it has been analyzed)
  bool hasDominationRange(Operation *op) const;

  // Get the domination range of a single op
  BufferDominationRange getDominationRange(Operation *op) const;

  // Get the domination range of a list of ops belonging to the same block
  ErrorOr<BufferDominationRange>
  getDominationRange(SmallVector<Operation *> ops) const;

  // Returns the domination range of an alloc and all of its uses (including
  // aliases)
  ErrorOr<BufferDominationRange>
  getAllocUseRange(MGP::BufferAllocDynamicOp allocOp) const;

private:
  // Return every use of a buffer including via aliases
  SmallVector<Operation *>
  getAllocUses(MGP::BufferAllocDynamicOp allocOp) const;
};

// Analysis pass to do dataflow postdominator analysis on buffer uses to assist
// with memory planning
struct BufferLifetimeAnalysisPass {
  BufferLifetimeAnalysisPass(Operation *op);

  BufferLifetimeInfo lifetimeInfo;
};

} // namespace M::MGP
