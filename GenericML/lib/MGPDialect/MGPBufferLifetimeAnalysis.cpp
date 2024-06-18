//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPBufferLifetimeAnalysis.h"

namespace M::MGP {

void BufferLifetimeInfo::calculateDataflowPostdominance(Operation *terminator) {
  using VisitState = std::pair<Operation *, bool>;
  SmallVector<VisitState> stack;

  stack.push_back({terminator, true});
  int index = 0;

  while (!stack.empty()) {
    auto [top, isFirstVisit] = stack.back();
    stack.pop_back();
    if (!top)
      continue;

    if (isFirstVisit) {
      // preorder
      auto it = dominationMap.find(top);
      if (it == dominationMap.end()) {
        dominationMap[top] = {index++, -1};

        // Schedule to visit again for post-order
        stack.push_back({top, false});

        for (auto operand : top->getOperands()) {
          Operation *op = operand.getDefiningOp();
          if (op)
            stack.push_back({op, true});
        }
      }
    } else {
      // postorder
      auto it = dominationMap.find(top);
      if (it != dominationMap.end() && it->getSecond().postIndex == -1)
        it->getSecond().postIndex = index++;
    }
  }
}

bool BufferLifetimeInfo::postdominates(Operation *a, Operation *b) const {
  if (a->getBlock() != b->getBlock())
    return false;

  auto map = dominationMap;
  auto aRange = map[a];
  auto bRange = map[b];

  return postdominates(aRange, bRange);
}

bool BufferLifetimeInfo::postdominates(const BufferDominationRange &a,
                                       const BufferDominationRange &b) {
  return a.preIndex <= b.preIndex && a.postIndex >= b.postIndex;
}

bool BufferLifetimeInfo::hasDominationRange(Operation *op) const {
  return dominationMap.contains(op);
}

BufferDominationRange
BufferLifetimeInfo::getDominationRange(Operation *op) const {
  return dominationMap.at(op);
}

ErrorOr<BufferDominationRange>
BufferLifetimeInfo::getDominationRange(SmallVector<Operation *> ops) const {
  BufferDominationRange range = {std::numeric_limits<int>::max(),
                                 std::numeric_limits<int>::min()};
  assert(!ops.empty());
  Block *opBlock = ops[0]->getBlock();
  for (auto op : ops) {
    if (opBlock != op->getBlock())
      return Error("different blocks");
    if (!hasDominationRange(op))
      return Error("no domination info");
    range = range.overlap(getDominationRange(op));
  }

  return range;
}

ErrorOr<BufferDominationRange>
BufferLifetimeInfo::getAllocUseRange(MGP::BufferAllocDynamicOp allocOp) const {
  auto uses = getAllocUses(allocOp);
  if (uses.empty())
    return BufferDominationRange{std::numeric_limits<int>::max(),
                                 std::numeric_limits<int>::min()};

  return getDominationRange(uses);
}

SmallVector<Operation *>
BufferLifetimeInfo::getAllocUses(MGP::BufferAllocDynamicOp allocOp) const {
  // Determine the uses of the allocated buffer by searching through all direct
  // uses as well as those of possible aliases

  SmallVector<Operation *> uses;
  SmallVector<Operation *> stack;

  stack.push_back(allocOp);

  while (!stack.empty()) {
    auto top = stack.back();
    stack.pop_back();

    for (auto &use : top->getUses()) {
      if (!isa<MGP::TensorExtractTensorSpecOp>(*use.getOwner()))
        uses.push_back(use.getOwner());

      const bool isBufferOperand =
          isa<MGP::DeviceBufferType>(use.get().getType());

      if (isBufferOperand &&
          isa<GRT::GRTMayAliasInterface, GRT::GRTExtendsLifetimeInterface>(
              *use.getOwner()))
        stack.push_back(use.getOwner());
    }
  }
  return uses;
}

BufferLifetimeAnalysisPass::BufferLifetimeAnalysisPass(Operation *op) {
  auto modelOp = llvm::dyn_cast<ModelOp>(*op);

  ASSERT_STREAM(modelOp,
                "BufferLifetimeAnalysisPass can only run on an MGP::ModelOp");

  modelOp.walk([&](Block *block) {
    lifetimeInfo.calculateDataflowPostdominance(block->getTerminator());
  });
}
}; // namespace M::MGP
