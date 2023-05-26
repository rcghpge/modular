//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "./AsyncDataflowDependencies.h"

#include "GraphRT/GRTDialect/GRTOps.h"
#include "GraphRT/MEFDialect/MEFNonStrict.h"
#include "GraphRT/MEFDialect/MEFOps.h"

using namespace M;
using namespace M::MGP;

/// TODO: Rewrite this to build the predicates/relations directly instead
/// of implementing the pointwise test methods. This will both increase
/// completeness (for isAlwaysReady) and avoid inf recursion.

/// Implements the following closure rules for control flow ops, where R is
/// the relation under definition. Note that inductions is only on the lhs
/// argument, and that the relation may be recursively invoked with a different
/// lhs and rhs. Returns std::nullopt if no rules fired.
///
/// Rules:
///  - [A] Given
///      ... grt.if ... (... %lhs ...) {
///        // %lhs bound as %lhs1
///      } else {
///        // %lhs bound as %lhs2
///      }
///    then R(%lhs{1,2}, %rhs) if R(%lhs, %rhs),
///  - [B] Given
///      ... %lhs ... = grt.if ... {
///        ...
///        mef.output ... %x1 ...
///      } else {
///        ...
///        mef.output ... %x2 ...
///      }
///    then R(%lhs, %rhs) if R(%x1, %rhs) and R(%x2, %rhs).
///  - [C] Given
///      ... grt.repeat.i32 ... (... %x ... ) {
///        // %x bound as %lhs
///        ...
///        mef.output ... %y ...
///      }
///    then R(%lhs, %rhs) if R(%x, %rhs) and R(%y, %lhs).
///  - [D] Given
///      ... %lhs ... = grt.repeat.i32 ... (... %x ...) {
//         // %x bound as %x1
///        ...
///        mef.output ... %y ...
///      }
///    then R(%lhs, %rhs) if R(%x, %rhs) and R(%y, %x1).
///  - [E] Given
///      ... grt.while (... %x ... ) {
///        // %x bound as %lhs
///        ...
///        mef.output ... %y1 ...
///      } do {
///        // %x bound as %x2
///        ...
///        mef.output ... %y2 ...
///      }
///    then R(%lhs, %rhs) if R(%x, %rhs) and R(%y2, %x2) and R(%y1, %lhs).
///  - [F] Given
///      ... grt.while (... %x ... ) {
///        // %x bound as %x1
///        ...
///        mef.output ... %y1 ...
///      } do {
///        // %x bound as %lhs
///        ...
///        mef.output ... %y2 ...
///      }
///    then R(%lhs, %rhs) if R(%x, %rhs) and R(%y2, %lhs) and R(%y1, %x).
///  - [G] Given
///      ... %lhs ... = grt.while (... %x ...) {
///        %x bound as %x1
///        ...
///        mef.output ... %y1 ...
///      } do {
///        %x bound as %x2
///        ...
///        mef.output ... %y2 ...
///      }
///    then R(%lhs, %rhs) if R(%x, %rhs) and R(%y1, %x1) and R(%y2, %x2).
///
/// Users of this helper will be responsible for implementing reflexive,
/// transitive and/or symmetric closure over these structural rules.
static std::optional<bool>
controlFlowRelation(Value lhs, Value rhs,
                    const std::function<bool(Value, Value)> &relation) {
  if (auto cfOp = lhs.getDefiningOp<GRT::GRTControlFlowInterface>()) {
    auto lhsResult = cast<OpResult>(lhs);
    switch (cfOp.getFlavor()) {
    case GRT::kLoopLike:
      // Rules [D] & [G].
      // We need the input corresponding to the result to satisfy the relation.
      // We also need each sub-block to satisfy the relation from corresponding
      // output to corresponding input.
      return relation(cfOp.getInputs()[lhsResult.getResultNumber()], rhs) &&
             llvm::all_of(cfOp->getRegions(), [&](Region &region) {
               Block &block = region.getBlocks().front();
               auto outputOp = cast<MEF::OutputOp>(block.getTerminator());
               // NOTE: Second argument is not rhs in this recursive call!
               return relation(
                   outputOp->getOperand(lhsResult.getResultNumber()),
                   block.getArgument(lhsResult.getResultNumber()));
             });
    case GRT::kBranchLike:
      // Rule [B].
      // We need ALL the corresponding outputs from each sub-block to satisfy
      // the relation.
      return llvm::all_of(cfOp->getRegions(), [&](Region &region) {
        Block &block = region.getBlocks().front();
        auto outputOp = cast<MEF::OutputOp>(block.getTerminator());
        return relation(outputOp->getOperand(lhsResult.getResultNumber()), rhs);
      });
    }
  }

  if (lhs.getDefiningOp())
    // Result of arbitrary op is not handled by these rules.
    return std::nullopt;

  auto lhsArg = cast<BlockArgument>(lhs);
  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(
          lhsArg.getOwner()->getParentOp())) {
    size_t inputIndex = lhsArg.getArgNumber();
    switch (cfOp.getFlavor()) {
    case GRT::kLoopLike:
      // Rule [C] & [E] & [F].
      // We need the corresponding input operand to satisfy the relation.
      // We also need each sub-block to satisfy the relation from corresponding
      // output to corresponding input. (One of those blocks will be binding
      // lhsArg, but we don't care which one.)
      return relation(cfOp.getInputs()[inputIndex], rhs) &&
             llvm::all_of(cfOp->getRegions(), [&](Region &region) {
               Block &block = region.getBlocks().front();
               auto outputOp = cast<MEF::OutputOp>(block.getTerminator());
               // NOTE: Second argument is not rhs in this recursive call!
               return relation(outputOp->getOperand(inputIndex),
                               block.getArgument(inputIndex));
             });
    case GRT::kBranchLike:
      // Rule [A].
      // We need the corresponding input operand to satisfy the relation.
      return relation(cfOp.getInputs()[inputIndex], rhs);
    }
  }

  // Top-level block var is not handled by these rules.
  return std::nullopt;
}

/// The sameAs rules:
///  - [A] Reflexive
///  - [B] Symmetric
///  - The control flow op rules from above.
bool AsyncDataflowDependencies::sameAs(Value lhs, Value rhs) const {
  // Remember the result for these args.
  auto memoize = [this, lhs, rhs](bool result) {
    sameAsCache.insert({{lhs, rhs}, result});
    // Rule [B].
    sameAsCache.insert({{rhs, lhs}, result});
    return result;
  };

  if (lhs == rhs)
    // Rule [A]
    return true;

  auto itr = sameAsCache.find({lhs, rhs});
  if (itr != sameAsCache.end())
    return itr->second;

  // Try control flow rules inducting on lhs.
  std::optional<bool> optControlFlowResultLhs =
      controlFlowRelation(lhs, rhs, [&](Value subLhs, Value subRhs) {
        return sameAs(subLhs, subRhs);
      });
  if (optControlFlowResultLhs.has_value())
    return memoize(*optControlFlowResultLhs);

  // Try control flow rules inducting on rhs, for rule [B].
  std::optional<bool> optControlFlowResultRhs =
      controlFlowRelation(rhs, lhs, [&](Value subLhs, Value subRhs) {
        return sameAs(subLhs, subRhs);
      });
  if (optControlFlowResultRhs.has_value())
    return memoize(*optControlFlowResultRhs);

  return memoize(false);
}

/// The isAlwaysReady rules:
///  - %x = grt.chain.create implies isAlwaysReady(%x)
///  - %x = grt.chain.merge (%y1, ..., %yn) and isAlwaysReady(%yi) for all i
///    implies isAlwaysReady(%x)
///
/// It would be nice to also have the closure rule that sameAs(%x, %y) and
/// isAlwaysReady(%y) implies isAlwaysReady(%x), but that would require
/// reifying the sameAs equivalence class or switching to a dataflow analysis
/// and don't think it's worth it.
bool AsyncDataflowDependencies::isAlwaysReady(Value value) const {
  // Remember the result for this arg.
  auto memoize = [this, value](bool result) {
    isAlwaysReadyCache.insert({value, result});
    return result;
  };

  auto itr = isAlwaysReadyCache.find(value);
  if (itr != isAlwaysReadyCache.end())
    return itr->second;

  if (value.getDefiningOp<GRT::ChainCreateOp>())
    return memoize(true);

  if (auto mergeOp = value.getDefiningOp<GRT::ChainMergeOp>()) {
    bool allTrue = llvm::all_of(mergeOp.getOperands(), [this](Value operand) {
      return isAlwaysReady(operand);
    });
    return memoize(allTrue);
  }

  // TODO: Here's where would like to implement
  //   any_of { isAlwaysReady(x) | sameAs(value, x) }
  // which would be possible if we switched to calculating the sameAs
  // relation all-at-once instead of pointwise.

  return memoize(false);
}

/// Given
///   ..., %lhs, ... = op(..., %rhs, ...)
/// returns true if valueOf(%lhs)->isReady() => valueOf(%rhs)->isReady()
/// The rule is:
///  - some.op is strict, or
///  - both %x and %y are of type !mef.chain.
static bool primitiveImpliesReady(Value lhs, Value rhs) {
  assert(lhs.getDefiningOp());
  if (!lhs.getDefiningOp<MEF::NonStrictOp>())
    // op is strict
    return true;

  if (isa<MEF::ChainType>(lhs.getType()) && isa<MEF::ChainType>(rhs.getType()))
    // op is non-strict, but assume chain inputs and outputs are strict
    return true;

  return false;
}

/// The impliesReady rules:
///  - [A] If sameAs(%lhs, %rhs) then impliesReady(%lhs, %rhs)
///  - [B] Transitive
///  - [C] If isAlwaysReady(%rhs) then impliesReady(%lhs, %rhs)
///  - [D] If primitiveImpliesReady(%lhs, %rhs) then impliesReady(%lhs, %rhs)
///  - The control flow op rules from above.
bool AsyncDataflowDependencies::impliesReady(Value lhs, Value rhs) const {
  // Remember the result for these args.
  auto memoize = [this, lhs, rhs](bool result) {
    impliesReadyCache.insert({{lhs, rhs}, result});
    return result;
  };

  if (isAlwaysReady(rhs))
    // Rule [C]
    return memoize(true);

  /// We'll use an explicit work item stack to track the dfs for Rule [B].
  /// However we'll also be making recursive calls to handle loops since the
  /// nesting of control flow ops is shallow (and capturing the continuations
  /// in the work items is ugly).
  SmallVector<Value> workItems;

  workItems.emplace_back(lhs);
  while (!workItems.empty()) {
    Value curr = workItems.back();
    workItems.pop_back();

    if (sameAs(curr, rhs))
      // Rule [A]
      return memoize(true);

    auto itr = impliesReadyCache.find({curr, rhs});
    if (itr != impliesReadyCache.end()) {
      if (itr->second) {
        // Success by memoization.
        return memoize(true);
      }
      // Dead end, keep searching.
      continue;
    }

    std::optional<bool> optControlFlowResult =
        controlFlowRelation(curr, rhs, [&](Value subLhs, Value subRhs) {
          return impliesReady(subLhs, subRhs);
        });
    if (optControlFlowResult.has_value()) {
      if (*optControlFlowResult) {
        // Success by control flow rules.
        return memoize(true);
      }
      // Dead end, keep searching.
      continue;
    }

    if (Operation *currOp = curr.getDefiningOp()) {
      // Rule [D].
      // Any of the operands may help satisfy the relation.
      for (Value operand : currOp->getOperands()) {
        if (primitiveImpliesReady(curr, operand))
          workItems.emplace_back(operand);
      }
      // Pushed new items, keep searching.
      continue;
    }

    // Top-level block vars are dead ends, so keep searching.
  }
  return memoize(false);
}

void AsyncDataflowDependencies::simplify(SmallVectorImpl<Value> &values) const {
  if (values.size() <= 1)
    return;
  for (size_t ip1 = values.size(); ip1 > 0; --ip1) {
    bool redundant = false;
    for (size_t j = 0; j < values.size(); ++j) {
      if (j == ip1 - 1)
        continue;
      if (impliesReady(values[j], values[ip1 - 1])) {
        redundant = true;
        break;
      }
    }
    if (redundant)
      values.erase(values.begin() + ip1 - 1);
  }
}
