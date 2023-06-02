//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "./AsyncDataflowDependencies.h"

#include "GraphRT/GRTDialect/GRTOps.h"
#include "GraphRT/MEFDialect/MEFNonStrict.h"
#include "GraphRT/MEFDialect/MEFOps.h"
#include "llvm/ADT/EquivalenceClasses.h"

#include "./Utils.h"

using namespace M;
using namespace M::MGP;

#define DEBUG_TYPE "asyncdataflowdeps"

AsyncDataflowDependencies::AsyncDataflowDependencies(ModelOp model)
    : index(model) {
  sameAsEquivalence = index.identityEquivalence();
  setupSameAs(model);
  setupAlwaysReady(model);
  setupImpliesReady(model);
}

bool AsyncDataflowDependencies::sameAs(Value lhs, Value rhs) const {
  return index.areEquivalent(sameAsEquivalence, lhs, rhs);
}

bool AsyncDataflowDependencies::isAlwaysReady(Value value) const {
  return containsDistinguished(alwaysReadyRelation, value);
}

bool AsyncDataflowDependencies::impliesReady(Value lhs, Value rhs) const {
  return containsEntryOrId(impliesReadyRelation, lhs, rhs);
}

void AsyncDataflowDependencies::simplify(SmallVectorImpl<Value> &values) const {
  for (size_t ip1 = values.size(); ip1 > 0; --ip1) {
    size_t i = ip1 - 1;
    bool redundant = isAlwaysReady(values[i]);
    if (!redundant) {
      for (size_t j = 0; j < values.size(); ++j) {
        if (j == i)
          continue;
        if (impliesReady(values[j], values[i])) {
          redundant = true;
          break;
        }
      }
    }
    if (redundant)
      values.erase(values.begin() + i);
  }
}

void AsyncDataflowDependencies::setEntriesAndId(Relation &relation, Value value,
                                                Set set) {
  value = index.getCanonical(sameAsEquivalence, value);
  index.addToSet(set, value);
  LLVM_DEBUG(llvm::dbgs() << "setEntriesAndId(";
             index.printValue(llvm::dbgs(), value); llvm::dbgs() << ", {";
             llvm::interleaveComma(
                 index.setToVector(set), llvm::dbgs(),
                 [&](Value value) { index.printValue(llvm::dbgs(), value); });
             llvm::dbgs() << "})\n");

  auto [itr, inserted] = relation.insert({value, set});
  if (inserted) {
    changed = true;
  } else {
    assert(index.setContains(itr->second, value));
    if (itr->second != set) {
      LLVM_DEBUG(
          llvm::dbgs() << "not equal to existing set {"; llvm::interleaveComma(
              index.setToVector(itr->second), llvm::dbgs(),
              [&](Value value) { index.printValue(llvm::dbgs(), value); });
          llvm::dbgs() << "})\n");
      itr->second = set;
      changed = true;
    }
  }
}

AsyncDataflowDependencies::Set
AsyncDataflowDependencies::getEntries(const Relation &relation,
                                      Value value) const {
  value = index.getCanonical(sameAsEquivalence, value);
  auto itr = relation.find(value);
  return itr != relation.end() ? itr->second : Set();
}

bool AsyncDataflowDependencies::containsDistinguished(const Relation &relation,
                                                      Value value) const {
  value = index.getCanonical(sameAsEquivalence, value);
  auto itr = relation.find(value);
  return itr != relation.end() &&
         index.setContains(itr->second, index.distinguished);
}

bool AsyncDataflowDependencies::containsEntryOrId(const Relation &relation,
                                                  Value lhs, Value rhs) const {
  lhs = index.getCanonical(sameAsEquivalence, lhs);
  rhs = index.getCanonical(sameAsEquivalence, rhs);
  if (lhs == rhs)
    return true;
  auto itr = relation.find(lhs);
  return itr != relation.end() && index.setContains(itr->second, rhs);
}

AsyncDataflowDependencies::Set
AsyncDataflowDependencies::intersection(const SmallVector<Set> &sets) {
  if (sets.empty())
    return index.emptySet();
  Set result = sets.front();
  for (const Set &set : llvm::drop_begin(sets))
    index.setIntersect(result, set);
  return result;
}

void AsyncDataflowDependencies::accumSequential(
    GRT::GRTControlFlowInterface cfOp, Relation &relation,
    const AccumSubBlock &accumSubBlock, const ValueRange &inputsOrResults) {
  for (size_t i = 0, n = cfOp->getNumRegions(); i < n; ++i) {
    Region &region = cfOp->getRegion(i);
    Block *subBlock = &region.front();
    if (i == 0) {
      // Connect sub-block args back to the control flow inputs/results.
      for (auto [inputOrResult, arg] :
           llvm::zip(inputsOrResults, subBlock->getArguments()))
        setEntriesAndId(relation, arg, getEntries(relation, inputOrResult));
    } else {
      Block *prevSubBlock = &cfOp->getRegion(i - 1).front();
      auto prevOutputOp = cast<MEF::OutputOp>(prevSubBlock->getTerminator());
      // Connect sub-block args back to the results of the previous
      // sub-block.
      for (auto [prevRes, arg] :
           llvm::zip(prevOutputOp->getOperands(), subBlock->getArguments()))
        setEntriesAndId(relation, arg, getEntries(relation, prevRes));
    }
    // Collection relations from sub-block.
    accumSubBlock(relation, subBlock);
  }
}

void AsyncDataflowDependencies::accumParallel(
    GRT::GRTControlFlowInterface cfOp, Relation &relation,
    const AccumSubBlock &accumSubBlock) {
  for (Region &region : cfOp->getRegions()) {
    Block *subBlock = &region.front();
    // Connect sub-block args back to the control flow inputs.
    for (auto [input, arg] :
         llvm::zip(cfOp.getInputs(), subBlock->getArguments()))
      setEntriesAndId(relation, arg, getEntries(relation, input));
    accumSubBlock(relation, &region.front());
  }
}

/// Implements the following rules for control flow ops, where R is
/// the relation under definition.
///
/// Rules:
///  - Given
///      ... grt.if ... (... %lhs ...) {
///        // %lhs bound as %lhs1
///      } else {
///        // %lhs bound as %lhs2
///      }
///    then R(%lhs{1,2}, %rhs) if R(%lhs, %rhs),
///  - Given
///      ... %lhs ... = grt.if ... {
///        ...
///        mef.output ... %x1 ...
///      } else {
///        ...
///        mef.output ... %x2 ...
///      }
///    then R(%lhs, %rhs) if R(%x1, %rhs) and R(%x2, %rhs).
///  - Given
///      ... grt.repeat.i32 ... (... %x ... ) {
///        // %x bound as %lhs
///        ...
///        mef.output ... %y ...
///      }
///    then R(%lhs, %rhs) if R(%x, %rhs) and R(%y, %lhs).
///  - Given
///      ... %lhs ... = grt.repeat.i32 ... (... %x ...) {
//         // %x bound as %x1
///        ...
///        mef.output ... %y ...
///      }
///    then R(%lhs, %rhs) if R(%x, %rhs) and R(%y, %x1).
///  - Given
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
///  - Given
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
///  - Given
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
void AsyncDataflowDependencies::accumControlFlow(
    GRT::GRTControlFlowInterface cfOp, Relation &relation,
    const AccumSubBlock &accumSubBlock) {
  switch (cfOp.getFlavor()) {
  case GRT::kLoopLike: {
    bool outerChanged = changed;
    SmallVector<Value> inputsOrResults = cfOp.getInputs();
    // Until fixed point...
    LLVM_DEBUG(llvm::dbgs() << "till fixed point...\n");
    do {
      changed = false;
      // Accumulate relations from each sub-block in 'series',
      // starting from either the control flow inputs (first loop) or the
      // current control flow outputs (subsequent loops).
      accumSequential(cfOp, relation, accumSubBlock, inputsOrResults);
      Block *lastSubBlock = &cfOp->getRegions().back().front();
      auto lastOutputOp = cast<MEF::OutputOp>(lastSubBlock->getTerminator());
      // Connect the control flow result back to the intersection of the last
      // sub-block's output and the control flow inputs.
      for (const auto &[input, lastResult, lhs] :
           llvm::zip(cfOp.getInputs(), lastOutputOp->getOperands(),
                     cfOp->getResults())) {
        setEntriesAndId(relation, lhs,
                        intersection({getEntries(relation, lastResult),
                                      getEntries(relation, input)}));
      }
      // Subsequent loops feed back from control flow op results.
      inputsOrResults = cfOp->getResults();
      outerChanged |= changed;
      LLVM_DEBUG(if (changed) llvm::dbgs() << "... repeat\n");
    } while (changed);
    changed = outerChanged;
    break;
  }
  case GRT::kBranchLike: {
    // Accumulate relation entries from each sub-block in 'parallel'.
    accumParallel(cfOp, relation, accumSubBlock);
    // Accumulate relations for the control flow results.
    for (size_t i = 0; i < cfOp->getNumResults(); ++i) {
      Value lhs = cfOp->getResult(i);
      // Connect the control flow result back to the intersection
      // over all the sub-block results.
      SmallVector<Set> sets;
      for (Region &region : cfOp->getRegions()) {
        Block *subBlock = &region.front();
        auto outputOp = cast<MEF::OutputOp>(subBlock->getTerminator());
        // Connect the result back to the block output.
        Value rhs = outputOp->getOperand(i);
        sets.emplace_back(getEntries(relation, rhs));
      }
      setEntriesAndId(relation, lhs, intersection(sets));
    }
    break;
  }
  }
}

// Rules are just the control flow rules from above.
void AsyncDataflowDependencies::accumPreSameAs(Relation &relation,
                                               Block *block) {
  for (Operation *op : getDataflowOrder(block)) {
    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
      accumControlFlow(cfOp, relation,
                       [&](Relation &subRelation, Block *subBlock) {
                         return accumPreSameAs(subRelation, subBlock);
                       });
    } else {
      for (Value result : op->getResults())
        setEntriesAndId(relation, result, index.emptySet());
    }
  }
}

void AsyncDataflowDependencies::setupSameAs(ModelOp modelOp) {
  Relation relation;

  for (Region &region : modelOp->getRegions()) {
    for (Value arg : region.front().getArguments())
      setEntriesAndId(relation, arg, index.emptySet());
    accumPreSameAs(relation, &region.front());
  }

  // Close under reflexivity, symmetry & transitivity.
  for (const auto &[lhs, set] : relation) {
    for (Value rhs : index.setToVector(set))
      index.setEquivalent(sameAsEquivalence, lhs, rhs);
  }
}

/// Implements the rules:
///  - Given
///      %x = grt.chain.create
///    then isAlwaysReady(%x)
///  - Given
///      %x = grt.chain.merge (%y1, ..., %yn)
///    where isAlwaysReady(%yi) for all i then isAlwaysReady(%x)
///  - The control flow rules from above, using the encoding
///       (x, distinguished) in the relation iff isAlwaysReady(x).
///  - Closure under sameAs.
void AsyncDataflowDependencies::accumAlwaysReady(Relation &relation,
                                                 Block *block) {
  for (Operation *op : getDataflowOrder(block)) {
    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
      accumControlFlow(cfOp, relation,
                       [&](Relation &subRelation, Block *subBlock) {
                         return accumAlwaysReady(subRelation, subBlock);
                       });
    } else if (auto createOp = dyn_cast<GRT::ChainCreateOp>(op)) {
      setEntriesAndId(relation, createOp.getResult(),
                      index.unitSet(index.distinguished));
    } else if (auto mergeOp = dyn_cast<GRT::ChainMergeOp>(op)) {
      // This is equivalent to:
      //   if (containsDistinguished(o) for all o in mergeOp.getOperands())
      //     setDistinguished(mergeOp.getResult())
      // but writen to mirror how accumControlFlow works with the relation.
      Set theIntersection = index.emptySet();
      index.addToSet(theIntersection, index.distinguished);
      for (Value operand : mergeOp.getOperands())
        index.setIntersect(theIntersection, getEntries(relation, operand));
      setEntriesAndId(relation, mergeOp.getResult(), theIntersection);
    } else {
      for (Value result : op->getResults())
        setEntriesAndId(relation, result, index.emptySet());
    }
  }
}

void AsyncDataflowDependencies::setupAlwaysReady(ModelOp modelOp) {
  for (Region &region : modelOp->getRegions()) {
    for (Value arg : region.front().getArguments())
      setEntriesAndId(alwaysReadyRelation, arg, index.emptySet());
    accumAlwaysReady(alwaysReadyRelation, &region.front());
  }
}

/// Given
///   ..., %lhs, ... = op(..., %rhs, ...)
/// returns true if
///  - some.op is strict, or
///  - both %x and %y are of type !mef.chain.
bool AsyncDataflowDependencies::primitiveImpliesReady(Value lhs, Value rhs) {
  assert(lhs.getDefiningOp());
  if (!lhs.getDefiningOp<MEF::NonStrictOp>())
    // op is strict
    return true;

  if (isa<MEF::ChainType>(lhs.getType()) && isa<MEF::ChainType>(rhs.getType()))
    // op is non-strict, but assume chain inputs and outputs are strict
    return true;

  return false;
}

/// Implements the rules:
///  - primitiveImpliesReady(%lhs, %rhs) implies impliesReady(%lhs, %rhs)
///  - the control flow rules from above
///  - closure under sameAs
void AsyncDataflowDependencies::accumImpliesReady(Relation &relation,
                                                  Block *block) {
  for (Operation *op : getDataflowOrder(block)) {
    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
      accumControlFlow(cfOp, relation,
                       [&](Relation &subRelation, Block *subBlock) {
                         return accumImpliesReady(subRelation, subBlock);
                       });
    } else {
      for (Value lhs : op->getResults()) {
        Set theUnion = index.emptySet();
        for (Value rhs : op->getOperands()) {
          // Primitive relation relation.
          if (primitiveImpliesReady(lhs, rhs))
            index.setUnion(theUnion, getEntries(relation, rhs));
        }
        setEntriesAndId(relation, lhs, theUnion);
      }
    }
  }
}

void AsyncDataflowDependencies::setupImpliesReady(ModelOp modelOp) {
  for (Region &region : modelOp->getRegions()) {
    for (Value arg : region.front().getArguments())
      setEntriesAndId(impliesReadyRelation, arg, index.emptySet());
    accumImpliesReady(impliesReadyRelation, &region.front());
  }
}
