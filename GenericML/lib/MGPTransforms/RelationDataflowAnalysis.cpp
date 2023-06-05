//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "./RelationDataflowAnalysis.h"

#include "GraphRT/GRTDialect/GRTOps.h"
#include "GraphRT/MEFDialect/MEFNonStrict.h"
#include "GraphRT/MEFDialect/MEFOps.h"
#include "llvm/ADT/EquivalenceClasses.h"

#include "./Utils.h"

using namespace M;
using namespace M::MGP;

#define DEBUG_TYPE "mgp-dataflow"

//===----------------------------------------------------------------------===//
// ForwardsRelationDataflowAnalysis
//===----------------------------------------------------------------------===//

void ForwardsRelationDataflowAnalysis::flowSequential(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowSubBlock &flowSubBlock) {
  for (size_t i = 0, n = cfOp->getNumRegions(); i < n; ++i) {
    Region &region = cfOp->getRegion(i);
    Block *subBlock = &region.front();
    if (i == 0) {
      // Flow from the current control flow op results back around to the first
      // sub-block arguments.
      for (const auto &[result, arg] :
           llvm::zip(cfOp->getResults(), subBlock->getArguments()))
        setEntriesAndId(domainMap, arg, getEntries(domainMap, result));
    } else {
      Block *prevSubBlock = &cfOp->getRegion(i - 1).front();
      auto prevOutputOp = cast<MEF::OutputOp>(prevSubBlock->getTerminator());
      // Flow from previous sub-block outputs forwards to current sub-block
      // arguments.
      for (auto [prevOutput, arg] :
           llvm::zip(prevOutputOp->getOperands(), subBlock->getArguments()))
        setEntriesAndId(domainMap, arg, getEntries(domainMap, prevOutput));
    }

    // Flow the sub block.
    flowSubBlock(domainMap, subBlock);
  }
}

void ForwardsRelationDataflowAnalysis::flowParallel(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowSubBlock &flowSubBlock) {
  for (Region &region : cfOp->getRegions()) {
    Block *subBlock = &region.front();
    // Flow from control flow op inputs forwards to sub-block args.
    for (auto [input, arg] :
         llvm::zip(cfOp.getInputs(), subBlock->getArguments()))
      setEntriesAndId(domainMap, arg, getEntries(domainMap, input));
    flowSubBlock(domainMap, subBlock);
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
void ForwardsRelationDataflowAnalysis::flowControlFlow(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowSubBlock &flowSubBlock) {
  switch (cfOp.getFlavor()) {
  case GRT::kLoopLike: {
    bool outerChanged = changed;

    // The first loop uses the control flow op inputs as the control flow
    // op results.
    for (auto [input, result] : llvm::zip(cfOp.getInputs(), cfOp->getResults()))
      setEntriesAndId(domainMap, result, getEntries(domainMap, input));

    // Until fixed point...
    LLVM_DEBUG(llvm::dbgs() << "till fixed point...\n");
    do {
      changed = false;

      // Flow each sub-block in 'series'.
      flowSequential(cfOp, domainMap, flowSubBlock);

      // Flow from the intersection of control flow op results and the last
      // sub-block outputs forwards to control flom op results.
      Block *lastSubBlock = &cfOp->getRegions().back().front();
      auto lastOutputOp = cast<MEF::OutputOp>(lastSubBlock->getTerminator());
      for (size_t i = 0, n = cfOp.getInputs().size(); i < n; ++i) {
        Value result = cfOp->getResult(i);
        Value lastOutput = lastOutputOp->getOperand(i);
        Set set = getEntries(domainMap, lastOutput);
        domainMap.setIntersection(set, getEntries(domainMap, result));
        setEntriesAndId(domainMap, result, set);
      }

      outerChanged |= changed;
      LLVM_DEBUG(if (changed) llvm::dbgs() << "... repeat\n");
    } while (changed);
    changed = outerChanged;
    break;
  }
  case GRT::kBranchLike: {
    // Flow each sub-block in 'parallel'.
    flowParallel(cfOp, domainMap, flowSubBlock);

    // Flow from intersection of the outputs of each sub-block forwards to the
    // control flow results.
    for (size_t i = 0, n = cfOp->getNumResults(); i < n; ++i) {
      Value result = cfOp->getResult(i);
      Set theIntersection = domainMap.fullSet;
      for (Region &region : cfOp->getRegions()) {
        Block *subBlock = &region.front();
        auto outputOp = cast<MEF::OutputOp>(subBlock->getTerminator());
        Value output = outputOp->getOperand(i);
        domainMap.setIntersection(theIntersection,
                                  getEntries(domainMap, output));
      }
      setEntriesAndId(domainMap, result, theIntersection);
    }
    break;
  }
  }
}

void ForwardsRelationDataflowAnalysis::flowDefault(Operation *op,
                                                   DomainMap &domainMap) {
  for (Value result : op->getResults())
    setEntriesAndId(domainMap, result, domainMap.emptySet);
}

void ForwardsRelationDataflowAnalysis::flowModel(
    ModelOp modelOp, DomainMap &domainMap, const FlowSubBlock &flowSubBlock) {
  for (Region &region : modelOp->getRegions()) {
    for (Value arg : region.front().getArguments())
      setEntriesAndId(domainMap, arg, domainMap.emptySet);
    flowSubBlock(domainMap, &region.front());
  }
}
