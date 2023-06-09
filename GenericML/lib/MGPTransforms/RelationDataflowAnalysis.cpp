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

void ForwardsRelationDataflowAnalysis::flowSimpleLoop(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  assert(cfOp->getNumRegions() == 1);
  Block *bodyBlock = &cfOp->getRegion(0).front();

  // Flow from the current control flow op results back around to the
  // sub-block arguments.
  for (const auto &[result, arg] :
       llvm::zip(cfOp->getResults(), bodyBlock->getArguments()))
    setEntriesAndId(domainMap, arg, getEntries(domainMap, result));

  // Flow the sub block.
  flowBlock(bodyBlock, domainMap, flowOp);

  // Flow from the sub-block outputs forwards to control flow op results, but
  // intersect with existing results.
  auto outputOp = cast<MEF::OutputOp>(bodyBlock->getTerminator());
  for (const auto [output, result] :
       llvm::zip(outputOp->getOperands(), cfOp->getResults())) {
    Set theIntersection = getEntries(domainMap, output);
    domainMap.setIntersection(theIntersection, getEntries(domainMap, result));
    setEntriesAndId(domainMap, result, theIntersection);
  }
}

void ForwardsRelationDataflowAnalysis::flowCondition(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  assert(cfOp->getNumRegions() == 2);
  Block *condBlock = &cfOp->getRegion(0).front();

  // Flow from the control flow op inputs to the
  // sub-block arguments.
  for (const auto &[input, arg] :
       llvm::zip(cfOp.getInputs(), condBlock->getArguments()))
    setEntriesAndId(domainMap, arg, getEntries(domainMap, input));

  // Flow the sub block.
  flowBlock(condBlock, domainMap, flowOp);

  // Flow from the sub-block outputs forwards to control flow op results.
  auto outputOp = cast<MEF::OutputOp>(condBlock->getTerminator());
  for (const auto [output, result] :
       llvm::zip(outputOp->getOperands(), cfOp->getResults()))
    setEntriesAndId(domainMap, result, getEntries(domainMap, output));
}

void ForwardsRelationDataflowAnalysis::flowConditionedLoop(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  assert(cfOp->getNumRegions() == 2);
  Block *condBlock = &cfOp->getRegion(0).front();
  Block *bodyBlock = &cfOp->getRegion(1).front();

  // Flow from the control flow op results to the body sub-block arguments.
  for (const auto &[result, arg] :
       llvm::zip(cfOp->getResults(), bodyBlock->getArguments()))
    setEntriesAndId(domainMap, arg, getEntries(domainMap, result));

  // Flow the body sub block.
  flowBlock(bodyBlock, domainMap, flowOp);

  // Flow from the body sub-block outputs forwards to the cond sub-block
  // arguments.
  auto bodyOutputOp = cast<MEF::OutputOp>(bodyBlock->getTerminator());
  for (const auto [output, arg] :
       llvm::zip(bodyOutputOp->getOperands(), condBlock->getArguments()))
    setEntriesAndId(domainMap, arg, getEntries(domainMap, output));

  // Flow the cond sub block.
  flowBlock(condBlock, domainMap, flowOp);

  // Flow from the cond sub-block outputs forwards to control flow op results,
  // but intersect with existing results.
  auto condOutputOp = cast<MEF::OutputOp>(condBlock->getTerminator());
  for (const auto [output, result] :
       llvm::zip(condOutputOp->getOperands(), cfOp->getResults())) {
    Set theIntersection = getEntries(domainMap, output);
    domainMap.setIntersection(theIntersection, getEntries(domainMap, result));
    setEntriesAndId(domainMap, result, theIntersection);
  }
}

void ForwardsRelationDataflowAnalysis::flowBranch(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  // Flow from control flow op inputs forwards into each sub-block's args.
  for (Region &region : cfOp->getRegions()) {
    Block *subBlock = &region.front();
    for (auto [input, arg] :
         llvm::zip(cfOp.getInputs(), subBlock->getArguments()))
      setEntriesAndId(domainMap, arg, getEntries(domainMap, input));
    flowBlock(subBlock, domainMap, flowOp);
  }

  // Flow from intersection of the outputs of each sub-block forwards to the
  // control flow op results.
  for (size_t i = 0, n = cfOp->getNumResults(); i < n; ++i) {
    Value result = cfOp->getResult(i);
    Set theIntersection = domainMap.fullSet;
    for (Region &region : cfOp->getRegions()) {
      Block *subBlock = &region.front();
      auto outputOp = cast<MEF::OutputOp>(subBlock->getTerminator());
      Value output = outputOp->getOperand(i);
      domainMap.setIntersection(theIntersection, getEntries(domainMap, output));
    }
    setEntriesAndId(domainMap, result, theIntersection);
  }
}

/// Implements the following rules for control flow ops, where R is
/// the relation under definition.
///
/// Rules:
///  - Given
///      ... grt.if ... (... %lhs ...) {
///        // %lhs bound as %lhs1
///        ...
///      } else {
///        // %lhs bound as %lhs2
///        ...
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
///    then R(%lhs, %rhs) if R(%x, %rhs) and R(%y2, %lhs) and R(%y1, %x1).
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
    const FlowOp &flowOp) {
  switch (cfOp.getFlavor()) {
  case GRT::kLoopLike: {
    bool outerChanged = changed;
    if (cfOp->getNumRegions() == 1) {
      // The sub-block may be executed zero times, so flow forwards from
      // inputs to results.
      for (auto [input, result] :
           llvm::zip(cfOp.getInputs(), cfOp->getResults()))
        setEntriesAndId(domainMap, result, getEntries(domainMap, input));
    } else {
      // The first sub-block will be executed at least once.
      flowCondition(cfOp, domainMap, flowOp);
    }

    // Until fixed point...
    LLVM_DEBUG(llvm::dbgs() << "till fixed point...\n");
    do {
      changed = false;

      if (cfOp->getNumRegions() == 1) {
        // The sub-block may be executed again.
        flowSimpleLoop(cfOp, domainMap, flowOp);
      } else {
        // The second sub-block then first-sub block may be executed again.
        flowConditionedLoop(cfOp, domainMap, flowOp);
      }

      outerChanged |= changed;
      LLVM_DEBUG(if (changed) llvm::dbgs() << "... repeat\n");
    } while (changed);
    changed = outerChanged;
    break;
  }
  case GRT::kBranchLike: {
    // Flow each sub-block.
    flowBranch(cfOp, domainMap, flowOp);
    break;
  }
  }
}

void ForwardsRelationDataflowAnalysis::flowDefault(Operation *op,
                                                   DomainMap &domainMap) {
  for (Value result : op->getResults())
    setEntriesAndId(domainMap, result, domainMap.emptySet);
}

void ForwardsRelationDataflowAnalysis::flowBlock(Block *block,
                                                 DomainMap &domainMap,
                                                 const FlowOp &flowOp) {
  for (Operation *op : getDataflowOrder(block))
    flowOp(op, domainMap);
}

void ForwardsRelationDataflowAnalysis::flowModel(ModelOp modelOp,
                                                 DomainMap &domainMap,
                                                 const FlowOp &flowOp) {
  for (Region &region : modelOp->getRegions()) {
    for (Value arg : region.front().getArguments())
      setEntriesAndId(domainMap, arg, domainMap.emptySet);
    flowBlock(&region.front(), domainMap, flowOp);
  }
}

//===----------------------------------------------------------------------===//
// BackwardsRelationDataflowAnalysis
//===----------------------------------------------------------------------===//

BackwardsRelationDataflowAnalysis::Set
BackwardsRelationDataflowAnalysis::operandIntersection(
    const DomainMap &domainMap, Value value) const {
  Set theIntersection = domainMap.fullSet;
  for (OpOperand &operand : value.getUses())
    domainMap.setIntersection(theIntersection, getEntries(domainMap, &operand));
  return theIntersection;
}

void BackwardsRelationDataflowAnalysis::flowSimpleLoop(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  assert(cfOp->getNumRegions() == 1);
  Block *bodyBlock = &cfOp->getRegion(0).front();

  // Flow from the current control flow op inputs back around to the
  // sub-block output operands.
  auto outputOp = cast<MEF::OutputOp>(bodyBlock->getTerminator());
  for (const auto &[input, output] :
       llvm::zip(cfOp.getInputOpOperands(), outputOp->getOpOperands()))
    setEntriesAndId(domainMap, &output, getEntries(domainMap, &input));

  // Flow the sub block.
  flowBlock(bodyBlock, domainMap, flowOp);

  // Flow from the sub-block arguments backwards to the control flow op
  // inputs, but intersect with the existing inputs.
  for (const auto &[input, arg] :
       llvm::zip(cfOp.getInputOpOperands(), bodyBlock->getArguments())) {
    Set theIntersection = operandIntersection(domainMap, arg);
    domainMap.setIntersection(theIntersection, getEntries(domainMap, &input));
    setEntriesAndId(domainMap, &input, theIntersection);
  }
}

void BackwardsRelationDataflowAnalysis::flowCondition(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  assert(cfOp->getNumRegions() == 2);
  Block *condBlock = &cfOp->getRegion(0).front();

  // Flow from the control flow op results to the sub-block outputs.
  auto outputOp = cast<MEF::OutputOp>(condBlock->getTerminator());
  for (const auto &[result, output] :
       llvm::zip(cfOp->getResults(), outputOp->getOpOperands()))
    setEntriesAndId(domainMap, &output, operandIntersection(domainMap, result));

  // Flow the sub block.
  flowBlock(condBlock, domainMap, flowOp);

  // Flow from the sub-block arguments backwards to control flow op inputs.
  for (const auto &[input, arg] :
       llvm::zip(cfOp.getInputOpOperands(), condBlock->getArguments()))
    setEntriesAndId(domainMap, &input, operandIntersection(domainMap, arg));
}

void BackwardsRelationDataflowAnalysis::flowConditionedLoop(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  assert(cfOp->getNumRegions() == 2);
  Block *condBlock = &cfOp->getRegion(0).front();
  Block *bodyBlock = &cfOp->getRegion(1).front();

  // Flow from the control flow op inputs to the cond sub-block outputs.
  auto condOutputOp = cast<MEF::OutputOp>(condBlock->getTerminator());
  for (const auto &[output, input] :
       llvm::zip(condOutputOp->getOpOperands(), cfOp.getInputOpOperands()))
    setEntriesAndId(domainMap, &output, getEntries(domainMap, &input));

  // Flow the cond sub block.
  flowBlock(condBlock, domainMap, flowOp);

  // Flow from the cond sub-block args backwards to the body sub-block
  // outputs.
  auto bodyOutputOp = cast<MEF::OutputOp>(bodyBlock->getTerminator());
  for (const auto &[output, arg] :
       llvm::zip(bodyOutputOp->getOpOperands(), condBlock->getArguments()))
    setEntriesAndId(domainMap, &output, operandIntersection(domainMap, arg));

  // Flow the body sub block.
  flowBlock(bodyBlock, domainMap, flowOp);

  // Flow from the body sub-block args backwards to control flow op inputs,
  // but intersect with existing inputs.
  for (const auto &[input, arg] :
       llvm::zip(cfOp.getInputOpOperands(), bodyBlock->getArguments())) {
    Set theIntersection = operandIntersection(domainMap, arg);
    domainMap.setIntersection(theIntersection, getEntries(domainMap, &input));
    setEntriesAndId(domainMap, &input, theIntersection);
  }
}

void BackwardsRelationDataflowAnalysis::flowBranch(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  for (size_t i = 0, n = cfOp->getNumResults(); i < n; ++i) {
    Value result = cfOp->getResult(i);
    // Flow from control flow op results backwards into each sub-block output.
    Set theIntersection = operandIntersection(domainMap, result);
    for (Region &region : cfOp->getRegions()) {
      Block *subBlock = &region.front();
      auto outputOp = cast<MEF::OutputOp>(subBlock->getTerminator());
      OpOperand *output = &outputOp->getOpOperand(i);
      setEntriesAndId(domainMap, output, theIntersection);
    }
  }

  for (Region &region : cfOp->getRegions()) {
    Block *subBlock = &region.front();
    // Flow the sub block.
    flowBlock(subBlock, domainMap, flowOp);
  }

  // Flow from the intersection of all sub-block arguments backwards to
  // the control flow op inputs.
  for (size_t i = 0, n = cfOp.getInputs().size(); i < n; ++i) {
    const OpOperand *input = &cfOp.getInputOpOperands()[i];
    Set theIntersection = domainMap.fullSet;
    for (Region &region : cfOp->getRegions()) {
      Block *subBlock = &region.front();
      domainMap.setIntersection(
          theIntersection,
          operandIntersection(domainMap, subBlock->getArgument(i)));
    }
    setEntriesAndId(domainMap, input, theIntersection);
  }
}

void BackwardsRelationDataflowAnalysis::flowControlFlow(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  switch (cfOp.getFlavor()) {
  case GRT::kLoopLike: {
    bool outerChanged = changed;
    if (cfOp->getNumRegions() == 1) {
      // The sub-block may be executed zero times, so flow backwards
      // from results to inputs.
      for (const auto &[input, result] :
           llvm::zip(cfOp.getInputOpOperands(), cfOp->getResults()))
        setEntriesAndId(domainMap, &input,
                        operandIntersection(domainMap, result));
    } else {
      // The first sub-block will be executed at least once.
      flowCondition(cfOp, domainMap, flowOp);
    }

    // Until fixed point...
    LLVM_DEBUG(llvm::dbgs() << "till fixed point...\n");
    do {
      changed = false;

      if (cfOp->getNumRegions() == 1) {
        // The sub-block may be executed again.
        flowSimpleLoop(cfOp, domainMap, flowOp);
      } else {
        // The second sub-block then first-sub block may be executed again.
        flowConditionedLoop(cfOp, domainMap, flowOp);
      }

      outerChanged |= changed;
      LLVM_DEBUG(if (changed) llvm::dbgs() << "... repeat\n");
    } while (changed);
    changed = outerChanged;
    break;
  }
  case GRT::kBranchLike: {
    // Flow each sub-block.
    flowBranch(cfOp, domainMap, flowOp);
    break;
  }
  }
}

void BackwardsRelationDataflowAnalysis::flowDefault(Operation *op,
                                                    DomainMap &domainMap) {
  if (isa<MEF::OutputOp>(op))
    // entries for the output op operands are established from the outer block.
    return;

  for (const OpOperand &operand : op->getOpOperands())
    setEntriesAndId(domainMap, &operand, domainMap.emptySet);
}

void BackwardsRelationDataflowAnalysis::flowBlock(Block *block,
                                                  DomainMap &domainMap,
                                                  const FlowOp &flowOp) {
  SmallVector<Operation *> ops = getDataflowOrder(block);
  for (Operation *op : llvm::reverse(ops))
    flowOp(op, domainMap);
}

void BackwardsRelationDataflowAnalysis::flowModel(ModelOp modelOp,
                                                  DomainMap &domainMap,
                                                  const FlowOp &flowOp) {
  for (Region &region : modelOp->getRegions()) {
    Block *block = &region.front();
    auto outputOp = cast<MEF::OutputOp>(block->getTerminator());
    for (const OpOperand &output : outputOp->getOpOperands())
      setEntriesAndId(domainMap, &output, domainMap.emptySet);
    flowBlock(&region.front(), domainMap, flowOp);
  }
}
