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
  // combine with existing results.
  auto outputOp = cast<MEF::OutputOp>(bodyBlock->getTerminator());
  for (const auto [output, result] :
       llvm::zip(outputOp->getOperands(), cfOp->getResults())) {
    Set set = getEntries(domainMap, output);
    domainMap.setIntersection(set, getEntries(domainMap, result));
    setEntriesAndId(domainMap, result, set);
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

  // Flow from the body sub-block outputs back around to the cond sub-block
  // arguments, but combine with existing args.
  auto bodyOutputOp = cast<MEF::OutputOp>(bodyBlock->getTerminator());
  for (const auto [output, arg] :
       llvm::zip(bodyOutputOp->getOperands(), condBlock->getArguments())) {
    Set set = getEntries(domainMap, output);
    domainMap.setIntersection(set, getEntries(domainMap, arg));
    setEntriesAndId(domainMap, arg, set);
  }

  // Flow the cond sub block.
  flowBlock(condBlock, domainMap, flowOp);

  // Flow from the cond sub-block outputs forwards to control flow op results,
  // but combine with existing results.
  auto condOutputOp = cast<MEF::OutputOp>(condBlock->getTerminator());
  for (const auto [output, result] :
       llvm::zip(condOutputOp->getOperands(), cfOp->getResults())) {
    Set set = getEntries(domainMap, output);
    domainMap.setIntersection(set, getEntries(domainMap, result));
    setEntriesAndId(domainMap, result, set);
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

  // Flow from combination of the outputs of each sub-block forwards to the
  // control flow op results.
  for (size_t i = 0, n = cfOp->getNumResults(); i < n; ++i) {
    Value result = cfOp->getResult(i);
    Set set = domainMap.fullSet;
    for (Region &region : cfOp->getRegions()) {
      Block *subBlock = &region.front();
      auto outputOp = cast<MEF::OutputOp>(subBlock->getTerminator());
      Value output = outputOp->getOperand(i);
      domainMap.setIntersection(set, getEntries(domainMap, output));
    }
    setEntriesAndId(domainMap, result, set);
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
    LLVM_DEBUG(llvm::dbgs() << "fixed point setup...\n");
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
    do {
      LLVM_DEBUG(llvm::dbgs() << "fixed point iteration...\n");
      changed = false;

      if (cfOp->getNumRegions() == 1) {
        // The sub-block may be executed again.
        flowSimpleLoop(cfOp, domainMap, flowOp);
      } else {
        // The second sub-block then first-sub block may be executed again.
        flowConditionedLoop(cfOp, domainMap, flowOp);
      }

      outerChanged |= changed;
    } while (changed);
    LLVM_DEBUG(llvm::dbgs() << "fixed point reached\n");
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
    setId(domainMap, result);
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
  Block *initBlock = modelOp.getInitBlock();
  Block *executeBlock = modelOp.getExecuteBlock();

  // Start with the 'init' block args.
  for (Value arg : initBlock->getArguments())
    setId(domainMap, arg);

  // Flow forwards through 'init' block.
  flowBlock(initBlock, domainMap, flowOp);

  // Flow from 'init' outputs {0, 1} to 'execute' args {0, 1}.
  auto initOutputOp = cast<MEF::OutputOp>(initBlock->getTerminator());
  for (const auto &[output, arg] :
       llvm::zip(initOutputOp->getOperands(), executeBlock->getArguments()))
    setEntriesAndId(domainMap, arg, getEntries(domainMap, output));
  // Start remaining 'execute' args start.
  for (Value arg : llvm::drop_begin(executeBlock->getArguments(), 2))
    setId(domainMap, arg);

  // Flow forwards through 'execute' block.
  flowBlock(executeBlock, domainMap, flowOp);
}

//===----------------------------------------------------------------------===//
// BackwardsRelationDataflowAnalysis
//===----------------------------------------------------------------------===//

BackwardsRelationDataflowAnalysis::Set
BackwardsRelationDataflowAnalysis::operandCombination(
    const DomainMap &domainMap, Value value) const {
  Set set = domainMap.fullSet;
  for (OpOperand &operand : value.getUses())
    domainMap.setIntersection(set, getEntries(domainMap, &operand));
  return set;
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
  // inputs, but combine with the existing inputs.
  for (const auto &[arg, input] :
       llvm::zip(bodyBlock->getArguments(), cfOp.getInputOpOperands())) {
    Set set = operandCombination(domainMap, arg);
    domainMap.setIntersection(set, getEntries(domainMap, &input));
    setEntriesAndId(domainMap, &input, set);
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
    setEntriesAndId(domainMap, &output, operandCombination(domainMap, result));

  // Flow the sub block.
  flowBlock(condBlock, domainMap, flowOp);

  // Flow from the sub-block arguments backwards to control flow op inputs.
  for (const auto &[arg, input] :
       llvm::zip(condBlock->getArguments(), cfOp.getInputOpOperands()))
    setEntriesAndId(domainMap, &input, operandCombination(domainMap, arg));
}

void BackwardsRelationDataflowAnalysis::flowConditionedLoop(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  assert(cfOp->getNumRegions() == 2);
  Block *condBlock = &cfOp->getRegion(0).front();
  Block *bodyBlock = &cfOp->getRegion(1).front();

  // Flow from the control flow op inputs to the body sub-block outputs.
  auto bodyOutputOp = cast<MEF::OutputOp>(bodyBlock->getTerminator());
  for (const auto &[input, output] :
       llvm::zip(cfOp.getInputOpOperands(), bodyOutputOp->getOpOperands()))
    setEntriesAndId(domainMap, &output, getEntries(domainMap, &input));

  // Flow the body sub block.
  flowBlock(bodyBlock, domainMap, flowOp);

  // Flow from the body sub-block args backwards to the cond sub-block
  // outputs, but combine with existing outputs.
  auto condOutputOp = cast<MEF::OutputOp>(condBlock->getTerminator());
  for (const auto &[arg, output] :
       llvm::zip(bodyBlock->getArguments(), condOutputOp->getOpOperands())) {
    Set set = operandCombination(domainMap, arg);
    domainMap.setIntersection(set, getEntries(domainMap, &output));
    setEntriesAndId(domainMap, &output, set);
  }

  // Flow the cond sub block.
  flowBlock(condBlock, domainMap, flowOp);

  // Flow from the cond sub-block args backwards to control flow op inputs,
  // but combine with existing inputs.
  for (const auto &[arg, input] :
       llvm::zip(condBlock->getArguments(), cfOp.getInputOpOperands())) {
    Set set = operandCombination(domainMap, arg);
    domainMap.setIntersection(set, getEntries(domainMap, &input));
    setEntriesAndId(domainMap, &input, set);
  }
}

void BackwardsRelationDataflowAnalysis::flowBranch(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  for (size_t i = 0, n = cfOp->getNumResults(); i < n; ++i) {
    Value result = cfOp->getResult(i);
    // Flow from control flow op results backwards into each sub-block output.
    Set set = operandCombination(domainMap, result);
    for (Region &region : cfOp->getRegions()) {
      Block *subBlock = &region.front();
      auto outputOp = cast<MEF::OutputOp>(subBlock->getTerminator());
      OpOperand *output = &outputOp->getOpOperand(i);
      setEntriesAndId(domainMap, output, set);
    }
  }

  for (Region &region : cfOp->getRegions()) {
    Block *subBlock = &region.front();
    // Flow the sub block.
    flowBlock(subBlock, domainMap, flowOp);
  }

  // Flow from the combination of all sub-block arguments backwards to
  // the control flow op inputs.
  for (size_t i = 0, n = cfOp.getInputs().size(); i < n; ++i) {
    const OpOperand *input = &cfOp.getInputOpOperands()[i];
    Set set = domainMap.fullSet;
    for (Region &region : cfOp->getRegions()) {
      Block *subBlock = &region.front();
      domainMap.setIntersection(
          set, operandCombination(domainMap, subBlock->getArgument(i)));
    }
    setEntriesAndId(domainMap, input, set);
  }
}

void BackwardsRelationDataflowAnalysis::flowControlFlow(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  switch (cfOp.getFlavor()) {
  case GRT::kLoopLike: {
    bool outerChanged = changed;
    LLVM_DEBUG(llvm::dbgs() << "fixed point setup...\n");
    if (cfOp->getNumRegions() == 1) {
      // The sub-block may be executed zero times, so flow backwards
      // from results to inputs.
      for (const auto &[result, input] :
           llvm::zip(cfOp->getResults(), cfOp.getInputOpOperands()))
        setEntriesAndId(domainMap, &input,
                        operandCombination(domainMap, result));
    } else {
      // The first sub-block will be executed at least once.
      flowCondition(cfOp, domainMap, flowOp);
    }

    // Until fixed point...
    do {
      LLVM_DEBUG(llvm::dbgs() << "fixed point iteration...\n");
      changed = false;

      if (cfOp->getNumRegions() == 1) {
        // The sub-block may be executed again.
        flowSimpleLoop(cfOp, domainMap, flowOp);
      } else {
        // The second sub-block then first-sub block may be executed again.
        flowConditionedLoop(cfOp, domainMap, flowOp);
      }

      outerChanged |= changed;
    } while (changed);
    LLVM_DEBUG(llvm::dbgs() << "fixed point reached\n");
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
    setId(domainMap, &operand);
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
  Block *initBlock = modelOp.getInitBlock();
  Block *execBlock = modelOp.getExecuteBlock();

  // Start with the 'execute' block outputs.
  auto executeOutputOp = cast<MEF::OutputOp>(execBlock->getTerminator());
  for (const OpOperand &output : executeOutputOp->getOpOperands())
    setId(domainMap, &output);

  // Flow backwards through 'execute' block.
  flowBlock(execBlock, domainMap, flowOp);

  // Flow from 'execute' args {0, 1} backwards to 'init' outputs {0, 1}.
  auto initOutputOp = cast<MEF::OutputOp>(initBlock->getTerminator());
  for (const auto &[arg, output] :
       llvm::zip(execBlock->getArguments(), initOutputOp->getOpOperands()))
    setEntriesAndId(domainMap, &output, operandCombination(domainMap, arg));

  // Flow backwards through 'init' block.
  flowBlock(initBlock, domainMap, flowOp);
}
