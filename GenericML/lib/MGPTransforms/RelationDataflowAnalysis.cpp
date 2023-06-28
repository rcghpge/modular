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

  // Flow the body sub block.
  flowBlock(bodyBlock, domainMap, flowOp);

  // Flow from the sub-block outputs back around to sub-block arguments, but
  // combine with existing results.
  for (const auto [output, arg] :
       llvm::zip(bodyBlock->getTerminator()->getOperands(),
                 bodyBlock->getArguments())) {
    Set set = getEntriesAndId(domainMap, output);
    domainMap.leastUpperBound(set, getEntries(domainMap, arg));
    setEntries(domainMap, arg, set);
  }
}

void ForwardsRelationDataflowAnalysis::flowConditionedLoop(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  assert(cfOp->getNumRegions() == 2);
  Block *condBlock = &cfOp->getRegion(0).front();
  Block *bodyBlock = &cfOp->getRegion(1).front();

  // Flow the cond sub block.
  flowBlock(condBlock, domainMap, flowOp);

  // Flow from the cond sub-block outputs forwards to body sub-block args.
  for (const auto [condOutput, bodyArg] :
       llvm::zip(condBlock->getTerminator()->getOperands(),
                 bodyBlock->getArguments()))
    setEntries(domainMap, bodyArg, getEntriesAndId(domainMap, condOutput));

  // Flow the body sub block.
  flowBlock(bodyBlock, domainMap, flowOp);

  // Flow from the body sub-block outputs back around to the cond sub-block
  // arguments, but combine with existing args.
  for (const auto [bodyOutput, condArg] :
       llvm::zip(bodyBlock->getTerminator()->getOperands(),
                 condBlock->getArguments())) {
    Set set = getEntriesAndId(domainMap, bodyOutput);
    domainMap.leastUpperBound(set, getEntries(domainMap, condArg));
    setEntries(domainMap, condArg, set);
  }
}

std::pair<ForwardsRelationDataflowAnalysis::Set,
          ForwardsRelationDataflowAnalysis::Set>
ForwardsRelationDataflowAnalysis::splitByScope(const Set &set, Block *block) {
  Set within = index->emptySet();
  Set without = index->emptySet();
  for (Value value : index->setToVector(set)) {
    if (value.getDefiningOp() && block->getParent()->isAncestor(
                                     value.getDefiningOp()->getParentRegion()))
      index->addToSet(within, value);
    else
      index->addToSet(without, value);
  }
  return {within, without};
}

void ForwardsRelationDataflowAnalysis::flowBranch(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  // Flow from control flow op inputs forwards into each sub-block's args.
  for (Region &region : cfOp->getRegions()) {
    Block *subBlock = &region.front();
    for (auto [input, arg] :
         llvm::zip(cfOp.getInputs(), subBlock->getArguments()))
      setEntries(domainMap, arg, getEntriesAndId(domainMap, input));
    flowBlock(subBlock, domainMap, flowOp);
  }

  // Flow from combination of the outputs of each sub-block forwards to the
  // control flow op results. However, account for the mutual-exclusivity
  // between branches if requested.
  bool shouldSplit = domainMap.handleDisjointBranches && isAtMostOnce(cfOp);
  for (size_t i = 0, n = cfOp->getNumResults(); i < n; ++i) {
    Value result = cfOp->getResult(i);
    Set glbOfWithin = domainMap.top;
    Set lubOfWithout = domainMap.bottom;
    for (Region &region : cfOp->getRegions()) {
      Block *subBlock = &region.front();
      Value output = subBlock->getTerminator()->getOperand(i);
      Set set = getEntriesAndId(domainMap, output);
      if (shouldSplit) {
        auto [within, without] = splitByScope(set, subBlock);
        domainMap.greatestLowerBound(glbOfWithin, within);
        domainMap.leastUpperBound(lubOfWithout, without);
      } else {
        domainMap.leastUpperBound(lubOfWithout, set);
      }
    }
    if (shouldSplit)
      domainMap.greatestLowerBound(lubOfWithout, glbOfWithin);
    setEntries(domainMap, result, lubOfWithout);
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
    LLVM_DEBUG(llvm::dbgs() << "fixed point setup...\n");
    // Flow from control flow inputs to args of first sub-block.
    for (auto [input, arg] :
         llvm::zip(cfOp.getInputs(), cfOp->getRegions().front().getArguments()))
      setEntries(domainMap, arg, getEntriesAndId(domainMap, input));

    // Until fixed point...
    bool outerChanged = domainMap.changed;
    do {
      LLVM_DEBUG(llvm::dbgs() << "fixed point iteration...\n");
      domainMap.changed = false;

      if (cfOp->getNumRegions() == 1) {
        // The sub-block may be executed again.
        flowSimpleLoop(cfOp, domainMap, flowOp);
      } else {
        // The second sub-block then first-sub block may be executed again.
        flowConditionedLoop(cfOp, domainMap, flowOp);
      }

      outerChanged |= domainMap.changed;
    } while (domainMap.changed);
    LLVM_DEBUG(llvm::dbgs() << "fixed point reached\n");
    domainMap.changed = outerChanged;

    // Flow from args of last sub-block to control flow outputs.
    for (auto [arg, result] : llvm::zip(
             cfOp->getRegions().back().getArguments(), cfOp->getResults()))
      setEntries(domainMap, result, getEntriesAndId(domainMap, arg));

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
    setEmpty(domainMap, result);
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
    setEmpty(domainMap, arg);

  // Flow forwards through 'init' block.
  flowBlock(initBlock, domainMap, flowOp);

  // Flow from 'init' outputs {0, 1} to 'execute' args {0, 1}.
  for (const auto &[output, arg] :
       llvm::zip(initBlock->getTerminator()->getOperands(),
                 executeBlock->getArguments()))
    setEntries(domainMap, arg, getEntriesAndId(domainMap, output));
  // Start remaining 'execute' args start.
  for (Value arg : llvm::drop_begin(executeBlock->getArguments(), 2))
    setEmpty(domainMap, arg);

  // Flow forwards through 'execute' block.
  flowBlock(executeBlock, domainMap, flowOp);
}

//===----------------------------------------------------------------------===//
// BackwardsRelationDataflowAnalysis
//===----------------------------------------------------------------------===//

BackwardsRelationDataflowAnalysis::Set
BackwardsRelationDataflowAnalysis::operandCombination(
    const DomainMap &domainMap, Value value) const {
  Set set = domainMap.bottom;
  for (OpOperand &operand : value.getUses())
    domainMap.leastUpperBound(set, getEntriesAndId(domainMap, &operand));
  return set;
}

void BackwardsRelationDataflowAnalysis::flowSimpleLoop(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  assert(cfOp->getNumRegions() == 1);
  Block *bodyBlock = &cfOp->getRegion(0).front();

  // Flow the body sub block.
  flowBlock(bodyBlock, domainMap, flowOp);

  // Flow from the sub-block arguments back around to the block outputs,
  // but combine with the existing inputs.
  for (const auto &[arg, output] :
       llvm::zip(bodyBlock->getArguments(),
                 bodyBlock->getTerminator()->getOpOperands())) {
    Set set = operandCombination(domainMap, arg);
    domainMap.leastUpperBound(set, getEntries(domainMap, &output));
    setEntries(domainMap, &output, set);
  }
}

void BackwardsRelationDataflowAnalysis::flowConditionedLoop(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  assert(cfOp->getNumRegions() == 2);
  Block *condBlock = &cfOp->getRegion(0).front();
  Block *bodyBlock = &cfOp->getRegion(1).front();

  // Flow the cond sub block.
  flowBlock(condBlock, domainMap, flowOp);

  // Flow from the cond sub-block args backwards to outputs of body block.
  for (const auto &[condArg, bodyOutput] :
       llvm::zip(condBlock->getArguments(),
                 bodyBlock->getTerminator()->getOpOperands()))
    setEntries(domainMap, &bodyOutput, operandCombination(domainMap, condArg));

  // Flow the body sub block.
  flowBlock(bodyBlock, domainMap, flowOp);

  // Flow from the body sub-block args backwards to the cond sub-block
  // outputs, but combine with existing outputs.
  for (const auto &[bodyArg, condOutput] :
       llvm::zip(bodyBlock->getArguments(),
                 condBlock->getTerminator()->getOpOperands())) {
    Set set = operandCombination(domainMap, bodyArg);
    domainMap.leastUpperBound(set, getEntries(domainMap, &condOutput));
    setEntries(domainMap, &condOutput, set);
  }
}

std::pair<BackwardsRelationDataflowAnalysis::Set,
          BackwardsRelationDataflowAnalysis::Set>
BackwardsRelationDataflowAnalysis::splitByScope(const Set &set, Block *block) {
  Set within = index->emptySet();
  Set without = index->emptySet();
  for (const OpOperand *operand : index->setToVector(set)) {
    if (operand->get().getDefiningOp() &&
        block->getParent()->isAncestor(
            operand->get().getDefiningOp()->getParentRegion()))
      index->addToSet(within, operand);
    else
      index->addToSet(without, operand);
  }
  return {within, without};
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
      OpOperand *output = &subBlock->getTerminator()->getOpOperand(i);
      setEntries(domainMap, output, set);
    }
  }

  for (Region &region : cfOp->getRegions()) {
    Block *subBlock = &region.front();
    // Flow the sub block.
    flowBlock(subBlock, domainMap, flowOp);
  }

  // Flow from the combination of all sub-block arguments backwards to
  // the control flow op inputs. However, account for the mutual-exclusivity
  // between the branches if requested.
  bool shouldSplit = domainMap.handleDisjointBranches && isAtMostOnce(cfOp);
  for (size_t i = 0, n = cfOp.getInputs().size(); i < n; ++i) {
    const OpOperand *input = &cfOp.getInputOpOperands()[i];
    Set glbOfWithin = domainMap.top;
    Set lubOfWithout = domainMap.bottom;
    for (Region &region : cfOp->getRegions()) {
      Block *subBlock = &region.front();
      Set set = operandCombination(domainMap, subBlock->getArgument(i));
      if (shouldSplit) {
        auto [within, without] = splitByScope(set, subBlock);
        domainMap.greatestLowerBound(glbOfWithin, within);
        domainMap.leastUpperBound(lubOfWithout, without);
      } else {
        domainMap.leastUpperBound(lubOfWithout, set);
      }
    }
    if (shouldSplit)
      domainMap.greatestLowerBound(lubOfWithout, glbOfWithin);
    setEntries(domainMap, input, lubOfWithout);
  }
}

void BackwardsRelationDataflowAnalysis::flowControlFlow(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  switch (cfOp.getFlavor()) {
  case GRT::kLoopLike: {
    LLVM_DEBUG(llvm::dbgs() << "fixed point setup...\n");
    // Flow from control flow op results to outputs of first sub-block.
    for (const auto &[result, output] :
         llvm::zip(cfOp->getResults(), cfOp->getRegions()
                                           .front()
                                           .front()
                                           .getTerminator()
                                           ->getOpOperands()))
      setEntries(domainMap, &output, operandCombination(domainMap, result));

    // Until fixed point...
    bool outerChanged = domainMap.changed;
    do {
      LLVM_DEBUG(llvm::dbgs() << "fixed point iteration...\n");
      domainMap.changed = false;

      if (cfOp->getNumRegions() == 1) {
        // The sub-block may be executed again.
        flowSimpleLoop(cfOp, domainMap, flowOp);
      } else {
        // The second sub-block then first-sub block may be executed again.
        flowConditionedLoop(cfOp, domainMap, flowOp);
      }

      outerChanged |= domainMap.changed;
    } while (domainMap.changed);
    LLVM_DEBUG(llvm::dbgs() << "fixed point reached\n");
    domainMap.changed = outerChanged;

    // Flow from args of first sub-block back to control flow op inputs.
    for (const auto &[arg, input] :
         llvm::zip(cfOp->getRegions().front().getArguments(),
                   cfOp.getInputOpOperands()))
      setEntries(domainMap, &input, operandCombination(domainMap, arg));

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
    setEmpty(domainMap, &operand);
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
  for (const OpOperand &output : execBlock->getTerminator()->getOpOperands())
    setEmpty(domainMap, &output);

  // Flow backwards through 'execute' block.
  flowBlock(execBlock, domainMap, flowOp);

  // Flow from 'execute' args {0, 1} backwards to 'init' outputs {0, 1}.
  for (const auto &[arg, output] :
       llvm::zip(execBlock->getArguments(),
                 initBlock->getTerminator()->getOpOperands()))
    setEntries(domainMap, &output, operandCombination(domainMap, arg));

  // Flow backwards through 'init' block.
  flowBlock(initBlock, domainMap, flowOp);
}
