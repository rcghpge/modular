//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "GenericML/MGPDialect/BufferBuilder.h"
#include "GenericML/MGPDialect/MGPOpInterfaces.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTensorInterface.h"
#include "GenericML/MGPDialect/MGPUtils.h"
#include "GenericML/MGPTransforms/IndexSet.h"
#include "GenericML/MGPTransforms/Utils.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "GraphRT/MEFDialect/MEFOps.h"
#include "Support/AssertStream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#include "./RelationDataflowAnalysis.h"

using namespace std::placeholders;
using namespace mlir;

#define DEBUG_TYPE "exec-invariant-code-motion"

namespace M::MGP {

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_EXECINVARIANTCODEMOTIONPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

/// Returns the buffer returned by op, which must be the only op result, and
/// the op must be safely liftable into the 'init' block. Otherwise returns
/// null.
Value getLiftableBufferOp(Operation *op) {
  if (auto liftableOp = dyn_cast<MGPLiftableBufferCreateOpInterface>(op))
    return liftableOp.getOutBuffer();
  return nullptr;
}

/// Returns true if op could be safely lifted into the 'init' block, but is
/// considered trivial to compute.
///
/// It's not worth lifting 'trivial' ops to the 'init' block if no other
/// 'init' block op requires its results, since the cost of caching and
/// retrieving the buffer would outweigh the cost of simply executing the
/// op in the 'execute' phase on every call.
bool isLiftableTrivialOp(Operation *op) {
  // Consider all pure ops to be trivial.
  if (isPure(op))
    return true;
  // Liftable buffer ops declare whether they are trivial.
  if (auto liftableOp = dyn_cast<MGPLiftableBufferCreateOpInterface>(op))
    return liftableOp.isTrivial();
  return false;
}

/// Returns the chain returned by op, which must be the only op result,
/// and the op must be safely liftable into the 'init' block. Otherwise
/// returns null. These ops can be lifted to the 'init' block provided the
/// chain finds its way into the 'init' block output chain, which in turn
/// is used by the 'execute' block input chain.
Value getLiftableChainOp(Operation *op) {
  if (auto createChainOp = dyn_cast<GRT::ChainCreateOp>(op))
    return createChainOp.getResult();
  if (auto createChainOp = dyn_cast<MGPLiftableChainCreateOpInterface>(op))
    return createChainOp.getOutChain();
  return nullptr;
}

/// Return true if the result of op can be conveyed from the 'init' block
/// to the 'execute' block via the !mgp.context cache or 'execute' block
/// arguments.
bool canBeConveyed(Operation *op) {
  return getLiftableBufferOp(op) || getLiftableChainOp(op);
}

/// Returns true if the result of op cannot be cached, but it is ok for the
/// op to be executed in both the 'init' and 'execute' blocks. Doing so will
/// allow downstream ops to also be lifted to the 'init' block.
bool isRecomputableOp(Operation *op) {
  // We'll allow all pure ops to be recomputed.
  if (isPure(op))
    return true;

  // We also allow compiled kernels to be recomputed if they return indices,
  // which signals they are just shape functions and side-effect free.
  if (auto executeOp = dyn_cast<KernelExecuteOp>(op)) {
    if (!executeOp.getOutDimensions().empty())
      return true;
  }

  return false;
}

/// Returns true if the execution of op could be safely lifted or copied to
/// the 'init' block without causing unintended side effects.
bool canBeLifted(Operation *op) {
  return canBeConveyed(op) || isRecomputableOp(op);
}

// NOTE: In the following analyses we are using the existing 'dense' machinery
// (ie treating Values and OpOperands individually) even though we really only
// need a 'sparse' analysis (ie working with just the Operations). We are also
// assuming control flow ops and all ops within then are intrinsically
// unliftable, which means we need to override some of the default dataflow
// results.
// TODO(#24751): Support lifting of and from within control flow.

//===----------------------------------------------------------------------===//
// DependsOnArgsAnalysis
//===----------------------------------------------------------------------===//

using OperationSet = IndexSet<Operation *>::Set;

/// A dense, forwards dataflow analysis to find ops which cannot be lifted
/// from the 'init' to the 'execute' block because:
///  - The op is already considered unliftable by an earlier analysis (if any).
///  - The op depends on an argument to the 'execute' block (other than the
///    !mef.chain and !mgp.context arguments).
///  - The op is a terminator.
///  - The op is, or is within, a control flow op.
///  - The op has side effects which must be observed on each call to the
///    'execute' block, and cannot instead be performed just once in the 'init'
///    block.
///  - The op depends on the result of an unliftable op.
///
/// We consider ops already in the 'init' block to be liftable so that
/// the 'init' block outputs don't taint the first two 'execute' block inputs
/// as being unliftable. This allows us to use the standard dataflow machinery.
///
/// During the analysis a value will be 'true' (ie mapped to a set containing
/// the distinguished value) if it is an 'execute' block arg or produced
/// by an unliftable op.
class DependsOnArgsAnalysis {
public:
  OperationSet unliftableOps;

  DependsOnArgsAnalysis(IndexSet<Value> *index,
                        IndexSet<Operation *> *operationIndex,
                        OperationSet unliftableOps)
      : unliftableOps(std::move(unliftableOps)), operationIndex(operationIndex),
        analysis(index), dependsOnArgsValues(index->emptySet()) {
    setup();
  }

  void print(llvm::raw_ostream &os) {
    analysis.printPredicate(os, "dependsOnArgsValues", dependsOnArgsValues);
    llvm::dbgs() << "unliftable ops: ";
    operationIndex->printVs(llvm::dbgs(),
                            operationIndex->setToVector(unliftableOps));
    llvm::dbgs() << "\n";
  }

private:
  using Analysis = ForwardsRelationDataflowAnalysis;
  using DomainMap = Analysis::DomainMap;

  void flowGeneric(Operation *op, DomainMap &domainMap) {
    // Is the op already considered unliftable by an earlier analysis?
    bool isAlreadyUnliftable = operationIndex->setContains(unliftableOps, op);
    // Does the op have an unliftable operand by this analysis?
    Analysis::Set set = domainMap.bottom;
    for (Value operand : op->getOperands()) {
      domainMap.leastUpperBound(set,
                                analysis.getEntriesAndId(domainMap, operand));
    }
    bool hasUnliftableOperands =
        set.test(0); // distinguished always has index 0
    // Can the op even be lifted?
    bool isNotLiftable = !canBeLifted(op);
    // Is the op inside control flow?
    bool inControlFlow = !isExactlyOnce(op);
    bool isUnliftable = isAlreadyUnliftable || hasUnliftableOperands ||
                        isNotLiftable || inControlFlow;
    if (isUnliftable) {
      LLVM_DEBUG({
        llvm::dbgs() << "DependsOnAnalysis: considering ";
        operationIndex->printV(llvm::dbgs(), op);
        llvm::dbgs() << " to be unliftable. isAlreadyUnliftable="
                     << isAlreadyUnliftable
                     << ", hasUnliftableOperands=" << hasUnliftableOperands
                     << ", isNotLiftable=" << isNotLiftable
                     << ", inControlFlow=" << inControlFlow << "\n";
      });
      operationIndex->addToSet(unliftableOps, op);
    }
    // Flow forwards to the op's results.
    for (Value result : op->getResults()) {
      if (isUnliftable)
        analysis.setTrue(domainMap, result);
      else
        analysis.setDefault(domainMap, result);
    }
  }

  void flow(Operation *op, DomainMap &domainMap) {
    bool inInitBlock =
        op->getParentRegion()->isAncestor(&analysis.index->model.getInit());
    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
      // Usual control flow rules
      analysis.flowControlFlow(
          cfOp, domainMap,
          std::bind(&DependsOnArgsAnalysis::flow, this, _1, _2));
      if (!inInitBlock) {
        // Control flow ops are intrisically not liftable.
        operationIndex->addToSet(unliftableOps, op);
        // Thus we must make sure all the control flow results are marked as
        // unliftable, irrespective of the above.
        for (Value result : op->getResults())
          analysis.setTrue(domainMap, result);
      }
    } else if (isa<MEF::OutputOp>(op)) {
      if (!inInitBlock) {
        // Plumbing ops are intrinsically not liftable.
        operationIndex->addToSet(unliftableOps, op);
      }
    } else if (inInitBlock) {
      // Ignore 'init' block ops.
      analysis.flowDefault(op, domainMap);
    } else {
      flowGeneric(op, domainMap);
    }
  }

  void setup() {
    Analysis::Relation dependsOnArgsRelation;
    DomainMap domainMap(dependsOnArgsRelation, Analysis::kLubIsUnion,
                        analysis.index->emptySet(),
                        analysis.index->unitSet(analysis.index->distinguished));

    // The extra 'execute' args (ie not the !mef.chain or !mgp.context or
    // devices) vacuously depend on 'execute' args.
    for (Value arg : analysis.index->model.getExecuteInputs())
      analysis.setTrue(domainMap, arg);

    analysis.flowModel(analysis.index->model, domainMap,
                       std::bind(&DependsOnArgsAnalysis::flow, this, _1, _2));
    analysis.addToPredicate(dependsOnArgsValues, dependsOnArgsRelation);
  }

  IndexSet<Operation *> *operationIndex;
  Analysis analysis;
  // Computed only for use by print for debugging.
  Analysis::Set dependsOnArgsValues;
};

//===----------------------------------------------------------------------===//
// SuppressTrivialAnalysis
//===----------------------------------------------------------------------===//

/// A dense, backwards dataflow analysis to find ops which cannot be lifted
/// from the 'init' to the 'execute' block because:
///  - the op is already considered unliftable by an earlier analysis.
///  - the op is trivial to compute, and all of its results are used by ops
///    which cannot be lifted. In this case the cost of retrieving the result
///    from the cache is comparable to the cost of simply recomputing, so
///    leave the op where it is.
///  - the op's results are unused. This minimizes changes to the model for
///    ops that will be eliminated by DCE anyway.
///
/// During the analysis an operand will be 'true' (ie mapped to a set
/// containing the distinguished value) its using op is unliftable.
class SuppressTrivialAnalysis {
public:
  OperationSet unliftableOps;

  SuppressTrivialAnalysis(IndexSet<const OpOperand *> *index,
                          IndexSet<Operation *> *operationIndex,
                          OperationSet unliftableOps)
      : unliftableOps(std::move(unliftableOps)), operationIndex(operationIndex),
        analysis(index), suppressTrivialOperands(index->emptySet()) {
    setup();
  }

  void print(llvm::raw_ostream &os) {
    analysis.printPredicate(os, "suppressTrivialOperands",
                            suppressTrivialOperands);
    llvm::dbgs() << "unliftable ops: ";
    operationIndex->printVs(llvm::dbgs(),
                            operationIndex->setToVector(unliftableOps));
    llvm::dbgs() << "\n";
  }

private:
  using Analysis = BackwardsRelationDataflowAnalysis;
  using DomainMap = Analysis::DomainMap;

  void flowGeneric(Operation *op, DomainMap &domainMap) {
    // Is the op already considered unliftable by an earlier analysis?
    bool isAlreadyUnliftable = operationIndex->setContains(unliftableOps, op);
    // Is the op's results unused?
    bool isUnused = op->getUses().empty();
    // Is the op trivial to compute?
    bool isTrivial = isLiftableTrivialOp(op);
    // Are all uses of the op considered unliftable by this analysis?
    Analysis::Set set = domainMap.bottom;
    for (OpOperand &operand : op->getUses()) {
      domainMap.leastUpperBound(set,
                                analysis.getEntriesAndId(domainMap, &operand));
    }
    bool allUsesUnliftable = set.test(0); // distinguished always has index 0
    bool isUnliftable =
        isAlreadyUnliftable || isUnused || (isTrivial && allUsesUnliftable);
    if (isUnliftable) {
      LLVM_DEBUG({
        llvm::dbgs() << "SuppressTrivialAnalysis: considering ";
        operationIndex->printV(llvm::dbgs(), op);
        llvm::dbgs() << " to be unliftable. isAlreadyUnliftable="
                     << isAlreadyUnliftable << ", isUnused=" << isUnused
                     << ", isTrivial=" << isTrivial
                     << ", allUsesUnliftable=" << allUsesUnliftable << "\n";
      });
      operationIndex->addToSet(unliftableOps, op);
    }
    // Flow backwards to the op's operands.
    for (const OpOperand &opOperand : op->getOpOperands()) {
      if (isUnliftable)
        analysis.setTrue(domainMap, &opOperand);
      else
        analysis.setDefault(domainMap, &opOperand);
    }
  }

  void flow(Operation *op, DomainMap &domainMap) {
    bool inInitBlock =
        op->getParentRegion()->isAncestor(&analysis.index->model.getInit());
    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
      // Usual control flow rules.
      analysis.flowControlFlow(
          cfOp, domainMap,
          std::bind(&SuppressTrivialAnalysis::flow, this, _1, _2));
      if (!inInitBlock) {
        // Since the control flow op is intrinsically unliftable, all of its
        // operands must be marked as unliftable, irrespective of the above.
        for (const OpOperand &opOperand : op->getOpOperands())
          analysis.setTrue(domainMap, &opOperand);
      }
    } else if (isa<MEF::OutputOp>(op)) {
      // Ignore 'plumbing'.
    } else if (inInitBlock) {
      // Ignore the 'init' block ops.
      analysis.flowDefault(op, domainMap);
    } else {
      flowGeneric(op, domainMap);
    }
  }

  void setup() {
    Analysis::Relation suppressTrivialRelation;
    DomainMap domainMap(suppressTrivialRelation, Analysis::kLubIsIntersection,
                        analysis.index->emptySet(),
                        analysis.index->unitSet(analysis.index->distinguished));

    // All 'execute' block outputs can be considered to be unliftable.
    for (OpOperand &result : analysis.index->model.getExecuteBlock()
                                 ->getTerminator()
                                 ->getOpOperands())
      analysis.setTrue(domainMap, &result);

    analysis.flowModel(analysis.index->model, domainMap,
                       std::bind(&SuppressTrivialAnalysis::flow, this, _1, _2));
    analysis.addToPredicate(suppressTrivialOperands, suppressTrivialRelation);
  }

  IndexSet<Operation *> *operationIndex;
  Analysis analysis;
  // Computed only for use by print for debugging.
  Analysis::Set suppressTrivialOperands;
};

//===----------------------------------------------------------------------===//
// SuppressMutatedAnalysis
//===----------------------------------------------------------------------===//

/// A dense, backwards dataflow analysis to find ops which cannot be lifted
/// from the 'init' to the 'execute' block because:
///  - the op is already considered unliftable by an earlier analysis.
///  - at least one of the op's results is mutated by an op which cannot
///    be lifted. Since values held by the !mgp.context cache are read-only,
///    attempting to lift the op would result in a runtime error.
///
/// During the analysis an operand will be 'true' (ie mapped to a set
/// containing the distinguished value) if it is mutated by an unliftable op.
class SuppressMutatedAnalysis {
public:
  OperationSet unliftableOps;

  SuppressMutatedAnalysis(IndexSet<const OpOperand *> *index,
                          IndexSet<Operation *> *operationIndex,
                          OperationSet unliftableOps)
      : unliftableOps(std::move(unliftableOps)), operationIndex(operationIndex),
        analysis(index), suppressMutatedOperands(index->emptySet()) {
    setup();
  }

  void print(llvm::raw_ostream &os) {
    analysis.printPredicate(os, "suppressMutatedOperands",
                            suppressMutatedOperands);
    llvm::dbgs() << "unliftable ops: ";
    operationIndex->printVs(llvm::dbgs(),
                            operationIndex->setToVector(unliftableOps));
    llvm::dbgs() << "\n";
  }

private:
  using Analysis = BackwardsRelationDataflowAnalysis;
  using DomainMap = Analysis::DomainMap;

  void flowGeneric(Operation *op, DomainMap &domainMap) {
    // Is the op already considered unliftable by an earlier analysis?
    bool isAlreadyUnliftable = operationIndex->setContains(unliftableOps, op);
    // Are any uses of the op's results considered unliftable by this
    // analysis?
    Analysis::Set set = domainMap.bottom;
    for (OpOperand &operand : op->getUses()) {
      domainMap.leastUpperBound(set,
                                analysis.getEntriesAndId(domainMap, &operand));
    }
    bool anyUsesMutatedByUnliftable =
        set.test(0); // distinguished olways has index 0
    bool isUnliftable = isAlreadyUnliftable || anyUsesMutatedByUnliftable;
    if (isUnliftable) {
      LLVM_DEBUG({
        llvm::dbgs() << "SuppressMutatedAnalysis: considering ";
        operationIndex->printV(llvm::dbgs(), op);
        llvm::dbgs() << " to be unliftable. isAlreadyUnliftable="
                     << isAlreadyUnliftable << ", anyUsesMutatedByUnliftable="
                     << anyUsesMutatedByUnliftable << "\n";
      });
      operationIndex->addToSet(unliftableOps, op);
    }
    // Flow backwards to the op's operands. We only want to suppress lifting
    // upstream ops whose results are *mutated* by a downstream unliftable op.
    for (OpOperand &opOperand : op->getOpOperands()) {
      if (isUnliftable && isMutated(opOperand))
        analysis.setTrue(domainMap, &opOperand);
      else
        analysis.setDefault(domainMap, &opOperand);
    }
  }

  void flow(Operation *op, DomainMap &domainMap) {
    bool inInitBlock =
        op->getParentRegion()->isAncestor(&analysis.index->model.getInit());
    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
      // Usual control flow rules.
      analysis.flowControlFlow(
          cfOp, domainMap,
          std::bind(&SuppressMutatedAnalysis::flow, this, _1, _2));
      // (No need to fiddle with the control flow operands, the above analysis
      // is what we want.)
    } else if (isa<MEF::OutputOp>(op)) {
      // Ignore 'plumbing'.
    } else if (inInitBlock) {
      // Ignore the 'init' block ops.
      analysis.flowDefault(op, domainMap);
    } else {
      flowGeneric(op, domainMap);
    }
  }

  void setup() {
    Analysis::Relation suppressMutatedRelation;
    DomainMap domainMap(suppressMutatedRelation, Analysis::kLubIsUnion,
                        analysis.index->emptySet(),
                        analysis.index->unitSet(analysis.index->distinguished));

    // Leave the 'execute' block outputs alone, since they are not considered
    // mutated by the output and so can take their default 'false' value.

    analysis.flowModel(analysis.index->model, domainMap,
                       std::bind(&SuppressMutatedAnalysis::flow, this, _1, _2));
    analysis.addToPredicate(suppressMutatedOperands, suppressMutatedRelation);
  }

  IndexSet<Operation *> *operationIndex;
  Analysis analysis;
  // Computed only for use by print for debugging.
  Analysis::Set suppressMutatedOperands;
};

//===----------------------------------------------------------------------===//
// NeedToCloneAnalysis
//===----------------------------------------------------------------------===//

/// A dense, backwards dataflow analysis to find ops which, though liftable,
/// must also be 'left behind' in the 'execute' block by cloning. The op must
/// be cloned if it is i) liftable, and ii) its results cannot be conveyed via
/// the !mgp.context or 'exceute' block args, and iii) any:
///  - it provides operands to an unliftable op
///  - it provides operands to an op already considered needing to be cloned
///    by this analysis.
///
/// During the analysis an operand will be 'true' (ie mapped to a set
/// containing the distinguished value) if it is an operand to an unliftable
/// op or an op already recognized as needing to be cloned.
class NeedToCloneAnalysis {
public:
  OperationSet clonedOps;

  NeedToCloneAnalysis(IndexSet<const OpOperand *> *index,
                      IndexSet<Operation *> *operationIndex,
                      const OperationSet &unliftableOps)
      : clonedOps(operationIndex->emptySet()), operationIndex(operationIndex),
        unliftableOps(unliftableOps), analysis(index),
        needToCloneOperands(index->emptySet()) {
    setup();
  }

  void print(llvm::raw_ostream &os) {
    analysis.printPredicate(os, "needToCloneOperands", needToCloneOperands);
    llvm::dbgs() << "cloned ops: ";
    operationIndex->printVs(llvm::dbgs(),
                            operationIndex->setToVector(clonedOps));
    llvm::dbgs() << "\n";
  }

private:
  using Analysis = BackwardsRelationDataflowAnalysis;
  using DomainMap = Analysis::DomainMap;

  void flowGeneric(Operation *op, DomainMap &domainMap) {
    if (operationIndex->setContains(unliftableOps, op)) {
      // Ops which produce these operands may need to be cloned.
      for (OpOperand &opOperand : op->getOpOperands())
        analysis.setTrue(domainMap, &opOperand);
    } else if (canBeConveyed(op)) {
      // This op can be lifted, and it's results recovered in the 'execute'
      // block. Thus, the op's operands don't impose any constraint on upstream
      // ops.
      analysis.flowDefault(op, domainMap);
    } else {
      // Are any uses of the op's results considered needed by a cloned or
      // unliftable op?
      Analysis::Set set = domainMap.bottom;
      for (OpOperand &operand : op->getUses()) {
        domainMap.leastUpperBound(
            set, analysis.getEntriesAndId(domainMap, &operand));
      }
      bool neededByClonedOrUnliftable =
          set.test(0); // distinguished olways has index 0
      if (neededByClonedOrUnliftable) {
        LLVM_DEBUG({
          llvm::dbgs() << "NeedToCloneAnalysis: considering ";
          operationIndex->printV(llvm::dbgs(), op);
          llvm::dbgs() << " to require cloning. neededByClonedOrUnliftable="
                       << neededByClonedOrUnliftable << "\n";
        });
        operationIndex->addToSet(clonedOps, op);
      }
      // Flow backwards to the op's operands.
      for (OpOperand &opOperand : op->getOpOperands()) {
        if (neededByClonedOrUnliftable)
          analysis.setTrue(domainMap, &opOperand);
        else
          analysis.setDefault(domainMap, &opOperand);
      }
    }
  }

  void flow(Operation *op, DomainMap &domainMap) {
    bool inInitBlock =
        op->getParentRegion()->isAncestor(&analysis.index->model.getInit());
    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
      // Usual control flow rules.
      analysis.flowControlFlow(
          cfOp, domainMap, std::bind(&NeedToCloneAnalysis::flow, this, _1, _2));
      if (!inInitBlock) {
        // Since the control flow op is intrinsically unliftable, all its
        // operands must be marked as unliftabel, irrespective of the above.
        for (OpOperand &opOperand : op->getOpOperands())
          analysis.setTrue(domainMap, &opOperand);
      }
    } else if (isa<MEF::OutputOp>(op)) {
      // Ignore 'plumbing'.
    } else if (inInitBlock) {
      // Ignore 'init' block ops.
      analysis.flowDefault(op, domainMap);
    } else {
      flowGeneric(op, domainMap);
    }
  }

  void setup() {
    Analysis::Relation neededByExecuteRelation;
    DomainMap domainMap(neededByExecuteRelation, Analysis::kLubIsUnion,
                        analysis.index->emptySet(),
                        analysis.index->unitSet(analysis.index->distinguished));

    // All 'execute' block outputs are vacuously needed an unliftable op.
    for (OpOperand &result : analysis.index->model.getExecuteBlock()
                                 ->getTerminator()
                                 ->getOpOperands())
      analysis.setTrue(domainMap, &result);

    analysis.flowModel(analysis.index->model, domainMap,
                       std::bind(&NeedToCloneAnalysis::flow, this, _1, _2));
    analysis.addToPredicate(needToCloneOperands, neededByExecuteRelation);
  }

  IndexSet<Operation *> *operationIndex;
  const OperationSet &unliftableOps;
  Analysis analysis;
  // Computed only for use by print for debugging.
  Analysis::Set needToCloneOperands;
};

//===----------------------------------------------------------------------===//
// ExecInvariantCodeMotionPass
//===----------------------------------------------------------------------===//

class ExecInvariantCodeMotionPass
    : public impl::ExecInvariantCodeMotionPassBase<
          ExecInvariantCodeMotionPass> {
  using ExecInvariantCodeMotionPassBase::ExecInvariantCodeMotionPassBase;

  // Figure out which ops in the 'execute' block are unliftable, and which
  // are liftable but must be cloned before being lifted.
  std::pair<OperationSet, OperationSet> analyze(ModelOp modelOp) {
    OperationSet unliftableOps = operationIndex->emptySet();
    size_t origNumUnliftableOps;

    // Until unliftableOps reaches a fixed point...
    do {
      origNumUnliftableOps = unliftableOps.count();
      LLVM_DEBUG(llvm::dbgs() << "-------- analysis iteration --------\n");

      // Don't lift if depends on 'execute' block args.
      DependsOnArgsAnalysis dependsOnArgsAnalysis(valueIndex, operationIndex,
                                                  std::move(unliftableOps));
      LLVM_DEBUG(dependsOnArgsAnalysis.print(llvm::dbgs()));

      // Don't lift if trivial to recompute and only used by unliftable.
      SuppressTrivialAnalysis suppressTrivialAnalysis(
          operandIndex, operationIndex,
          std::move(dependsOnArgsAnalysis.unliftableOps));
      LLVM_DEBUG(suppressTrivialAnalysis.print(llvm::dbgs()));

      // Don't lift if mutated by an unliftable op.
      SuppressMutatedAnalysis suppressMutatedAnalysis(
          operandIndex, operationIndex,
          std::move(suppressTrivialAnalysis.unliftableOps));
      LLVM_DEBUG(suppressMutatedAnalysis.print(llvm::dbgs()));

      unliftableOps = std::move(suppressMutatedAnalysis.unliftableOps);
    } while (origNumUnliftableOps < unliftableOps.count());

    // Clone if an op is unliftable but needed, directly or indirectly, by
    // unliftable ops.
    NeedToCloneAnalysis needToCloneAnalysis(operandIndex, operationIndex,
                                            unliftableOps);

    LLVM_DEBUG({
      llvm::dbgs() << "-------- analysis fixed point --------------\n";
      needToCloneAnalysis.print(llvm::dbgs());
      llvm::dbgs() << "--------------------------------------------\n";
    });

    return {std::move(unliftableOps), std::move(needToCloneAnalysis.clonedOps)};
  }

  /// Lift (and clone if necessary) the ops that need to be lifted from the
  /// 'execute' to the 'init' blocks.
  void rewrite(const std::pair<OperationSet, OperationSet> &sets) {
    // Setup to use the analysis results.
    DenseSet<Operation *> unliftableOps;
    for (Operation *op : operationIndex->setToVector(sets.first))
      unliftableOps.insert(op);
    DenseSet<Operation *> cloneOps;
    for (Operation *op : operationIndex->setToVector(sets.second))
      cloneOps.insert(op);

    // Is operand directly used by an unliftable op?
    auto isOpOperandUnliftable = [&unliftableOps](const OpOperand &operand) {
      return unliftableOps.contains(operand.getOwner());
    };

    auto initOutputOp = cast<MEF::OutputOp>(*initState->init->rbegin());
    Value initOriginalChain = initState->getFinalChain();
    Location outputLoc = initOutputOp.getLoc();
    initState->builder.setInsertionPoint(initOutputOp);

    // This placeholder chain is used to gate storing all the lifted buffers
    // into the cache until all the side effects on those buffers have
    // completed. We'll replace this placeholder with the actual chain once
    // we've finished lifting all the ops. It is always a 'host' chain.
    auto placeholderBufferCacheInputChain =
        initState->builder.create<GRT::ChainCreateOp>(outputLoc);

    // TODO(#24751): The following chain dependencies may be too aggressive,
    // relax via some analysis

    // Collects additional chains (converted to !mef.chain if required)
    // which are ready when all side effects for lifted buffers have
    // completed. We'll need to gate adding buffers to the !mgp.context cache
    // slots until ALL these side effects have completed.
    SmallVector<Value> potentialBufferActionChains;

    // Collects additional chains (always as !mef.chains) for inserting lifted
    // buffers into the !mgp.context cache. These will need to be joined into
    // the final 'init' block output chain.
    SmallVector<Value> finalOutChains{initOriginalChain};

    // TODO(#24751): We only consider ops in the top-level 'execute' block.
    SmallVector<Operation *> allOps = getDataflowOrder(executeState->execute);

    LLVM_DEBUG(llvm::dbgs() << "first pass...\n");

    // First pass: clone ops from the 'execute' block into itself.
    // (We must do this as a separate pass so that the results of lifted ops
    // can be renamed in the right places.)
    IRMapping cloneIRMapping;
    for (Operation *op : allOps) {
      if (!cloneOps.contains(op))
        continue;
      LLVM_DEBUG({
        llvm::dbgs() << "cloning ";
        operationIndex->printV(llvm::dbgs(), op);
        llvm::dbgs() << "\n";
      });
      ASSERT_STREAM(!unliftableOps.contains(op),
                    << "attempting to clone an op (" << *op
                    << ") which is unliftable");
      ASSERT_STREAM(isRecomputableOp(op), << "attempting to clone an op ("
                                          << *op
                                          << ") which is not recomputable");

      // Although not required, this makes debugging easier.
      executeState->builder.setInsertionPoint(op);

      Operation *clonedOp = executeState->builder.clone(*op, cloneIRMapping);
      // As far as the following pass is concerned, the cloned ops will remain
      // in the 'execute' block so should be considered unliftable.
      unliftableOps.insert(clonedOp);
      // Uses of the original op which will remain in the 'execute' block
      // now use the cloned op.
      op->replaceUsesWithIf(clonedOp, isOpOperandUnliftable);
    }

    LLVM_DEBUG(llvm::dbgs() << "second pass...\n");

    // Second pass: lift ops from the 'execute' block to the 'init' block.
    for (Operation *op : allOps) {
      if (unliftableOps.contains(op))
        continue;
      LLVM_DEBUG({
        llvm::dbgs() << "moving ";
        operationIndex->printV(llvm::dbgs(), op);
        llvm::dbgs() << "\n";
      });

      // Although not required, this makes debugging easier.
      executeState->builder.setInsertionPoint(op);

      if (Value oldChain = getLiftableChainOp(op)) {
        // We will lift op to the 'init' block. Any references to its oldChain
        // which will remain in the 'execute' block must be replaced with (the
        // device appropriate form of) the 'execute' block's root chain. We'll
        // let CSE clean up any duplication for those new chains.
        // The remaining references to oldChain remain in-place, since those
        // ops will likewise be lifted.
        BufferBuilder *bufferBuilder =
            TensorBufferDialectInterface::getBuilderForTypeStorage(
                *builderContext, oldChain.getType());
        Value newChain;
        if (isa<MEF::ChainType>(oldChain.getType())) {
          newChain = executeState->getRootChain();
        } else {
          newChain = bufferBuilder->fromHostChain(
              executeState->builder, op->getLoc(), executeState->getContext(),
              executeState->getRootChain());
        }
        oldChain.replaceUsesWithIf(newChain, isOpOperandUnliftable);
        LLVM_DEBUG(llvm::dbgs() << "will lift chain op and replace old chain:\n"
                                << oldChain << "\nwith:\n"
                                << newChain << "\n");

      } else if (Value oldBuffer = getLiftableBufferOp(op)) {
        // We will lift op to the 'init' block. Any references to its
        // oldBuffer which will remain in the 'execute' block must be replaced
        // with a retrieval of the buffer from the cache, and the 'init' block
        // must store oldBuffer into the cache.
        if (llvm::any_of(op->getUses(), isOpOperandUnliftable)) {
          // We must also collect the output chains (in !mef.chain form) for
          // all the ops which may write to the buffer. We can't safely store
          // the buffer in the 'init' block until all those ops have
          // completed. However, rather than tracking this for each buffer
          // individually, we instead simply wait for *all* such ops to have
          // completed in the 'init' block before *all* buffers are cached.
          for (OpOperand &use : op->getUses()) {
            if (isMutated(use)) {
              // Note: Very few ops are actually trivial. Marking ops as trivial
              // can trigger this error, specifically in the case where the
              // parent op can be lifted, but the "trivial" child op cannot be.
              // We could deal with this properly by cloning the parent or just
              // moving the trivial child anyways, but since very few ops are
              // trivial anyways it won't happen often.
              ASSERT_STREAM(!isOpOperandUnliftable(use),
                            << "attempting to lift buffer (" << *op
                            << ") which will be further mutated "
                               "in the 'execute' block");
              for (Value res : use.getOwner()->getResults()) {
                if (isa<MEF::MEFChainTypeInterface>(res.getType())) {
                  BufferBuilder *bufferBuilder =
                      TensorBufferDialectInterface::getBuilderForTypeStorage(
                          *builderContext, res.getType());
                  potentialBufferActionChains.push_back(
                      bufferBuilder->toHostChain(initState->builder,
                                                 op->getLoc(),
                                                 initState->getContext(), res));
                }
              }
            }
          }

          // Cache the buffer in the 'init' block, but only once all the
          // chains associated with side effects on all the buffers have
          // completed.
          BufferBuilder *bufferBuilder =
              TensorBufferDialectInterface::getBuilderForTypeStorage(
                  *builderContext, oldBuffer.getType());
          auto [bufferCacheOutChain, currBufferSlot] = bufferBuilder->cache(
              initState->builder, op->getLoc(), initState->getContext(),
              placeholderBufferCacheInputChain, initState->contextCreateOp,
              oldBuffer);

          // The final 'init' block chain must wait for this new cache op.
          finalOutChains.push_back(bufferCacheOutChain);

          // Recover the cached buffer in the 'execute' block.
          auto newBuffer = bufferBuilder->getCached(
              executeState->builder, op->getLoc(), executeState->getContext(),
              executeState->getRootChain(), currBufferSlot,
              oldBuffer.getType());

          // Replace old uses that will remain in the 'execute' block.
          oldBuffer.replaceUsesWithIf(newBuffer, isOpOperandUnliftable);
          LLVM_DEBUG(llvm::dbgs()
                     << "will lift buffer op and replace old buffer:\n"
                     << oldBuffer << "\nwith:\n"
                     << newBuffer << "\nvia cached:\n"
                     << bufferCacheOutChain << "\n");
        } else {
          LLVM_DEBUG(llvm::dbgs() << "will lift buffer op without caching\n");
        }

      } else {
        ASSERT_STREAM(isRecomputableOp(op),
                      << "attempting to lift an op (" << *op
                      << ") which is neither a liftable chain or "
                         "buffer op, nor a recomputable op");

        // Note that the op may have been cloned by the first pass.
      }

      LLVM_DEBUG(llvm::dbgs() << "considering op to be recomputable:\n"
                              << *op << "\n");
      // Move the original op
      op->moveBefore(initOutputOp);
    }

    // Replace all uses of the 'execute' block's input chain with the output
    // chain of the 'init' block BEFORE we updated in above.
    executeState->getRootChain().replaceUsesWithIf(
        initOriginalChain, [&](OpOperand &operand) {
          return operand.getOwner()->getBlock() == initState->init;
        });

    // Similarly, replace all uses of the 'execute' block's input context
    // with the context from the 'init' block.
    executeState->getContext().replaceUsesWithIf(
        initState->contextCreateOp, [&](OpOperand &operand) {
          return operand.getOwner()->getBlock() == initState->init;
        });

    // Replace the caching placeholder chain with the !mef.chain which is
    // ready only when all side effects on all buffers lifted to the 'init'
    // block have completed.
    placeholderBufferCacheInputChain.replaceAllUsesWith(
        initState->builder
            .create<GRT::ChainCreateOp>(outputLoc, potentialBufferActionChains)
            .getResult());
    placeholderBufferCacheInputChain.erase();

    // Make sure the final output !mef.chain waits for both the original
    // 'init' block output chain AND all the chains for the caching ops
    // we just inserted.
    auto mergedFinalOutChain = initState->builder.create<GRT::ChainCreateOp>(
        outputLoc, finalOutChains);
    initState->updateFinalChain(mergedFinalOutChain);
  }

  void runOnOperation() override {
    assert(!initState && !executeState && !valueIndex && !operandIndex &&
           !operationIndex && !builderContext);
    ModelOp modelOp = getOperation();
    InitState theInitState(modelOp.getContext(), modelOp.getInitBlock());
    initState = &theInitState;
    ExecuteState theExecuteState(modelOp.getContext(), modelOp);
    executeState = &theExecuteState;
    IndexSet<Value> theValueIndex = createValueIndexSet(modelOp);
    valueIndex = &theValueIndex;
    IndexSet<const OpOperand *> theOperandIndex =
        createOperandIndexSet(modelOp);
    operandIndex = &theOperandIndex;
    IndexSet<Operation *> theOperationIndex = createOperationIndexSet(modelOp);
    operationIndex = &theOperationIndex;

    // Recover type and buffer builders appropriate for the model.
    BufferBuilderContext theBuilderContext =
        TensorBufferDialectInterface::getAllBufferBuilders(modelOp);
    builderContext = &theBuilderContext;

    LLVM_DEBUG({
      llvm::dbgs() << "-------- before ExecInvariantCodeMotionPass --------\n";
      valueIndex->printModel(llvm::dbgs());
      llvm::dbgs() << "----------------------------------------------------\n";
    });

    rewrite(analyze(modelOp));

    // Clear state because the instance might be reused
    builderContext = nullptr;
    operationIndex = nullptr;
    operandIndex = nullptr;
    valueIndex = nullptr;
    executeState = nullptr;
    initState = nullptr;
  }

  /// Rewrite helpers.
  InitState *initState = nullptr;
  ExecuteState *executeState = nullptr;
  IndexSet<Value> *valueIndex = nullptr;
  IndexSet<const OpOperand *> *operandIndex = nullptr;
  IndexSet<Operation *> *operationIndex = nullptr;

  /// All the type and buffer builders we'll need while working on the model.
  BufferBuilderContext *builderContext = nullptr;
};

} // namespace
} // namespace M::MGP
