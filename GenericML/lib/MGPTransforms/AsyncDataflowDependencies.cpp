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
using namespace std::placeholders;

#define DEBUG_TYPE "mgp-dataflow"

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

AsyncDataflowDependencies::AsyncDataflowDependencies(
    IndexSet<Value> *valueIndex, IndexSet<Operation *> *operationIndex)
    : valueAnalysis(valueIndex), operationAnalysis(operationIndex),
      sameAsEquivalence(valueIndex->identityEquivalence()),
      maybeSameAsEquivalence(valueIndex->identityEquivalence()),
      alwaysReadyPredicate(valueIndex->emptySet()) {
  setupSameAs();
  setupMaybeSameAs();
  setupAlwaysReady();
  setupImpliesReady();
  setupStructurallyDisjoint();
  setupRunsLater();
}

bool AsyncDataflowDependencies::sameAs(Value lhs, Value rhs) const {
  return valueAnalysis.index->areEquivalent(sameAsEquivalence, lhs, rhs);
}

bool AsyncDataflowDependencies::maybeSameAs(Value lhs, Value rhs) const {
  return valueAnalysis.index->areEquivalent(maybeSameAsEquivalence, lhs, rhs);
}

AsyncDataflowDependencies::ValueSet
AsyncDataflowDependencies::maybeSameAsSet(Value lhs) const {
  return valueAnalysis.index->getEquivalenceClass(maybeSameAsEquivalence, lhs);
}

bool AsyncDataflowDependencies::isAlwaysReady(Value value) const {
  return valueAnalysis.index->setContains(alwaysReadyPredicate, value);
}

bool AsyncDataflowDependencies::impliesReady(Value lhs, Value rhs) const {
  return lhs == rhs ||
         valueAnalysis.containsEntry(impliesReadyRelation, lhs, rhs);
}

AsyncDataflowDependencies::ValueSet
AsyncDataflowDependencies::downwardsImpliesReadySet(Value rhs) const {
  return valueAnalysis.getEntriesAndId(inverseImpliesReadyRelation, rhs);
}

void AsyncDataflowDependencies::simplify(SmallVectorImpl<Value> &values) const {
  // CAUTION: Don't underflow ip1.
  // CAUTION: Size of values changes inside loop body.
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

bool AsyncDataflowDependencies::areDisjoint(Operation *lhsOp,
                                            Operation *rhsOp) const {
  return operationAnalysis.containsEntry(runsLaterRelation, lhsOp, rhsOp) ||
         operationAnalysis.containsEntry(runsEarlierRelation, lhsOp, rhsOp) ||
         operationAnalysis.containsEntry(structurallyDisjointRelation, lhsOp,
                                         rhsOp);
}

AsyncDataflowDependencies::ValueSet
AsyncDataflowDependencies::exclusiveSet(Value indicator) const {
  // TODO: Implementation pending.
  return valueAnalysis.index->emptySet();
}

bool AsyncDataflowDependencies::isNonSelfOverlapping(Value value) const {
  Operation *op = value.getDefiningOp();
  if (!op)
    // We can only talk about op results.
    return false;

  if (isAtMostOnce(op))
    // Structurally impossible to execute op more than once.
    return true;

  // TODO: Remaining implementation pending.
  return false;
}

void AsyncDataflowDependencies::print(llvm::raw_ostream &os) {
  valueAnalysis.printRelation(os, "sameAs", sameAsRelation);
  valueAnalysis.printRelation(os, "maybeSameAs", maybeSameAsRelation);
  valueAnalysis.printPredicate(os, "isAlwaysReady", alwaysReadyPredicate);
  valueAnalysis.printRelation(os, "impliesReady", impliesReadyRelation);
  operationAnalysis.printRelation(os, "structurallyDisjoint",
                                  structurallyDisjointRelation);
  operationAnalysis.printRelation(os, "runsLater", runsLaterRelation);
  operationAnalysis.printRelation(os, "runsEarlier", runsEarlierRelation);
}

//===----------------------------------------------------------------------===//
// Core relations
//===----------------------------------------------------------------------===//

// Rules are just the control flow rules.
void AsyncDataflowDependencies::flowSameAs(Operation *op,
                                           ValueDomainMap &domainMap) {
  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op))
    valueAnalysis.flowControlFlow(
        cfOp, domainMap,
        std::bind(&AsyncDataflowDependencies::flowSameAs, this, _1, _2));
  else
    valueAnalysis.flowDefault(op, domainMap);
}

void AsyncDataflowDependencies::setupSameAs() {
  LLVM_DEBUG(llvm::dbgs() << "setupSameAs\n");
  ValueAnalysis::DomainMap domainMap(
      sameAsRelation, ValueAnalysis::kBottomIsFull,
      valueAnalysis.index->emptySet(), valueAnalysis.index->fullSet());
  valueAnalysis.flowModel(
      valueAnalysis.index->model, domainMap,
      std::bind(&AsyncDataflowDependencies::flowSameAs, this, _1, _2));
  valueAnalysis.addRelationToEquivalence(sameAsEquivalence, sameAsRelation);
}

// Rules are just the control flow rules.
void AsyncDataflowDependencies::flowMaybeSameAs(Operation *op,
                                                ValueDomainMap &domainMap) {
  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op))
    valueAnalysis.flowControlFlow(
        cfOp, domainMap,
        std::bind(&AsyncDataflowDependencies::flowMaybeSameAs, this, _1, _2));
  else
    valueAnalysis.flowDefault(op, domainMap);
}

void AsyncDataflowDependencies::setupMaybeSameAs() {
  LLVM_DEBUG(llvm::dbgs() << "setupMaybeSameAs\n");
  ValueDomainMap domainMap(maybeSameAsRelation, ValueAnalysis::kBottomIsEmpty,
                           valueAnalysis.index->emptySet(),
                           valueAnalysis.index->fullSet());
  valueAnalysis.flowModel(
      valueAnalysis.index->model, domainMap,
      std::bind(&AsyncDataflowDependencies::flowMaybeSameAs, this, _1, _2));
  valueAnalysis.addRelationToEquivalence(maybeSameAsEquivalence,
                                         maybeSameAsRelation);
}

/// Implements the rules:
///  - Given
///      %x = grt.chain.create (%y1, ..., %yn)
///    where isAlwaysReady(%yi) for all i then isAlwaysReady(%x)
///  - The control flow rules.
void AsyncDataflowDependencies::flowAlwaysReady(Operation *op,
                                                ValueDomainMap &domainMap) {
  if (auto chainCreateOp = dyn_cast<GRT::ChainCreateOp>(op)) {
    // This is equivalent to:
    //   if (isTrue(o) for all o in mergeOp.getOperands())
    //     setTrue(mergeOp.getResult())
    // but writen to mirror how flowControlFlow works with the relation.
    ValueSet set = domainMap.bottom;
    for (Value operand : chainCreateOp.getOperands())
      domainMap.leastUpperBound(
          set, valueAnalysis.getEntriesAndId(domainMap, operand));
    valueAnalysis.setEntries(domainMap, chainCreateOp.getResult(), set);
  } else if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
    valueAnalysis.flowControlFlow(
        cfOp, domainMap,
        std::bind(&AsyncDataflowDependencies::flowAlwaysReady, this, _1, _2));
  } else {
    valueAnalysis.flowDefault(op, domainMap);
  }
}

void AsyncDataflowDependencies::setupAlwaysReady() {
  LLVM_DEBUG(llvm::dbgs() << "setupAlwaysReady\n");
  ValueRelation alwaysReadyRelation;
  ValueDomainMap domainMap(
      alwaysReadyRelation, ValueAnalysis::kBottomIsFull,
      valueAnalysis.index->emptySet(),
      valueAnalysis.index->unitSet(valueAnalysis.index->distinguished));
  valueAnalysis.flowModel(
      valueAnalysis.index->model, domainMap,
      std::bind(&AsyncDataflowDependencies::flowAlwaysReady, this, _1, _2));
  valueAnalysis.addToPredicate(alwaysReadyPredicate, alwaysReadyRelation);
}

/// Implements the rules:
///  - Given
///      ..., %lhs, ... = op(..., %rhs, ...)
///    if %lhs is an 'indicating result' and %rhs is a 'gating operand'
///    then impliesReady(%lhs, %rhs)
///  - the control flow rules
void AsyncDataflowDependencies::flowImpliesReady(Operation *op,
                                                 ValueDomainMap &domainMap) {
  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
    valueAnalysis.flowControlFlow(
        cfOp, domainMap,
        std::bind(&AsyncDataflowDependencies::flowImpliesReady, this, _1, _2));
  } else {
    ValueSet set = domainMap.top;
    for (Value rhs : getGatingOperands(op))
      domainMap.greatestLowerBound(
          set, valueAnalysis.getEntriesAndId(domainMap, rhs));
    for (Value lhs : getIndicatingResults(op))
      valueAnalysis.setEntries(domainMap, lhs, set);
  }
}

void AsyncDataflowDependencies::setupImpliesReady() {
  LLVM_DEBUG(llvm::dbgs() << "setupImpliesReady\n");
  // Given:
  //    %a = if ... {
  //      %w = ...
  //      mef.output %x
  //    } else {
  //      %y = ...
  //      mef.output %z
  //    }
  // where impliesReady(%z, %y), we have two possibly interpretations of
  // impliesReady(%a, %y):
  //  - the relation can only hold if %y was actually created, thus it
  //    must be false in case the 'then' sub-block was taken.
  //  - the relation is allowed to hold 'vacuously' if %y was never created,
  //    thus can be true.
  //
  // We take the second interpretation so that impliesReady can give us
  // more entries in the runsEarlier relation, which in turn gives us more
  // entries in the exclusive relation.
  //
  // Thus, we turn on the handleDisjointBranches option for this dataflow.
  ValueDomainMap domainMap(impliesReadyRelation, ValueAnalysis::kBottomIsFull,
                           valueAnalysis.index->emptySet(),
                           valueAnalysis.index->fullSet(),
                           /*handleDisjointBranches=*/true);
  valueAnalysis.flowModel(
      valueAnalysis.index->model, domainMap,
      std::bind(&AsyncDataflowDependencies::flowImpliesReady, this, _1, _2));
  inverseImpliesReadyRelation = valueAnalysis.inverse(impliesReadyRelation);
}

void AsyncDataflowDependencies::accumStructurallyDisjointForBlock(
    OperationRelation &relation, Block *block) {
  block->walk([&](GRT::GRTControlFlowInterface cfOp) {
    if (cfOp.getFlavor() != GRT::kBranchLike)
      return;

    // Collect all the ops for each branch sub-block (including nested
    // sub-blocks). Note that these ops may be inside loops, including loops on
    // the outside of this cfOp, and loops between the op and this cfOp.
    // However, thankfully loops are 'non overlapping', so we know only one
    // instance of the op can be executing at any one time.
    SmallVector<SmallVector<Operation *>> allOpsPerBranch;
    for (Region &region : cfOp->getRegions()) {
      SmallVector<Operation *> &allOps = allOpsPerBranch.emplace_back();
      region.walk([&](Operation *op) {
        if (getIndicatingResults(op).empty())
          // We are only interested in ops with indicating results since that's
          // the only way we have to track their execution.
          return;
        allOps.emplace_back(op);
      });
    }

    // For all pairs of distinct blocks...
    size_t n = cfOp->getNumRegions();
    for (size_t i = 0; i < n; ++i) {
      size_t numLhs = allOpsPerBranch[i].size();
      for (size_t j = 0; j < n; ++j) {
        if (i == j)
          continue;
        size_t numRhs = allOpsPerBranch[j].size();
        // For all pairs of ops from disjoint blocks...
        for (size_t k = 0; k < numLhs; ++k) {
          Operation *lhsOp = allOpsPerBranch[i][k];
          for (size_t l = 0; l < numRhs; ++l) {
            Operation *rhsOp = allOpsPerBranch[j][l];
            // These ops can never both be spliced into the overall
            // execution graph.
            operationAnalysis.add(relation, lhsOp, rhsOp);
          }
        }
      }
    }
  });
}

void AsyncDataflowDependencies::setupStructurallyDisjoint() {
  LLVM_DEBUG(llvm::dbgs() << "setupStructurallyDisjoint\n");
  accumStructurallyDisjointForBlock(
      structurallyDisjointRelation,
      operationAnalysis.index->model.getInitBlock());
  accumStructurallyDisjointForBlock(
      structurallyDisjointRelation,
      operationAnalysis.index->model.getExecuteBlock());
}

AsyncDataflowDependencies::OperationSet
AsyncDataflowDependencies::runsLaterSet(Operation *earlierOp) const {
  OperationSet result = operationAnalysis.index->emptySet();
  // For each result of earlier op which indicates op has FINISHED...
  for (Value indicatingResult : getIndicatingResults(earlierOp)) {
    // Find all values which imply the indicatingResult is ready (including
    // the indicatingResult itself).
    ValueSet set = downwardsImpliesReadySet(indicatingResult);
    // Any use of a value in that set as a gating operand forces the
    // using operation to BEGIN only after earlierOp has FINISHED.
    for (Value user : valueAnalysis.index->setToVector(set)) {
      for (OpOperand &opOperand : user.getUses()) {
        if (opOperand.getOwner()->getNumResults() == 0)
          // Ignore ops with no results since we can't track them.
          continue;
        if (isGatingOperand(&opOperand))
          operationAnalysis.index->addToSet(result, opOperand.getOwner());
      }
    }
  }
  return result;
}

void AsyncDataflowDependencies::setupRunsLater() {
  LLVM_DEBUG(llvm::dbgs() << "setupRunsLater\n");
  for (Operation *op : operationAnalysis.index->getAllVs()) {
    if (op == operationAnalysis.index->distinguished)
      continue;
    OperationSet set = runsLaterSet(op);
    if (set.count())
      runsLaterRelation.insert({op, set});
  }
  runsEarlierRelation = operationAnalysis.inverse(runsLaterRelation);
}
