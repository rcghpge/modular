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
// AsyncDataflowDependencies
//===----------------------------------------------------------------------===//

AsyncDataflowDependencies::AsyncDataflowDependencies(IndexSet<Value> *index)
    : ForwardsRelationDataflowAnalysis(index) {
  sameAsEquivalence = index->identityEquivalence();
  maybeSameAsEquivalence = index->identityEquivalence();
  setupSameAs(index->model);
  setupMaybeSameAs(index->model);
  setupAlwaysReady(index->model);
  setupImpliesReady(index->model);
  setupConsumedWhenReady();
}

bool AsyncDataflowDependencies::sameAs(Value lhs, Value rhs) const {
  return index->areEquivalent(sameAsEquivalence, lhs, rhs);
}

AsyncDataflowDependencies::Set
AsyncDataflowDependencies::sameAsEquivalenceClass(Value lhs) const {
  return index->getEquivalenceClass(sameAsEquivalence, lhs);
}

bool AsyncDataflowDependencies::maybeSameAs(Value lhs, Value rhs) const {
  return index->areEquivalent(maybeSameAsEquivalence, lhs, rhs);
}

bool AsyncDataflowDependencies::isAlwaysReady(Value value) const {
  return isTrue(sameAsEquivalence, alwaysReadyRelation, value);
}

bool AsyncDataflowDependencies::impliesReady(Value lhs, Value rhs) const {
  return containsEntryOrId(sameAsEquivalence, impliesReadyRelation, lhs, rhs);
}

AsyncDataflowDependencies::Set
AsyncDataflowDependencies::downwardsImpliesReadySet(Value rhs) const {
  return getEntries(sameAsEquivalence, inverseImpliesReadyRelation, rhs);
}

AsyncDataflowDependencies::Set
AsyncDataflowDependencies::upwardsImpliesReadySet(Value lhs) const {
  return getEntries(sameAsEquivalence, impliesReadyRelation, lhs);
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

AsyncDataflowDependencies::Set
AsyncDataflowDependencies::consumedWhenReady(Value lhs) const {
  return getEntries(sameAsEquivalence, consumedWhenReadyRelation, lhs);
}

AsyncDataflowDependencies::Set
AsyncDataflowDependencies::indicatesConsumed(Value lhs) const {
  return getEntries(sameAsEquivalence, indicatesConsumedRelation, lhs);
}

void AsyncDataflowDependencies::print(llvm::raw_ostream &os) {
  printEquivalence(os, "sameAs", sameAsEquivalence);
  printEquivalence(os, "maybeSameAs", maybeSameAsEquivalence);
  printPredicate(os, "isAlwaysReady",
                 toStableForm(sameAsEquivalence, alwaysReadyRelation));
  printRelation(os, "impliesReady",
                toStableForm(sameAsEquivalence, impliesReadyRelation));
  printRelation(os, "consumedWhenReady",
                toStableForm(sameAsEquivalence, consumedWhenReadyRelation));
}

// Rules are just the control flow rules.
void AsyncDataflowDependencies::flowPreSameAs(Operation *op,
                                              DomainMap &domainMap) {
  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op))
    flowControlFlow(
        cfOp, domainMap,
        std::bind(&AsyncDataflowDependencies::flowPreSameAs, this, _1, _2));
  else
    flowDefault(op, domainMap);
}

void AsyncDataflowDependencies::setupSameAs(ModelOp modelOp) {
  LLVM_DEBUG(llvm::dbgs() << "setupSameAs\n");
  Relation relation;
  DomainMap domainMap(sameAsEquivalence, relation, kBottomIsFull,
                      index->emptySet(), index->fullSet());
  flowModel(modelOp, domainMap,
            std::bind(&AsyncDataflowDependencies::flowPreSameAs, this, _1, _2));
  addRelationToEquivalence(sameAsEquivalence, relation);
}

// Rules are just the control flow rules.
void AsyncDataflowDependencies::flowPreMaybeSameAs(Operation *op,
                                                   DomainMap &domainMap) {
  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op))
    flowControlFlow(
        cfOp, domainMap,
        std::bind(&AsyncDataflowDependencies::flowPreSameAs, this, _1, _2));
  else
    flowDefault(op, domainMap);
}

void AsyncDataflowDependencies::setupMaybeSameAs(ModelOp modelOp) {
  LLVM_DEBUG(llvm::dbgs() << "setupMaybeSameAs\n");
  Relation relation;
  DomainMap domainMap(maybeSameAsEquivalence, relation, kBottomIsEmpty,
                      index->emptySet(), index->fullSet());
  flowModel(
      modelOp, domainMap,
      std::bind(&AsyncDataflowDependencies::flowPreMaybeSameAs, this, _1, _2));
  addRelationToEquivalence(maybeSameAsEquivalence, relation);
}

/// Implements the rules:
///  - Given
///      %x = grt.chain.create (%y1, ..., %yn)
///    where isAlwaysReady(%yi) for all i then isAlwaysReady(%x)
///  - The control flow rules.
///  - Closure under sameAs.
void AsyncDataflowDependencies::flowAlwaysReady(Operation *op,
                                                DomainMap &domainMap) {
  if (auto chainCreateOp = dyn_cast<GRT::ChainCreateOp>(op)) {
    // This is equivalent to:
    //   if (isTrue(o) for all o in mergeOp.getOperands())
    //     setTrue(mergeOp.getResult())
    // but writen to mirror how flowControlFlow works with the relation.
    Set set = domainMap.bottom;
    for (Value operand : chainCreateOp.getOperands())
      domainMap.leastUpperBound(set, getEntries(domainMap, operand));
    setEntriesAndId(domainMap, chainCreateOp.getResult(), set);
  } else if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
    flowControlFlow(
        cfOp, domainMap,
        std::bind(&AsyncDataflowDependencies::flowAlwaysReady, this, _1, _2));
  } else {
    flowDefault(op, domainMap);
  }
}

void AsyncDataflowDependencies::setupAlwaysReady(ModelOp modelOp) {
  LLVM_DEBUG(llvm::dbgs() << "setupAlwaysReady\n");
  DomainMap domainMap(sameAsEquivalence, alwaysReadyRelation, kBottomIsFull,
                      index->emptySet(), index->unitSet(index->distinguished));
  flowModel(
      modelOp, domainMap,
      std::bind(&AsyncDataflowDependencies::flowAlwaysReady, this, _1, _2));
}

/// Implements the rules:
///  - Given
///      ..., %lhs, ... = op(..., %rhs, ...)
///    if %lhs is an 'indicating result' and %rhs is a 'gating operand'
///    then impliesReady(%lhs, %rhs)
///  - the control flow rules
///  - closure under sameAs
void AsyncDataflowDependencies::flowImpliesReady(Operation *op,
                                                 DomainMap &domainMap) {
  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
    flowControlFlow(
        cfOp, domainMap,
        std::bind(&AsyncDataflowDependencies::flowImpliesReady, this, _1, _2));
  } else {
    for (Value lhs : getIndicatingResults(op)) {
      Set set = domainMap.top;
      for (Value rhs : getGatingOperands(op))
        domainMap.greatestLowerBound(set, getEntries(domainMap, rhs));
      setEntriesAndId(domainMap, lhs, set);
    }
  }
}

void AsyncDataflowDependencies::setupImpliesReady(ModelOp modelOp) {
  LLVM_DEBUG(llvm::dbgs() << "setupImpliesReady\n");
  DomainMap domainMap(sameAsEquivalence, impliesReadyRelation, kBottomIsFull,
                      index->emptySet(), index->fullSet());
  flowModel(
      modelOp, domainMap,
      std::bind(&AsyncDataflowDependencies::flowImpliesReady, this, _1, _2));
  inverseImpliesReadyRelation = inverse(impliesReadyRelation);
}

AsyncDataflowDependencies::Set
AsyncDataflowDependencies::consumedSet(Value value) const {
  Set set = index->emptySet();
  for (Value maybeSameValue : index->setToVector(
           index->getEquivalenceClass(maybeSameAsEquivalence, value))) {
    for (OpOperand &operand : maybeSameValue.getUses()) {
      if (Value representativeIndicatingResult =
              getRepresentativeIndicatingResult(operand.getOwner())) {
        // When representativeIndicatingResult is ready we know this use of
        // maybeSameValue (which could be the same as value) has been
        // consumed.
        // Note that the op may have multiple indicating results, but we don't
        // want to start trying to represent conjunctions-of-disjunctions or
        // similar craziness. So heuristically choose just one, typically the
        // chain result.
        index->addToSet(set,
                        index->getCanonical(sameAsEquivalence,
                                            representativeIndicatingResult));
      }
    }
  }
  return set;
}

AsyncDataflowDependencies::Set
AsyncDataflowDependencies::impliesAllReadySet(const Set &values) const {
  if (values.count() == 0)
    return index->emptySet();
  Set intersection = index->fullSet();
  for (Value value : index->setToVector(values))
    intersection &= downwardsImpliesReadySet(value);
  return intersection;
}

void AsyncDataflowDependencies::setupConsumedWhenReady() {
  LLVM_DEBUG(llvm::dbgs() << "setupConsumedWhenReady\n");
  DomainMap domainMap(sameAsEquivalence, consumedWhenReadyRelation,
                      kBottomIsEmpty, index->emptySet(), index->fullSet());
  SmallVector<Value> canons =
      index->setToVector(index->getAllCanonicalVs(sameAsEquivalence));
  for (Value lhs : canons)
    setEntriesAndId(domainMap, lhs, impliesAllReadySet(consumedSet(lhs)));
  indicatesConsumedRelation = inverse(consumedWhenReadyRelation);
}

//===----------------------------------------------------------------------===//
// AsyncOperationDependencies
//===----------------------------------------------------------------------===//

AsyncOperationDependencies::AsyncOperationDependencies(
    IndexSet<Operation *> *indexSet,
    AsyncDataflowDependencies *valueDependencies)
    : RelationDataflowAnalysisBase(indexSet),
      valueDependencies(valueDependencies) {
  setupRunsLater(index->model);
}

AsyncOperationDependencies::Set
AsyncOperationDependencies::runsLaterOps(Operation *earlierOp) const {
  return getEntries(equivalence, runsLaterRelation, earlierOp);
}

AsyncOperationDependencies::Set
AsyncOperationDependencies::runsEarlierOps(Operation *laterOp) const {
  return getEntries(equivalence, runsEarlierRelation, laterOp);
}

bool AsyncOperationDependencies::areDisjoint(Operation *lhsOp,
                                             Operation *rhsOp) const {
  return index->setContains(runsLaterOps(lhsOp), rhsOp) ||
         index->setContains(runsEarlierOps(lhsOp), rhsOp);
}

void AsyncOperationDependencies::print(llvm::raw_ostream &os) {
  printRelation(os, "runsLater", runsLaterRelation);
}

AsyncOperationDependencies::Set
AsyncOperationDependencies::runsLaterSet(Operation *earlierOp) const {
  Set result = index->emptySet();
  for (Value indicatingResult : getIndicatingResults(earlierOp)) {
    // Find all (canonical) values which imply the indicatingResult is ready
    // (including indicatingResult itself).
    Set set = valueDependencies->downwardsImpliesReadySet(indicatingResult);
    // Expand to all values under the sameAs equivalence.
    set = valueDependencies->index->closeSetUnderEquivalence(
        valueDependencies->sameAsEquivalence, set);
    // Any use of a value in that set as a gating operand forces the
    // using operation to run after earlierOp.
    for (Value user : valueDependencies->index->setToVector(set)) {
      for (OpOperand &opOperand : user.getUses()) {
        if (isGatingOperand(&opOperand))
          index->addToSet(result, opOperand.getOwner());
      }
    }
  }
  return result;
}

void AsyncOperationDependencies::setupRunsLaterForBlock(DomainMap &domainMap,
                                                        Block *block) {
  for (Operation &op : block->getOperations()) {
    setEntriesAndId(domainMap, &op, runsLaterSet(&op));
    for (Region &subRegion : op.getRegions())
      setupRunsLaterForBlock(domainMap, &subRegion.front());
  }
}

void AsyncOperationDependencies::setupRunsLater(ModelOp modelOp) {
  equivalence = index->identityEquivalence();
  DomainMap domainMap(equivalence, runsLaterRelation, kBottomIsEmpty,
                      index->emptySet(), index->fullSet());
  LLVM_DEBUG(llvm::dbgs() << "setupRunsLater\n");
  setupRunsLaterForBlock(domainMap, modelOp.getInitBlock());
  setupRunsLaterForBlock(domainMap, modelOp.getExecuteBlock());
  runsEarlierRelation = inverse(runsLaterRelation);
}
