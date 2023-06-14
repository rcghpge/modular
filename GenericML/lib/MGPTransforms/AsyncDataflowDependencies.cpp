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
  setupSameAs(index->model);
  setupAlwaysReady(index->model);
  setupImpliesReady(index->model);
}

bool AsyncDataflowDependencies::sameAs(Value lhs, Value rhs) const {
  return index->areEquivalent(equivalence, lhs, rhs);
}

AsyncDataflowDependencies::Set
AsyncDataflowDependencies::sameAsEquivalenceClass(Value lhs) const {
  return index->getEquivalenceClass(equivalence, lhs);
}

bool AsyncDataflowDependencies::isAlwaysReady(Value value) const {
  return isTrue(alwaysReadyRelation, value);
}

bool AsyncDataflowDependencies::impliesReady(Value lhs, Value rhs) const {
  return containsEntryOrId(impliesReadyRelation, lhs, rhs);
}

AsyncDataflowDependencies::Set
AsyncDataflowDependencies::downwardsImpliesReadySet(Value rhs) const {
  return getEntries(inverseImpliesReadyRelation, rhs);
}

AsyncDataflowDependencies::Set
AsyncDataflowDependencies::upwardsImpliesReadySet(Value lhs) const {
  return getEntries(impliesReadyRelation, lhs);
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

void AsyncDataflowDependencies::print(llvm::raw_ostream &os) {
  printEquivalence(os, "sameAs", equivalence);
  printPredicate(os, "isAlwaysReady", toStableForm(alwaysReadyRelation));
  printRelation(os, "impliesReady", toStableForm(impliesReadyRelation));
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
  DomainMap domainMap(relation, index->emptySet(), index->fullSet());
  flowModel(modelOp, domainMap,
            std::bind(&AsyncDataflowDependencies::flowPreSameAs, this, _1, _2));
  addRelationToEquivalence(relation);
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
    Set set = domainMap.fullSet;
    for (Value operand : chainCreateOp.getOperands())
      domainMap.setIntersection(set, getEntries(domainMap, operand));
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
  DomainMap domainMap(alwaysReadyRelation, index->emptySet(),
                      index->unitSet(index->distinguished));
  flowModel(
      modelOp, domainMap,
      std::bind(&AsyncDataflowDependencies::flowAlwaysReady, this, _1, _2));
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
///  - the control flow rules
///  - closure under sameAs
void AsyncDataflowDependencies::flowImpliesReady(Operation *op,
                                                 DomainMap &domainMap) {
  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
    flowControlFlow(
        cfOp, domainMap,
        std::bind(&AsyncDataflowDependencies::flowImpliesReady, this, _1, _2));
  } else {
    for (Value lhs : op->getResults()) {
      Set theUnion = domainMap.emptySet;
      for (Value rhs : op->getOperands()) {
        // Primitive relation relation.
        if (primitiveImpliesReady(lhs, rhs))
          domainMap.setUnion(theUnion, getEntries(domainMap, rhs));
      }
      setEntriesAndId(domainMap, lhs, theUnion);
    }
  }
}

void AsyncDataflowDependencies::setupImpliesReady(ModelOp modelOp) {
  LLVM_DEBUG(llvm::dbgs() << "setupImpliesReady\n");
  DomainMap domainMap(impliesReadyRelation, index->emptySet(),
                      index->fullSet());
  flowModel(
      modelOp, domainMap,
      std::bind(&AsyncDataflowDependencies::flowImpliesReady, this, _1, _2));
  inverseImpliesReadyRelation = inverse(impliesReadyRelation);
}
