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

#define DEBUG_TYPE "mgp-dataflow"

AsyncDataflowDependencies::AsyncDataflowDependencies(IndexSet<Value> *index)
    : ForwardsRelationDataflowAnalysis(index) {
  setupSameAs(index->model);
  setupAlwaysReady(index->model);
  setupImpliesReady(index->model);
}

bool AsyncDataflowDependencies::sameAs(Value lhs, Value rhs) const {
  return index->areEquivalent(equivalence, lhs, rhs);
}

bool AsyncDataflowDependencies::isAlwaysReady(Value value) const {
  return isTrue(alwaysReadyRelation, value);
}

bool AsyncDataflowDependencies::impliesReady(Value lhs, Value rhs) const {
  return containsEntryOrId(impliesReadyRelation, lhs, rhs);
}

void AsyncDataflowDependencies::simplify(SmallVectorImpl<Value> &values) const {
  // CAUTION: Don't underflow ip1.

  for (size_t n = values.size(), ip1 = values.size(); ip1 > 0; --ip1) {
    size_t i = ip1 - 1;
    bool redundant = isAlwaysReady(values[i]);
    if (!redundant) {
      for (size_t j = 0; j < n; ++j) {
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

// Rules are just the control flow rules.
void AsyncDataflowDependencies::flowPreSameAs(DomainMap &domainMap,
                                              Block *block) {
  for (Operation *op : getDataflowOrder(block)) {
    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
      flowControlFlow(cfOp, domainMap,
                      [&](DomainMap &subDomainMap, Block *subBlock) {
                        return flowPreSameAs(subDomainMap, subBlock);
                      });
    } else {
      flowDefault(op, domainMap);
    }
  }
}

void AsyncDataflowDependencies::setupSameAs(ModelOp modelOp) {
  Relation relation;
  DomainMap domainMap(relation, index->fullSet(), index->emptySet());
  flowModel(modelOp, domainMap, [&](DomainMap &subDomainMap, Block *subBlock) {
    return flowPreSameAs(subDomainMap, subBlock);
  });
  addRelationToEquivalence(relation);
}

/// Implements the rules:
///  - Given
///      %x = grt.chain.create
///    then isAlwaysReady(%x)
///  - Given
///      %x = grt.chain.merge (%y1, ..., %yn)
///    where isAlwaysReady(%yi) for all i then isAlwaysReady(%x)
///  - The control flow rules.
///  - Closure under sameAs.
void AsyncDataflowDependencies::flowAlwaysReady(DomainMap &domainMap,
                                                Block *block) {
  for (Operation *op : getDataflowOrder(block)) {
    if (auto createOp = dyn_cast<GRT::ChainCreateOp>(op)) {
      setTrue(domainMap, createOp.getResult());
    } else if (auto mergeOp = dyn_cast<GRT::ChainMergeOp>(op)) {
      // This is equivalent to:
      //   if (isTrue(o) for all o in mergeOp.getOperands())
      //     setTrue(mergeOp.getResult())
      // but writen to mirror how flowControlFlow works with the relation.
      Set theIntersection = domainMap.fullSet;
      for (Value operand : mergeOp.getOperands())
        domainMap.setIntersection(theIntersection,
                                  getEntries(domainMap, operand));
      setEntriesAndId(domainMap, mergeOp.getResult(), theIntersection);
    } else if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
      flowControlFlow(cfOp, domainMap,
                      [&](DomainMap &subDomainMap, Block *subBlock) {
                        return flowAlwaysReady(subDomainMap, subBlock);
                      });
    } else {
      flowDefault(op, domainMap);
    }
  }
}

void AsyncDataflowDependencies::setupAlwaysReady(ModelOp modelOp) {
  DomainMap domainMap(alwaysReadyRelation, index->unitSet(index->distinguished),
                      index->emptySet());
  flowModel(modelOp, domainMap, [&](DomainMap &subDomainMap, Block *subBlock) {
    return flowAlwaysReady(subDomainMap, subBlock);
  });
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
void AsyncDataflowDependencies::flowImpliesReady(DomainMap &domainMap,
                                                 Block *block) {
  for (Operation *op : getDataflowOrder(block)) {
    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
      flowControlFlow(cfOp, domainMap,
                      [&](DomainMap &subDomainMap, Block *subBlock) {
                        return flowImpliesReady(subDomainMap, subBlock);
                      });
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
}

void AsyncDataflowDependencies::setupImpliesReady(ModelOp modelOp) {
  DomainMap domainMap(impliesReadyRelation, index->fullSet(),
                      index->emptySet());
  flowModel(modelOp, domainMap, [&](DomainMap &subDomainMap, Block *subBlock) {
    return flowImpliesReady(subDomainMap, subBlock);
  });
}
