//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "./AsyncDataflowDependencies.h"

#include "GraphRT/GRTDialect/GRTExtendsLifetimeInterface.h"
#include "GraphRT/GRTDialect/GRTMayAliasInterface.h"
#include "GraphRT/GRTDialect/GRTOps.h"
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
    IndexSet<Value> *valueIndex, IndexSet<Operation *> *operationIndex,
    AnalysisLevel analysisLevel)
    : valueAnalysis(valueIndex), operationAnalysis(operationIndex),
      sameAsEquivalence(valueIndex->identityEquivalence()),
      maybeSameAsEquivalence(valueIndex->identityEquivalence()),
      approximateMaybeSameAsEquivalence(valueIndex->identityEquivalence()),
      alwaysReadyPredicate(valueIndex->emptySet()) {
  setupSameAs();
  setupMaybeSameAs();
  setupApproximateMaybeSameAs();
  setupAlwaysReady();
  setupImpliesReady();
  if (analysisLevel >= kDisjoint) {
    setupStructurallyDisjoint();
    setupRunsLater();
  }
  if (analysisLevel >= kExclusive) {
    setupExclusive();
  }
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

bool AsyncDataflowDependencies::approximateMaybeSameAs(Value lhs,
                                                       Value rhs) const {
  return valueAnalysis.index->areEquivalent(approximateMaybeSameAsEquivalence,
                                            lhs, rhs);
}

AsyncDataflowDependencies::ValueSet
AsyncDataflowDependencies::approximateMaybeSameAsSet(Value lhs) const {
  return valueAnalysis.index->getEquivalenceClass(
      approximateMaybeSameAsEquivalence, lhs);
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
  return valueAnalysis.getEntries(exclusiveRelation, indicator);
}

bool AsyncDataflowDependencies::isNonSelfOverlapping(Value value) const {
  Operation *op = value.getDefiningOp();
  if (!op) {
    // We can only talk about op results.
    return false;
  }

  if (isAtMostOnce(op)) {
    // Structurally impossible to execute op more than once.
    return true;
  }

  ValueSet dead = exclusiveSet(value);

  // The only way value could still be alive is if some alias of it is
  // fed back via some enclosing kLoop control flow op, and that alias
  // is not dead when value is defined.
  return llvm::all_of(allLoopBoundArgs(op), [&](Value arg) {
    return !maybeSameAs(value, arg) ||
           valueAnalysis.index->setContains(dead, arg);
  });
}

void AsyncDataflowDependencies::print(llvm::raw_ostream &os) {
  valueAnalysis.printRelation(os, "sameAs", sameAsRelation);
  valueAnalysis.printRelation(os, "maybeSameAs", maybeSameAsRelation);
  valueAnalysis.printEquivalence(os, "maybeSameAsEquivalence",
                                 maybeSameAsEquivalence);
  valueAnalysis.printRelation(os, "approximateMaybeSameAs",
                              approximateMaybeSameAsRelation);
  valueAnalysis.printEquivalence(os, "approximateMaybeSameAsEquivalence",
                                 approximateMaybeSameAsEquivalence);
  valueAnalysis.printPredicate(os, "isAlwaysReady", alwaysReadyPredicate);
  valueAnalysis.printRelation(os, "impliesReady", impliesReadyRelation);
  operationAnalysis.printRelation(os, "structurallyDisjoint",
                                  structurallyDisjointRelation);
  operationAnalysis.printRelation(os, "runsLater", runsLaterRelation);
  operationAnalysis.printRelation(os, "runsEarlier", runsEarlierRelation);
  valueAnalysis.printRelation(os, "exclusive", exclusiveRelation);
}

//===----------------------------------------------------------------------===//
// Core relations
//===----------------------------------------------------------------------===//

// Rules are just the control flow rules.
void AsyncDataflowDependencies::flowSameAs(Operation *op,
                                           ValueDomainMap &domainMap) {
  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
    valueAnalysis.flowControlFlow(
        cfOp, domainMap,
        std::bind(&AsyncDataflowDependencies::flowSameAs, this, _1, _2));
  } else {
    valueAnalysis.flowDefault(op, domainMap);
  }
}

void AsyncDataflowDependencies::setupSameAs() {
  LLVM_DEBUG(llvm::dbgs() << "setupSameAs\n");
  ValueAnalysis::DomainMap domainMap(
      sameAsRelation, ValueAnalysis::kLubIsIntersection,
      valueAnalysis.index->emptySet(), valueAnalysis.index->fullSet());
  valueAnalysis.flowModel(
      valueAnalysis.index->model, domainMap,
      std::bind(&AsyncDataflowDependencies::flowSameAs, this, _1, _2));
  valueAnalysis.addRelationToEquivalence(sameAsEquivalence, sameAsRelation);
}

// Rules are:
//  - Given
//      ..., %lhs, ... = op (..., %rhs, ...)
//    where op has the GRTMayAliasInterface or GRTExtendsLifetimeInterface,
//    then maybeSameAs(%lhs, %rhs). Note that maybeSameAs does not distinguish
//    direct aliasing from lifetime extension, so the two interfaces are
//    currently equivalent.
//  - The control flow rules.
void AsyncDataflowDependencies::flowMaybeSameAs(Operation *op,
                                                ValueDomainMap &domainMap) {
  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
    valueAnalysis.flowControlFlow(
        cfOp, domainMap,
        std::bind(&AsyncDataflowDependencies::flowMaybeSameAs, this, _1, _2));
  } else if (isa<GRT::GRTExtendsLifetimeInterface>(op) ||
             isa<GRT::GRTMayAliasInterface>(op)) {
    // If an op declares that its results may alias its arguments, simply
    // expand the maybeSameAs relation to include that.
    //
    // If an op declares that its arguments may be lifetime extended to
    // account for its results, treat that as if they were possibly aliased.
    // We thus approximate this by just treating all the results and arguments
    // to have the same worst-case lifetime.
    ValueSet set = domainMap.bottom;
    for (Value operand : op->getOperands()) {
      domainMap.leastUpperBound(
          set, valueAnalysis.getEntriesAndId(domainMap, operand));
    }
    for (Value result : op->getResults())
      valueAnalysis.setEntries(domainMap, result, set);
  } else {
    valueAnalysis.flowDefault(op, domainMap);
  }
}

void AsyncDataflowDependencies::setupMaybeSameAs() {
  LLVM_DEBUG(llvm::dbgs() << "setupMaybeSameAs\n");
  ValueDomainMap domainMap(maybeSameAsRelation, ValueAnalysis::kLubIsUnion,
                           valueAnalysis.index->emptySet(),
                           valueAnalysis.index->fullSet());
  valueAnalysis.flowModel(
      valueAnalysis.index->model, domainMap,
      std::bind(&AsyncDataflowDependencies::flowMaybeSameAs, this, _1, _2));
  valueAnalysis.addRelationToEquivalence(maybeSameAsEquivalence,
                                         maybeSameAsRelation);
}

void AsyncDataflowDependencies::setupApproximateMaybeSameAs() {
  LLVM_DEBUG(llvm::dbgs() << "setupApproximateMaybeSameAs\n");
  ValueDomainMap domainMap(
      approximateMaybeSameAsRelation, ValueAnalysis::kLubIsUnion,
      valueAnalysis.index->emptySet(), valueAnalysis.index->fullSet(),
      /*handleDisjointBranches=*/false, /*approximateLoops=*/true);
  // Note we can use the same transfer functions as for maybeSameAs, only
  // the 'approximateLoops' mode needs to be changed.
  valueAnalysis.flowModel(
      valueAnalysis.index->model, domainMap,
      std::bind(&AsyncDataflowDependencies::flowMaybeSameAs, this, _1, _2));
  valueAnalysis.addRelationToEquivalence(approximateMaybeSameAsEquivalence,
                                         approximateMaybeSameAsRelation);
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
    for (Value operand : chainCreateOp.getOperands()) {
      domainMap.leastUpperBound(
          set, valueAnalysis.getEntriesAndId(domainMap, operand));
    }
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
      alwaysReadyRelation, ValueAnalysis::kLubIsIntersection,
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
///    where %lhs is an 'indicating result' and %rhs is a 'gating operand'
///    then impliesReady(%lhs, %rhs)
///  - The control flow rules
///  - Given
///      ..., %lhs, ... = grt.if %rhs (...) ...
///    then impliesReady(%lhs, %rhs), since condition must have been ready
///    for grt.if to return.
///  - Given
///      ..., %lhs, ... = grt.repeat.i32 %rhs (...) ...
///    then impliesReady(%lhs, %rhs), since trip count must have been ready
///    for grt.repeat.i32 to return.
///  - Given
///      ..., %lhs, ... = grt.repeat.i32 ... {
///                         ...
///                         mef.output ..., %rhs, ...
///                       }
///    then impliesReady(%lhs, %rhs), since grt.repeat.i32 waits for all
///    loop body outputs to be ready after splicing it into the execution
///    graph.
///  - Given
///      ..., %lhs, ... = grt.while ... {
///                         ...
///                       } do {
///                         ...
///                         mef.output ..., %rhs, ...
///                      }
///    then impliesReady(%lhs, %rhs), since grt.while waits for all
///    loop body outputs to be ready after splicing it into the execution
///    graph.
void AsyncDataflowDependencies::flowImpliesReady(Operation *op,
                                                 ValueDomainMap &domainMap) {
  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
    valueAnalysis.flowControlFlow(
        cfOp, domainMap,
        std::bind(&AsyncDataflowDependencies::flowImpliesReady, this, _1, _2));
    // Union in the additional relation entries described above.
    for (Value lhs : cfOp->getResults()) {
      ValueSet set = valueAnalysis.getEntries(domainMap, lhs);
      // Condition/trip counts are strict.
      for (Value rhs : cfOp.getStrictExtraOperands()) {
        domainMap.greatestLowerBound(
            set, valueAnalysis.getEntriesAndId(domainMap, rhs));
      }
      if (cfOp.getFlavor() == GRT::kLoopLike) {
        // Loop body outputs are known to be ready.
        Block *lastBlock = &cfOp->getRegions().back().front();
        for (Value rhs : lastBlock->getTerminator()->getOperands()) {
          domainMap.greatestLowerBound(
              set, valueAnalysis.getEntriesAndId(domainMap, rhs));
        }
      }
      valueAnalysis.setEntries(domainMap, lhs, set);
    }
  } else {
    // Default 'gating' to 'indicating' dependency rule.
    ValueSet set = domainMap.top;
    for (Value rhs : getGatingOperands(op)) {
      domainMap.greatestLowerBound(
          set, valueAnalysis.getEntriesAndId(domainMap, rhs));
    }
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
  ValueDomainMap domainMap(
      impliesReadyRelation, ValueAnalysis::kLubIsIntersection,
      valueAnalysis.index->emptySet(), valueAnalysis.index->fullSet(),
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
        if (getIndicatingResults(op).empty()) {
          // We are only interested in ops with indicating results since that's
          // the only way we have to track their execution.
          return;
        }
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
        if (opOperand.getOwner()->getNumResults() == 0) {
          // Ignore ops with no results since we can't track them.
          continue;
        }
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

//===----------------------------------------------------------------------===//
// 'exclusiveSet'
//
// The following implements the 'exclusiveSet' relation. We define it directly
// using the maybeSameAsRelation (tracking possibly aliasing), the
// impliesReadyRelation (tracking causal ordering of values), and the
// runsEarlierRelation (tracking causal ordering of ops, but derived from the
// impliesReadyRelation for convenience).
//
// In essence the algorithm is:
//
//   1. Given "%x = op ..."
//   2. Find all the ops "... = op %y" which must already be finished when op
//      begins (runsEarlier).
//   3. Find all the values %z which may be aliases of %y (maybeSameAs).
//   4. If all uses of %z are by ops in 2 then we know %x is exclusive to %z.
//   5. If impliesReady(%a, %z) we can say the same for %a (impliesReady).
//
// Note the use of two closures: 'upwards' for the earlier ops, and 'downwards'
// for the ready values.
//
// Though the three relations already account for kLoop control flow ops,
// there's a little be of extra structural fiddling we can do to improve
// the completeness of 'exclusiveSet' for SSA values inside loops. In
// particular, when considering aliasing in step 3 we can exploit two facts:
//   1. The SSA value bound by an op can be considered a 'fresh' instance,
//      distinct from previous values bound in earlier loop iterations.
//   2. Any in-flight loop has not yet allowed its loop results to escape
//      into the rest of the graph.
//
// In other words, though our notion of 'program point' is for the most part
// just 'ready AsyncValue' for the core relations, when it comes to step 3
// we also want a notion of 'program point' representing the overall execution
// graph state, since only that lets us distinguish 'previous' from 'next'
// loop iterations.
//
// TODO(#15840): Another approach would be to define 'exclusiveSet' by another
// dataflow analysis, which is how liveness is traditionally done (though
// for the much simpler situation of program point == operation). I think that
// would still need something like getFullyConsumedWhenOpBegins, but perhaps
// using the three primitive relations without closure under loops. I'm leaving
// this for another day.
//
//===----------------------------------------------------------------------===//

Operation *
AsyncDataflowDependencies::getEffectiveOwner(const OpOperand *opOperand) const {
  Operation *op = opOperand->getOwner();

  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
    ArrayRef<OpOperand> inputOpOperands = cfOp.getInputOpOperands();
    if (llvm::find_if(inputOpOperands, [&](const OpOperand &inputOpOperand) {
          return &inputOpOperand == opOperand;
        }) != inputOpOperands.end()) {
      // Inputs to control flow ops are just plumbing.
      return nullptr;
    }
  }

  if (isa<MEF::OutputOp>(op)) {
    auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op->getParentOp());
    if (!cfOp || cfOp.getFlavor() != GRT::kLoopLike) {
      // Operands to mef.output outside of loops are just plumbing.
      return nullptr;
    }

    // Though we don't consider mef.output to be a true 'op' requiring tracking
    // as a possible consumer of operands, we do need to consider the loop op
    // itself as a possible future consumer.
    return cfOp;
  }

  // Normal case.
  return op;
}

bool AsyncDataflowDependencies::allUsesAreByOperationsInSet(
    const ValueSet &values, const OperationSet &ops) const {
  for (Value value : valueAnalysis.index->setToVector(values)) {
    for (OpOperand &opOperand : value.getUses()) {
      Operation *owningOp = getEffectiveOwner(&opOperand);
      if (!owningOp) {
        // Ignore 'plumbing' uses.
        continue;
      }
      if (!isGatingOperand(&opOperand)) {
        // We don't know when the op will be done with the operand so must
        // consider the use still alive.
        LLVM_DEBUG({
          valueAnalysis.index->printV(llvm::dbgs(), value);
          llvm::dbgs() << " not consumed since non-gating arg to ";
          operationAnalysis.index->printV(llvm::dbgs(), owningOp);
          llvm::dbgs() << "\n";
        });
        return false;
      }
      if (!operationAnalysis.index->setContains(ops, owningOp)) {
        // The operand has a relevant use by an op outside the set, so
        // must consider the use still alive.
        LLVM_DEBUG({
          valueAnalysis.index->printV(llvm::dbgs(), value);
          llvm::dbgs() << " not consumed since used by unfinished op ";
          operationAnalysis.index->printV(llvm::dbgs(), owningOp);
          llvm::dbgs() << "\n";
        });
        return false;
      }
    }
  }
  return true;
}

AsyncDataflowDependencies::ValueEquivalence
AsyncDataflowDependencies::getEquivalenceAtOp(Operation *op) const {
  // Collect results of op which are known to be 'fresh', ie cannot alias
  // with any op operand.
  ValueSet freshResultValues = valueAnalysis.index->emptySet();
  if (!isa<GRT::GRTExtendsLifetimeInterface>(op) &&
      !isa<GRT::GRTMayAliasInterface>(op)) {
    for (Value result : op->getResults())
      valueAnalysis.index->addToSet(freshResultValues, result);
  }

  // Collect all arguments to blocks which may be in a loop.
  ValueSet loopArgs = valueAnalysis.index->emptySet();
  for (Value arg : allLoopBoundArgs(op))
    valueAnalysis.index->addToSet(loopArgs, arg);

  // Collect all results of loop control flow ops containing op.
  ValueSet loopResults = valueAnalysis.index->emptySet();
  for (Value res : allLoopResults(op))
    valueAnalysis.index->addToSet(loopResults, res);

  LLVM_DEBUG({
    llvm::dbgs() << "freshResultValues = ";
    valueAnalysis.index->printVs(
        llvm::dbgs(), valueAnalysis.index->setToVector(freshResultValues));
    llvm::dbgs() << "\n";
    llvm::dbgs() << "loopArgs = ";
    valueAnalysis.index->printVs(llvm::dbgs(),
                                 valueAnalysis.index->setToVector(loopArgs));
    llvm::dbgs() << "\n";
    llvm::dbgs() << "loopResults = ";
    valueAnalysis.index->printVs(llvm::dbgs(),
                                 valueAnalysis.index->setToVector(loopResults));
    llvm::dbgs() << "\n";
  });

  // Take the symmetric/transitive/reflexive closure of the maybeSameAs
  // relation. However:
  //  - entries (%lhs, ...) can be ignored if %lhs is a loop result, since
  //    the loop is still in flight so alias cannot yet have 'escaped' via
  //    the loop result.
  //  - entries (%lhs, %rhs) can be ignored if %lhs is a loop arg and %rhs
  //    is a fresh result value, since we know %rhs will not be the same
  //    as the previous version from %lhs.
  ValueEquivalence equivalence = valueAnalysis.index->identityEquivalence();
  for (const auto &[lhs, set] : maybeSameAsRelation) {
    if (valueAnalysis.index->setContains(loopResults, lhs))
      continue;
    for (auto rhs : valueAnalysis.index->setToVector(set)) {
      if (valueAnalysis.index->setContains(loopArgs, lhs) &&
          valueAnalysis.index->setContains(freshResultValues, rhs))
        continue;
      valueAnalysis.index->setEquivalent(equivalence, lhs, rhs);
    }
  }

  return equivalence;
}

AsyncDataflowDependencies::ValueSet
AsyncDataflowDependencies::getFullyConsumedWhenOpBegins(Operation *op) const {
  LLVM_DEBUG({
    llvm::dbgs() << "getFullyConsumedWhenOpBegins(";
    operationAnalysis.index->printV(llvm::dbgs(), op);
    llvm::dbgs() << ")\n";
  });

  // Which ops have definitely finished?
  OperationSet earlierOps =
      operationAnalysis.getEntries(runsEarlierRelation, op);

  LLVM_DEBUG({
    llvm::dbgs() << "earlierOps = ";
    operationAnalysis.index->printVs(
        llvm::dbgs(), operationAnalysis.index->setToVector(earlierOps));
    llvm::dbgs() << "\n";
  });

  // Collect the candidate SSA values by looking at all the operands of
  // operations in earlierOps.
  ValueSet candidates = valueAnalysis.index->emptySet();
  for (Operation *earlierOp :
       operationAnalysis.index->setToVector(earlierOps)) {
    for (Value value : earlierOp->getOperands())
      valueAnalysis.index->addToSet(candidates, value);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "candidates = ";
    valueAnalysis.index->printVs(llvm::dbgs(),
                                 valueAnalysis.index->setToVector(candidates));
    llvm::dbgs() << "\n";
  });

  // Though we know the candidate SSA values appear to have been consumed, it
  // is actually the underlying AsyncValues we must show to have no more
  // references. For that we must expand each candidate into its equivalence
  // class w.r.t. the maybeSameAs relation.
  //
  // However, if op is within a kLoop sub-block it is sound to consider
  // everything before this op to be from a 'previous' iteration of the loop,
  // and everything after to be for the 'next' iteration. This means the
  // equivalence can ignore some entries in the maybeSameAs relation which are
  // due to tying the loop knot.
  ValueEquivalence equivalence = getEquivalenceAtOp(op);

  LLVM_DEBUG({
    valueAnalysis.printEquivalence(llvm::dbgs(), "equivalence", equivalence);
  });

  // Now, only if ALL the equivalent SSA values have been consumed can we
  // consider all the equivalent SSA values dead. Note we must do this as a
  // unit -- it is not sound to just expand candidates by the above equivalence
  // and filter each one.
  ValueSet result = valueAnalysis.index->emptySet();
  for (Value value : valueAnalysis.index->setToVector(candidates)) {
    ValueSet maybeSameValues =
        valueAnalysis.index->getEquivalenceClass(equivalence, value);
    LLVM_DEBUG({
      llvm::dbgs() << "considering candidate ";
      valueAnalysis.index->printV(llvm::dbgs(), value);
      llvm::dbgs() << " which is equivalent to ";
      valueAnalysis.index->printVs(
          llvm::dbgs(), valueAnalysis.index->setToVector(maybeSameValues));
      llvm::dbgs() << "\n";
    });
    if (allUsesAreByOperationsInSet(maybeSameValues, earlierOps)) {
      LLVM_DEBUG({
        llvm::dbgs() << "include because consumed ";
        valueAnalysis.index->printVs(
            llvm::dbgs(), valueAnalysis.index->setToVector(maybeSameValues));
        llvm::dbgs() << "\n";
      });
      result |= maybeSameValues;
    }
  }

  return result;
}

void AsyncDataflowDependencies::setupExclusive() {
  LLVM_DEBUG(llvm::dbgs() << "setupExclusive\n");
  for (Operation *op : operationAnalysis.index->getAllVs()) {
    if (op == operationAnalysis.index->distinguished)
      continue;

    if (op->getNumResults() == 0) {
      // Don't waste time calculating the consumedSet if there's no results
      // by which to index it.
      continue;
    }

    // Collect the values which have been created and fully consumed (or were
    // never created in the first place) when op BEGINS execution.
    ValueSet consumedSet = getFullyConsumedWhenOpBegins(op);

    // Any value which implies an indicating result of op is ready (including
    // the indicating result itself) can't overlap with any value in
    // consumedSet.
    for (Value indicatingResult : getIndicatingResults(op)) {
      for (Value alsoReadyIndicatingResult : valueAnalysis.index->setToVector(
               downwardsImpliesReadySet(indicatingResult))) {
        LLVM_DEBUG({
          llvm::dbgs() << "add because consumed ";
          valueAnalysis.index->printV(llvm::dbgs(), alsoReadyIndicatingResult);
          llvm::dbgs() << " |-> ";
          valueAnalysis.index->printVs(
              llvm::dbgs(), valueAnalysis.index->setToVector(consumedSet));
          llvm::dbgs() << "\n";
        });
        valueAnalysis.addAll(exclusiveRelation, alsoReadyIndicatingResult,
                             consumedSet);
      }
    }
  }

  // Since the above works only backwards from values bound by ops, we need to
  // account for the reverse, so just close the relation under symmetry.
  valueAnalysis.unionInPlace(exclusiveRelation,
                             valueAnalysis.inverse(exclusiveRelation));

  // Finally, try to exploit operator exclusivity.
  //
  // Note that, unlike the above per-op processing loop, it is unsound to
  // use downwardsImpliesReadySet here. Ie given:
  //   %x = lhsOp ...
  //   %y = rhsOp ...
  //   lhsOp is structurally disjoint from rhsOp
  //   %z = otherOp ...
  //   impliesReady(%z, %x)
  // It is NOT the case that:
  //   %y and %z are exclusive.
  //
  // Note also that, though the structurallyDisjointRelation tells us op and
  // disjointOp can never execute *simultaneously*, they can still be executed
  // *repeatedly*. Thus, the lifetimes of their results may still overlap. It
  // would be nice to detect that by just checking the isNonSelfOverlapping
  // predicate, but that is defined in terms of the exclusiveRelation we're
  // trying to define here, so we'd need to build this by iteration till
  // fixed point. That seems overkill, so instead we'll just use the
  // isAtMostOnce syntactic check to ensure soundness at the cost of more
  // incompleteness.
  for (Operation *op : operationAnalysis.index->getAllVs()) {
    if (op == operationAnalysis.index->distinguished)
      continue;

    if (!isAtMostOnce(op))
      continue;

    for (Operation *disjointOp : operationAnalysis.index->setToVector(
             operationAnalysis.getEntries(structurallyDisjointRelation, op))) {
      if (!isAtMostOnce(disjointOp))
        continue;
      for (Value opResult : op->getResults()) {
        for (Value disjointOpResult : disjointOp->getResults()) {
          LLVM_DEBUG({
            llvm::dbgs() << "add because disjoint ";
            valueAnalysis.index->printV(llvm::dbgs(), opResult);
            llvm::dbgs() << " |-> ";
            valueAnalysis.index->printV(llvm::dbgs(), disjointOpResult);
            llvm::dbgs() << "\n";
          });
          valueAnalysis.add(exclusiveRelation, opResult, disjointOpResult);
        }
      }
    }
  }
}
