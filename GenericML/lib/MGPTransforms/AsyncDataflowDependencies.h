//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Helper for tracking the sameAs equivalence relation and the impliesReady
// partial order over Values which, at runtime, are represented by AsyncValues.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPTRANSFORMS_ASYNCVALUEDEPENDENCIES_H
#define GENERICML_MGPTRANSFORMS_ASYNCVALUEDEPENDENCIES_H

#include "GenericML/MGPDialect/MGPOps.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"

#include "./IndexSet.h"
#include "./RelationDataflowAnalysis.h"

namespace M::MGP {

//===----------------------------------------------------------------------===//
// AsyncDataflowDependencies
//===----------------------------------------------------------------------===//

/// Implements the sameAs and impliesReady relations, and the isAlwaysReady
/// predicate. These can be used to simplify grt.chain.create operations, and
/// decide when buffer lifetimes do not intersect.
///
/// Implemented using forwards, dense dataflow analysis on relations of
/// values.
///
/// (In the following we'll write valueOf(%x) to mean the AsyncValue pointer
/// stored in %x at runtime.)
class AsyncDataflowDependencies : public ForwardsRelationDataflowAnalysis {
public:
  /// Performs the dataflow analysis.
  AsyncDataflowDependencies(IndexSet<Value> *index);

  /// The sameAs(%lhs, %rhs) equivalence relation is true only if at runtime
  ///   valueOf(%lhs) == valueOf(%rhs)
  /// (ie pointer equality) under all execution paths. The relation need not be
  /// complete.
  bool sameAs(Value lhs, Value rhs) const;

  /// Returns { %rhs | sameAs(%lhs, %rhs) }
  Set sameAsEquivalenceClass(Value lhs) const;

  /// The maybeSameAs(%lhs, %rhs) equivalence relation is *false* only if ot
  /// runtime
  ///   valueOf(%lhs) != valueOf(%rhs)
  /// (ie pointer inequality) under all execution paths. The relation need not
  /// be complete.
  bool maybeSameAs(Value lhs, Value rhs) const;

  /// The isAlwaysReady(%value) predicate is true only if at runtime %x is bound
  /// to a chain with no control dependencies, such as the result of
  /// grt.chain.create (). The predicate may not be complete.
  ///
  /// Note we do NOT consider the 'execute' block's first argument as always
  /// ready.
  bool isAlwaysReady(Value value) const;

  /// The impliesReady(%lhs, %rhs) partial order is true only if at runtime
  ///   valueOf(%lhs)->isReady() => valueOf(%rhs)->isReady()
  /// under all execution paths. The relation need not be complete.
  bool impliesReady(Value lhs, Value rhs) const;

  /// Returns the set
  ///   { %lhs | impliesReady(%lhs, %rhs) }
  /// up to the sameAs equivalence. We use 'downwards' because values in the
  /// result set will be later in a dataflow ordering of the model.
  Set downwardsImpliesReadySet(Value rhs) const;

  /// Returns the set
  ///   { %rhs | impliesReady(%lhs, %rhs) }
  /// up to the sameAs equivalence. We use 'upwards' because values in the
  /// result set will be earlier in a dateflow ordering of the model.
  Set upwardsImpliesReadySet(Value lhs) const;

  /// Assuming
  ///   ... = some.op(values)
  /// (for strict some.op), removes as many values from 'values' without
  /// changing when some.op would be ready to execute under all execution paths.
  /// Prefers to remove later values above earlier values in the list. May
  /// remove all values.
  ///
  /// The rules are:
  ///  - Remove %x from values if isAlwaysReady(%x).
  ///  - Remove %x from values if %y in values and impliesReady(%y, %x).
  ///
  /// This can be used to simply the arguments to grt.chain.create ops. Since
  /// impliesReady includes the identity relation this will also remove outright
  /// duplicates.
  ///
  /// CAUTION: O(values.size()^2) implementation.
  void simplify(SmallVectorImpl<Value> &values) const;

  /// Returns the set:
  ///   { %indicator | valueOf(%indicator)->isReady() => %value is dead }
  /// under all execution paths, up to the sameAs equivalence.
  Set consumedWhenReady(Value value) const;

  /// Returns the set:
  ///   { %value | valueOf(%indicator)->isReady() => %value is dead }
  /// under all execution paths, up to the sameAsEquivalence.
  Set indicatesConsumed(Value indicator) const;

  /// Returns true if value is bound at most once (because its defining
  /// op can only be executed at most once), or value is known to be dead
  /// by the time its defining op is (re-)executed (due to some enclosing
  /// kLoop control flow op).
  ///
  /// TODO: Though sound, currently very incomplete, and is to be replaced
  /// by the actual implementation.
  bool isNonSelfOverlapping(Value value) const;

  /// Returns the set of values who's underlying AsyncValue lifetimes can never
  /// overlap with that of indicator, under all execution paths. That is, a
  /// value is in the result only if it can never point to some AsyncValue
  /// at the same time as indicator points to some AsyncValue.
  ///
  /// Note we are using 'point to AsyncValues' here, rather than 'point to
  /// ready AsyncValues'. In particular, for:
  ///    %y = op %x
  /// with no other uses of %x and op strict, the 'point to AsyncValues'
  /// definition would consider %x and %y to be possibly overlapping, but the
  /// 'point to ready AsyncValues' would consider them exclusive. We prefer
  /// the former so that %y can be created while %x is still being processed.
  ///
  /// TODO: Though sound, currently just returns the empty set so is as
  /// incomplete as you can get, but is to be replaced by the actual
  /// implementation.
  Set exclusiveSet(Value value) const;

  /// Prints a representation of all equivalences and relations to os.
  /// For testing only.
  void print(llvm::raw_ostream &os);

private:
  /// Grows domainMap to capture the sameAs relation for operation. However, the
  /// relation will not yet be closed under identity/symmetry/transitivity
  /// since that's very inefficient to compute and represent with the Relation
  /// representation. Instead, setupSameAs will add entries found here to the
  /// Equivalence, thus getting identity/symmetry/transitivity very easily.
  /// Sets changed to true if domainMap was changed.
  void flowPreSameAs(Operation *op, DomainMap &domainMap);

  /// Sets the sameAsEquivalence for the root 'init' and 'execute' blocks.
  void setupSameAs(ModelOp modelOp);

  /// Grows domainMap to capture the maybeSameAs relation for operation. As
  /// for flowPreSameAs, the relaiton will not yet be closed under
  /// identity/symmetry/transitivity.
  void flowPreMaybeSameAs(Operation *op, DomainMap &domainMap);

  /// Sets the maybeSameAsEquivalence for the root 'init' and 'execute' blocks.
  void setupMaybeSameAs(ModelOp modelOp);

  /// Grows domainMap to capture the always-ready predicate into domainMap for
  /// operation. Sets changed to true if domainMap was changed.
  void flowAlwaysReady(Operation *op, DomainMap &domainMap);

  /// Sets the alwaysReadyRelation for the root 'init' and 'execute' blocks.
  void setupAlwaysReady(ModelOp modelOp);

  /// Grows domainMap to capture the impliesReady relation for operation. Sets
  /// changed to true if domainMap was changed.
  void flowImpliesReady(Operation *op, DomainMap &domainMap);

  /// Sets the impliesReadyRelation and its inverse for the root 'init' and
  /// 'execute' blocks.
  void setupImpliesReady(ModelOp modelOp);

  /// Returns the set:
  ///   { %x | valueOf(%x)->isReady()) => %value is dead }
  /// under all execution paths, up to the sameAs equivalence. Note that ALL
  /// values in the result set must be ready for value to be dead. If the
  /// result is empty then it is not possible to know when value is dead,
  /// eg for a mef.output operand.
  Set consumedSet(Value value) const;

  /// Returns the set
  ///   { %lhs | forall %rhs in values . impliesReady(%lhs, %rhs) }
  /// The result may be empty. Note that ANY value in the result set may be
  /// ready for ALL values in the argument set to be ready. If values is
  /// empty then the result will be empty. Otherwise, if the result is empty
  /// then it is not possible to known when all argument values are dead.
  Set impliesAllReadySet(const Set &values) const;

  /// Sets the consumedWhenReadyRelation and its inverse
  /// indicatesConsumedRelation for all values in the model.
  void setupConsumedWhenReady();

  /// The alwaysReady predicate, encoded as a relation using the distinguished
  /// value.
  Relation alwaysReadyRelation;

  /// The sameAs equivalence.
  Equivalence sameAsEquivalence;
  /// The maybeSameAs equivalence.
  Equivalence maybeSameAsEquivalence;

  /// The impliesReady relation and its inverse. Uses canonical values from the
  /// sameAs equivalence.
  Relation impliesReadyRelation;
  Relation inverseImpliesReadyRelation;

  /// The consumedWhenReady relation and its inverse. Uses canonical values from
  /// the sameAs equivalence. An entry (%value, %indicator) is in
  /// consumedWhenReady only if
  ///   valueOf(%indicator)->isReady() => %value is dead
  /// under all execution paths.
  Relation consumedWhenReadyRelation;
  Relation indicatesConsumedRelation;

  friend class AsyncOperationDependencies;
};

//===----------------------------------------------------------------------===//
// AsyncOperationDependencies
//===----------------------------------------------------------------------===//

/// Implements the runsLater and runsEarlier relation over operations.
class AsyncOperationDependencies
    : public RelationDataflowAnalysisBase<Operation *> {
public:
  AsyncOperationDependencies(IndexSet<Operation *> *indexSet,
                             AsyncDataflowDependencies *valueDependencies);

  /// Returns the set:
  ///   { laterOp | laterOp can start executing only
  ///               after earlierOp has finished under all
  ///               execution paths }
  Set runsLaterOps(Operation *earlierOp) const;

  /// Returns the set:
  ///   { earlierOp | laterOp can start executing only
  ///                 after earlierOp has finished under all
  ///                 execution paths }
  Set runsEarlierOps(Operation *laterOp) const;

  /// Returns true if lhsOp and rhsOp can never over overlap in execution
  /// under all execution paths.
  bool areDisjoint(Operation *lhsOp, Operation *rhsOp) const;

  /// Prints a representation of all equivalences and relations to os.
  /// For testing only.
  void print(llvm::raw_ostream &os);

private:
  /// Returns the set:
  ///   { laterOp | %laterOp must begin execution after earlierOp completes }
  /// under all execution paths.
  Set runsLaterSet(Operation *earlierOp) const;

  /// Fills the runsLaterRelation for all operations in block and it's
  /// sub-blocks.
  void setupRunsLaterForBlock(DomainMap &domainMap, Block *block);

  /// Sets the runsLaterRelation and its inverse for all operations in model.
  void setupRunsLater(ModelOp modelOp);

  AsyncDataflowDependencies *valueDependencies;

  /// Identity equivalence.
  Equivalence equivalence;

  /// The runsLater relation and its inverse.
  Relation runsLaterRelation;
  Relation runsEarlierRelation;
};

} // namespace M::MGP

#endif // GENERICML_MGPTRANSFORMS_ASYNCVALUEDEPENDENCIES_H
