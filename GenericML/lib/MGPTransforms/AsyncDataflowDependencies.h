//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Helper for relating Values to properties of their underlying AsyncValues
// at runtime, including the impliesReady and impliedDead relations.
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

/// This class helps recover implied control flow and liveness relations from
/// the asynchronous dataflow graph. There are four aspects to account for:
///  1. SSA values represent pointers to (reference counted) AsyncValues. There
///     may thus be many SSA names for the same underlying AsyncValue. When we
///     talk about liveness it is the AsyncValues we care about, not the SSA
///     names.
///  2. 'Primitive' ops (ie not control flow or output ops) BEGIN execution
///     when their 'gating' input AsyncValues are ready, and FINISH execution
///     when their 'indicating' result AsyncValues are ready. (See Utils.h for
///     the definition of 'gating' and 'indicating'.) The causal ordering of
///     ops depends on dataflow between these arguments and results. We try
///     very hard to avoid having to talk about the point at which an op RETURNS
///     (thus binding its result values) which may be before it FINISHES. As a
///     result, we don't need to distinguish between ops which emplace and
///     return their result AsyncValues immediately vs those which launch an
///     asynchronous sub-task.
///  3. Branching control flow ops will only 'splice in' one of their
///     sub-blocks. Thus it is possible for some ops to never be executed.
///     It is not the case that the sub-block ops are finished when the branch
///     block 'returns'.
///  4. Looping control flow ops may obviously cause some ops to be executed
///     more than once, and the results from an earlier loop iteration may find
///     their way into the next iteration. However, unlike branching control
///     flow, loops are assumed to only 'splice in' sub-blocks for the next
///     loop iteration strictly after all results from the previous iteration
///     are ready. Ie only when all the ops in the previous iteration have
///     finished. This greatly simplifies reasoning about overlapping
///     operations.
///
/// We assume the model is 'overhang free': if all the results of the model
/// are ready then no ops can still be executing. Thus any op which launches
/// asynchronous computation must signal completion of that computation via
/// an indicating result, and that result cannot be ignored.
///
/// Ops are allowed to return results which 'lifetime extend' their arguments.
/// This means a result AsyncValue may contain a pointer to an argument
/// AsyncValue. Such ops must implement GRTExtendsLifetimeInterface.
///
/// Similarly, ops are allowed to return results which 'alias' their arguments.
/// This means a result may be a pointer to an argument AsyncValue. Such ops
/// must implement GRTMayAliasInterface. Any possible aliasing or sharing
/// *within* AsyncValues can be considered as aliasing of the AsyncValues
/// themselves and signalled with the same interface. For this analysis,
/// GRTExtendsLifetimeInterface and GRTMayAliasInterface are treated
/// identically.
///
/// To make sense of all this we use:
///  - sameAs equivalence (two values always point to the same AsyncValue)
///  - maybeSameAs equivalence (two values may point to the same AsyncValue)
///  - isAlwaysReady predicate (the value is always a ready AsyncValue)
///  - impliesReady relation (if one value is ready then so is another)
///  - structurallyDisjoint relation (two ops can never overlap due to branches)
///  - runsLater relation (a later op can begin only after an earlier op
///    finishes)
///
/// In the following we'll write valueOf(%x) to mean the AsyncValue pointer
/// stored in %x at runtime.
class AsyncDataflowDependencies {
private:
  using ValueAnalysis = ForwardsRelationDataflowAnalysis;
  using OperationAnalysis = RelationDataflowAnalysisBase<Operation *>;

public:
  using ValueSet = ValueAnalysis::Set;
  using ValueEquivalence = ValueAnalysis::Equivalence;
  using ValueRelation = ValueAnalysis::Relation;
  using OperationSet = OperationAnalysis::Set;
  using OperationRelation = OperationAnalysis::Relation;

  /// Helpers for working with domain maps and relations over both values and
  /// operations.
  ValueAnalysis valueAnalysis;
  OperationAnalysis operationAnalysis;

  /// Performs the dataflow analysis.
  AsyncDataflowDependencies(IndexSet<Value> *valueIndex,
                            IndexSet<Operation *> *operationIndex);

  /// The sameAs(%lhs, %rhs) equivalence is true only if at runtime:
  ///   valueOf(%lhs) == valueOf(%rhs)
  /// under all execution paths. The relation may not be complete. There is no
  /// causal ordering in this relation, and it is possible %lhs and %rhs are
  /// bound at different times.
  bool sameAs(Value lhs, Value rhs) const;

  /// The maybeSameAs(%lhs, %rhs) equivalence is *false* only if ot runtime:
  ///   valueOf(%lhs) != valueOf(%rhs), and
  ///   valueOf(%lhs) does not lifetime extend valueOf(%rhs)
  /// under all execution paths. The relation may not be complete. There is no
  /// causal ordering in this relation, and it is possible %lhs and %rhs are
  /// bound at different times.
  ///
  /// Note that we need the 'lifetime extends' property to support the
  /// 'exclusive' relation below.
  bool maybeSameAs(Value lhs, Value rhs) const;

  /// Returns the set { %rhs | maybeSameAs(%lhs, %rhs) }.
  ValueSet maybeSameAsSet(Value lhs) const;

  /// The isAlwaysReady(%value) predicate is true only if at runtime
  ///   valueOf(%value)->isReady()
  /// under all execution paths. For example, %value may be the result of a
  /// grt.chain.create (). The predicate may not be complete.
  bool isAlwaysReady(Value value) const;

  /// The impliesReady(%lhs, %rhs) partial order is true only if at runtime
  ///   valueOf(%lhs)->isReady() =>
  ///     if %rhs is bound then valueOf(%rhs)->isReady()
  /// under all execution paths. The relation may not be complete, and
  /// impliesReady(%x, %x) may not be true.
  ///
  /// The additional 'is bound' qualification allows us to consider %lhs
  /// to imply %rhs as ready even though the block containing the binding for
  /// %rhs may never be executed (eg because it is within a kBronch control
  /// flow op sub-branch).
  bool impliesReady(Value lhs, Value rhs) const;

  /// Returns the set
  ///   { %lhs | impliesReady(%lhs, %rhs) or %lhs == %rhs }
  ///
  /// Here 'downwards' is in the sense of the usual top-to-bottom execution
  /// order of sequential programs, even though operations may be ordered
  /// arbitrarily within the block of the input model.
  ValueSet downwardsImpliesReadySet(Value rhs) const;

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

  /// The areDisjoint(lhsOp, rhsOp) relation is true only if lhsOp and rhsOp
  /// can never overlap in their execution over all execution paths. Here
  /// 'execution' is from BEGIN to FINISH (as opposed to BEGIN to RETURNED).
  ///
  /// Ops may be disjoint because lhsOp and rhsOp can never both be
  /// incorporated into the execution graph at the same time (eg because they
  /// are in distinct arms of an kBronch), or because the execution of lhsOp and
  /// rhsOp are known to be strictly sequenced by AsyncValue dataflow
  /// dependencies. The relation may not be complete.
  bool areDisjoint(Operation *lhsOp, Operation *rhsOp) const;

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
  /// the former so that if %x is known to be exclusive to %y, %y can be
  /// created by the op itself without risk of conflict.
  ///
  /// TODO: Though sound, currently just returns the empty set so is as
  /// incomplete as you can get, but is to be replaced by the actual
  /// implementation.
  ValueSet exclusiveSet(Value indicator) const;

  /// Returns true if value is bound at most once (because its defining
  /// op can only be executed at most once), or value is known to be dead
  /// by the time its defining op is (re-)executed (due to some enclosing
  /// kLoop control flow op).
  ///
  /// TODO: Though sound, currently very incomplete, and is to be replaced
  /// by the actual implementation.
  bool isNonSelfOverlapping(Value value) const;

  /// Prints a representation of all equivalences and relations to os.
  /// For testing only.
  void print(llvm::raw_ostream &os);

private:
  using ValueDomainMap = ValueAnalysis::DomainMap;

  /// Grows domainMap to capture the sameAs relation for operation. Sets
  /// changed to true if domainMap was changed.
  void flowSameAs(Operation *op, ValueDomainMap &domainMap);

  /// Sets the sameAsRelation for the root 'init' and 'execute' blocks.
  void setupSameAs();

  /// Grows domainMap to capture the maybeSameAs relation for operation. Sets
  ///  changed to true if domainMap was changed.
  void flowMaybeSameAs(Operation *op, ValueDomainMap &domainMap);

  /// Sets the maybeSameAsRelation for the root 'init' and 'execute' blocks.
  void setupMaybeSameAs();

  /// Grows domainMap to capture the always-ready predicate into domainMap for
  /// operation. Sets changed to true if domainMap was changed.
  void flowAlwaysReady(Operation *op, ValueDomainMap &domainMap);

  /// Sets the alwaysReadyRelation for the root 'init' and 'execute' blocks.
  void setupAlwaysReady();

  /// Grows domainMap to capture the impliesReady relation for operation. Sets
  /// changed to true if domainMap was changed.
  void flowImpliesReady(Operation *op, ValueDomainMap &domainMap);

  /// Sets the impliesReadyRelation and its inverse for the root 'init' and
  /// 'execute' blocks.
  void setupImpliesReady();

  /// Grows relation to capture which operations can never co-execute
  /// in block and its sub-blocks.
  void accumStructurallyDisjointForBlock(OperationRelation &relation,
                                         Block *block);

  /// Sets the structurallyDisjoint relation for all operations in the model.
  void setupStructurallyDisjoint();

  /// Returns the set:
  ///   { laterOp | laterOp can BEGIN execution only after earlierOp FINISHES }
  /// under all execution paths.
  OperationSet runsLaterSet(Operation *earlierOp) const;

  /// Sets the runsLaterRelation and its inverse for all operations in model.
  void setupRunsLater();

  /// (%lhs, %rhs) is in sameAsRelation only if %lhs points to the same
  /// AsyncValue as %rhs when it is bound.
  ///
  /// Note that even if (%x, %z) and (%y, %z) are in the relation it may not be
  /// the case that (%x, %y) or (%y, %x) is in the relation. It is necessary to
  /// take the symmetric/transitive/identity closure to discover the equivalence
  /// class {%x, %y, %z} of SSA values all pointing to the same AsyncValue.
  ValueRelation sameAsRelation;
  /// The above closed under symmetry/transitivity/identity.
  ValueEquivalence sameAsEquivalence;

  /// (%lhs, %rhs) is in maybeSameAsRelation if %lhs could point to the same
  /// AsyncValue as %rhs when it is bound, or %lhs could lifetime extend %rhs.
  ValueRelation maybeSameAsRelation;
  /// The above closed under symmetry/transitivity/identity.
  ValueEquivalence maybeSameAsEquivalence;

  /// %x is in alwaysReadyPredicate only if %x points to an AsyncValue which is
  /// known to be always ready at runtime.
  ValueSet alwaysReadyPredicate;

  /// (%lhs, %rhs) is in impliesReady relation only if the AsyncValue pointed
  /// to by %rhs is known to be ready or never created if the AsyncValue pointed
  /// to by %lhs is ready at runtime.
  ValueRelation impliesReadyRelation;
  /// Inverse of above.
  ValueRelation inverseImpliesReadyRelation;

  /// (lhsOp, rhsOp) is in structurallyDisjointRelation if lhsOp and rhsOp
  /// con never overlap in execution because they are distinct branches of
  /// a kBronch control flow op. Unlike the impliesReady relation, this is a
  /// purely structural property that does not depend on the dataflow of
  /// AsyncValues.
  OperationRelation structurallyDisjointRelation;

  /// (lhsOp, rhsOp) is in runsLaterRelation only if rhsOp can BEGIN execution
  /// only after either lhsOp has FINISHED execution, or rhsOp is never
  /// executed, under all execution paths.
  OperationRelation runsLaterRelation;
  /// Inverse of above.
  OperationRelation runsEarlierRelation;
};

} // namespace M::MGP

#endif // GENERICML_MGPTRANSFORMS_ASYNCVALUEDEPENDENCIES_H
