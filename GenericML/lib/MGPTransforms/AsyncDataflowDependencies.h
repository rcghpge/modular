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

#include "./ValueIndex.h"

namespace M::MGP {

/// Implements the sameAs and impliesReady relations, and the isAlwaysReady
/// set. These can be used to simplify grt.chain.merge operations, and decide
/// when buffer lifetimes do not intersect.
///
/// Implemented using forwards, dense dataflow analysis.
///
/// (In the following we'll write valueOf(%x) to mean the AsyncValue pointer
/// stored in %x at runtime.)
class AsyncDataflowDependencies {
public:
  /// Bijection between values and indexes.
  ValueIndex index;

  /// Performs the dataflow analysis on model.
  AsyncDataflowDependencies(ModelOp model);

  /// The sameAs(%lhs, %rhs) equivalence relation is true only if at runtime
  ///   valueOf(%lhs) == valueOf(%rhs)
  /// (ie pointer equality) under all execution paths. The relation need not be
  /// complete.
  bool sameAs(Value lhs, Value rhs) const;

  /// The isAlwaysReady(%value) predicate is true only if at runtime %x is bound
  /// to a chain with no control dependencies, such as the result of
  /// grt.chain.create. The predicate may not be complete.
  ///
  /// Note we do NOT consider the 'execute' block's first argument as always
  /// ready.
  bool isAlwaysReady(Value value) const;

  /// The impliesReady(%lhs, %rhs) partial order is true only if at runtime
  ///   valueOf(%lhs)->isReady() => valueOf(%rhs)->isReady()
  /// under all execution paths. The relation need not be complete.
  bool impliesReady(Value lhs, Value rhs) const;

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
  /// This can be used to simply the arguments to grt.chain.merge ops. Since
  /// impliesReady includes the identity relation this will also remove outright
  /// duplicates.
  ///
  /// CAUTION: O(values.size()^2) implementation.
  void simplify(SmallVectorImpl<Value> &values) const;

private:
  using Set = ValueIndex::Set;
  using Equivalence = ValueIndex::Equivalence;
  using Relation = DenseMap<Value, Set>;

  /// Sets relation s.t. (value, x) in relation iff (x in set or x == value),
  /// up to equivalence. Marks changed true if relation was changed.
  void setEntriesAndId(Relation &relation, Value value, Set set);

  /// Returns the set {x | (value, x) in relation}, up to equivalence.
  Set getEntries(const Relation &relation, Value value) const;

  /// Returns true if (value, distinguished) is in relation, up to equivalence.
  bool containsDistinguished(const Relation &relation, Value value) const;

  /// Returns true if lhs == rhs, or (lhs, rhs) is in relation, up to
  /// equivalence.
  bool containsEntryOrId(const Relation &relation, Value lhs, Value rhs) const;

  /// Returns the intersection of sets.
  Set intersection(const SmallVector<Set> &sets);

  using AccumSubBlock = std::function<void(Relation &, Block *)>;

  /// Grows relation for the sub-blocks of control flow op in 'series', where
  /// the first block takes its inputs from the given inputsOrResults. Sets
  /// changed to true if relation was changed.
  void accumSequential(GRT::GRTControlFlowInterface cfOp, Relation &relation,
                       const AccumSubBlock &accumSubBlock,
                       const ValueRange &inputsOrResults);

  /// Grows relation for the sub-blocks of control flow op in 'parallel'.
  /// Sets changed to true if relation was changed.
  void accumParallel(GRT::GRTControlFlowInterface cfOp, Relation &relation,
                     const AccumSubBlock &accumSubBlock);

  /// Grows relation using the control flow closure rules, using processSubBlock
  /// to grow the relations for each control flow sub-block. Sets changed to
  /// true if relation was changed.
  void accumControlFlow(GRT::GRTControlFlowInterface cfOp, Relation &relation,
                        const AccumSubBlock &accumSubBlock);

  /// Accumulates the sameAs relation into relation for block. However, the
  /// relation will not yet be closed under identity/symmetry/transitivity.
  /// Sets changed to true if relation was changed.
  void accumPreSameAs(Relation &relation, Block *block);

  /// Sets the sameAsEquivalence for the root 'init' and 'execute' blocks.
  void setupSameAs(ModelOp modelOp);

  /// Accumulates the always-ready values into relation for block.
  /// An entry (value, distinguished) in relation indicates value is always
  /// ready. Sets changed to true if relation was changed.
  void accumAlwaysReady(Relation &relation, Block *block);

  // Sets the alwaysReadyRelation for the root 'init' and 'execute' blocks.
  void setupAlwaysReady(ModelOp modelOp);

  /// Returns true if lhs implies rhs directly.
  static bool primitiveImpliesReady(Value lhs, Value rhs);

  /// Accumulates the impliesReady relation into relation for block. Sets
  /// changed to true if relation was changed.
  void accumImpliesReady(Relation &relation, Block *block);

  // Sets the impliesReadyRelation for the root 'init' and 'execute' blocks.
  void setupImpliesReady(ModelOp modelOp);

  // Set to true when setEntriesAndId changes its relation.
  bool changed;

  // The sameAs equivalence relation.
  Equivalence sameAsEquivalence;

  // The alwaysReady set, encoded as a relation:
  //    (value, distinguished) is in the relation iff value is in the set.
  // This way we can share the same relation processing machinery needed by
  // sameAs and impliedReady. Uses canonical values from the sameAs equivalence.
  Relation alwaysReadyRelation;

  // The impliesReady relation. Uses canonical values from the sameAs
  // equivalence.
  Relation impliesReadyRelation;
};

} // namespace M::MGP

#endif // GENERICML_MGPTRANSFORMS_ASYNCVALUEDEPENDENCIES_H
