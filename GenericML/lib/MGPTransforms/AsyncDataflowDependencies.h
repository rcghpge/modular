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

  /// Grows domainMap to capture the always-ready predicate into domainMap for
  /// operation. Sets changed to true if domainMap was changed.
  void flowAlwaysReady(Operation *op, DomainMap &domainMap);

  /// Sets the alwaysReadyRelation for the root 'init' and 'execute' blocks.
  void setupAlwaysReady(ModelOp modelOp);

  /// Returns true if lhs implies rhs directly.
  static bool primitiveImpliesReady(Value lhs, Value rhs);

  /// Grows domainMap to capture the impliesReady relation for operation. Sets
  /// changed to true if domainMap was changed.
  void flowImpliesReady(Operation *op, DomainMap &domainMap);

  /// Sets the impliesReadyRelation for the root 'init' and 'execute' blocks.
  void setupImpliesReady(ModelOp modelOp);

  /// The alwaysReady predicate, encoded as a relation using the distinguished
  /// value.
  Relation alwaysReadyRelation;

  /// The impliesReady relation. Uses canonical values from the sameAs
  /// equivalence.
  Relation impliesReadyRelation;
};

} // namespace M::MGP

#endif // GENERICML_MGPTRANSFORMS_ASYNCVALUEDEPENDENCIES_H
