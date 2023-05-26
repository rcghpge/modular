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

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"

namespace M::MGP {

/// Implements the sameAs and impliesReady relations with memoization for
/// efficiency.
///
/// We'll write valueOf(%x) to mean the AsyncValue pointer stored in %x at
/// runtime.
class AsyncDataflowDependencies {
public:
  AsyncDataflowDependencies() = default;

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

  /// If %x and %y are in values (in that relative order) but
  /// impliesReady(%x, %y) then %y is removed. This can be used to simply
  /// the arguments to grt.chain.merge ops. Since impliesReady includes
  /// the identity relation this will also remove outright duplicates.
  ///
  /// CAUTION: O(values.size()^2) implementation.
  void simplify(SmallVectorImpl<Value> &values) const;

private:
  /// Cached results of sameAs.
  mutable DenseMap<std::pair<Value, Value>, bool> sameAsCache;

  /// Cached results of isAlwaysReady.
  mutable DenseMap<Value, bool> isAlwaysReadyCache;

  /// Cached results of impliesReady. The best hit rate will be when
  /// impliesReady is called with (%lhs, %rhs) progressing in roughly
  /// dataflow order.
  mutable DenseMap<std::pair<Value, Value>, bool> impliesReadyCache;
};

} // namespace M::MGP

#endif // GENERICML_MGPTRANSFORMS_ASYNCVALUEDEPENDENCIES_H
