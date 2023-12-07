//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements various small utilities shared by subsystems of the MEF
// executor.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/MEFExecutor/MEFInvokable.h"
#include "GraphRT/MEFExecutor/PrimitiveDescriptor.h"
#include "LLCL/Runtime/Algorithms.h"
using namespace M;

//===----------------------------------------------------------------------===//
// Invokable implementation
//===----------------------------------------------------------------------===//

// NOTE: This could be moved out to somewhere non-MEF specific if there are
// other kinds of invokables someday.
void MEFInvokable::vtableAnchor() {}

/// Invoke this using the specified runtime, passing the specified
/// arguments. This fills in one AsyncValue for each result.  The caller is
/// allowed to take from the `arguments` async values.
SmallVector<AnyAsyncValueRef, 1> MEFInvokable::createResultValuesAndInvoke(
    MutableArrayRef<AnyAsyncValueRef> arguments, Runtime &runtime) const {
  SmallVector<AnyAsyncValueRef, 1> results;
  results.resize(getNumResults());

  // Final entry point before we fall off to model code.
  // We want to make sure only one of our worker threads run this.
  // NOTE: The invoke needs to be synchronous here so that we ensure the
  // `AsyncValues` for results are created before we return to await(results).
  // The result values are created in the execute() call after the first set of
  // primitives has been scheduled.
  AnyAsyncValueRef finished = AnyAsyncValueRef::allocate<Chain>(runtime);
  addTask(runtime, [&, finished = finished.copy()]() mutable {
    invoke(arguments, results, runtime);
    std::move(finished).emplace<Chain>();
  });

  await(finished);
  return results;
}

//===----------------------------------------------------------------------===//
// RunnablePrimitiveDescriptor implementation
//===----------------------------------------------------------------------===//

/// Create an AsyncValue with the specified cause, and our current location.
/// This can be set into a result or used for other purposes.
AnyAsyncValueRef RunnablePrimitiveDescriptor::getError(Error cause) {
  return AnyAsyncValueRef::createError(
      runtime, EncodedDiagnostic(std::move(cause), copyLocation()));
}

/// Mark that execution of the primitive failed with the specified reason.
/// This fills in all remaining unspecified results with that error value.
void RunnablePrimitiveDescriptor::fail(Error cause) {
  AnyAsyncValueRef *errorOperand = nullptr;
  for (auto &result : getResults()) {
    // Ignore results already specified.
    if (result)
      continue;

    // If this is our first error operand, allocate the error.
    if (!errorOperand) {
      result = getError(std::move(cause));
      errorOperand = &result;
      continue;
    }

    // Otherwise, reuse the previous error.
    result = errorOperand->copy();
  }
}
