//===- BEFExecutor.cpp ----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements various small utilities shared by subsystems of the BEF
// executor.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFExecutor/BEFInvokable.h"
#include "GraphRT/BEFExecutor/PrimitiveFrame.h"
using namespace M;

//===----------------------------------------------------------------------===//
// Invokable implementation
//===----------------------------------------------------------------------===//

// NOTE: This could be moved out to somewhere non-BEF specific if there are
// other kinds of invokables someday.
void BEFInvokable::vtableAnchor() {}

/// Invoke this using the specified runtime, passing the specified
/// arguments. This fills in one AsyncValue for each result.  The caller is
/// allowed to take from the `arguments` async values.
SmallVector<AnyAsyncValueRef, 1>
BEFInvokable::invoke(MutableArrayRef<AnyAsyncValueRef> arguments,
                     Runtime &runtime) const {
  SmallVector<AnyAsyncValueRef, 1> results;
  results.resize(getNumResults());
  invoke(arguments, results, runtime);
  return results;
}

//===----------------------------------------------------------------------===//
// PrimitiveFrame implementation
//===----------------------------------------------------------------------===//

/// Create an AsyncValue with the specified cause, and our current location.
/// This can be set into a result or used for other purposes.
AnyAsyncValueRef PrimitiveFrame::getError(Error cause) {
  return copyLocation().createErrorValue(std::move(cause));
}

/// Mark that execution of the primitive failed with the specified reason.
/// This fills in all remaining unspecified results with that error value.
void PrimitiveFrame::fail(Error cause) {
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
