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
using namespace M;

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
