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

#include "GraphRT/BEFExecutor/Invokable.h"
#include "GraphRT/BEFExecutor/PrimitiveFrame.h"
using namespace M;

//===----------------------------------------------------------------------===//
// Invokable implementation
//===----------------------------------------------------------------------===//

// NOTE: This could be moved out to somewhere non-BEF specific if there are
// other kinds of invokables someday.
void Invokable::vtableAnchor() {}

//===----------------------------------------------------------------------===//
// PrimitiveFrame implementation
//===----------------------------------------------------------------------===//

/// Create an AsyncValue with the specified cause, and our current location.
/// This can be set into a result or used for other purposes.
RCRef<AsyncValue> PrimitiveFrame::getError(Error cause) {
  return AsyncValue::createError(getRuntime(), std::move(cause));
}

/// Mark that execution of the primitive failed with the specified reason.
/// This fills in all remaining unspecified results with that error value.
void PrimitiveFrame::fail(Error cause) {
  RCRef<AsyncValue> *errorOperand = nullptr;
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
