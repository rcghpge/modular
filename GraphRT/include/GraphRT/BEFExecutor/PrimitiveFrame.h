//===- GraphRT/BEFExecutor/PrimitiveFrame.h ---------------------*- C++ -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_BEFEXECUTOR_PRIMITIVE_FRAME_H
#define GRAPHRT_BEFEXECUTOR_PRIMITIVE_FRAME_H

#include "LLCL/ForwardDecls.h"
#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Support/RCRef.h"
#include "Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace M {

/// This class captures the states associated with an invocation of a primitive,
/// including the input arguments, attributes, result values, location and
/// runtime.
///
/// The implementation of a primitive is responsible for creating AsyncValue
/// objects and setting them into the frame.  The correct number of results are
/// present, but each AsyncValue pointers is null when the implementation is
/// invoked.
class PrimitiveFrame {
public:
  PrimitiveFrame(Runtime &runtime) : runtime(runtime) {}

  Runtime &getRuntime() const { return runtime; }

  // TODO: Add location information for error reporting.

  //===--------------------------------------------------------------------===//
  // Arguments
  //===--------------------------------------------------------------------===//

  size_t getNumArgs() const { return numArguments; }

  /// Get the argument at the given index as AsyncValue*.
  AsyncValue *getArg(size_t index) const {
    assert(index < getNumArgs() && "invalid argument number");
    return argsResultsAndAttrs[index].value;
  }

  /// Look through the argument AsyncValue to get the element as type T.
  template <typename T>
  T &getArg(size_t index) const {
    return getArg(index)->get<T>();
  }

  ArrayRef<AsyncValue *> getArguments() const {
    AsyncValue *const *firstArg = &argsResultsAndAttrs.data()[0].value;
    return llvm::makeArrayRef(firstArg, getNumArgs());
  }

  //===--------------------------------------------------------------------===//
  // Attributes
  //===--------------------------------------------------------------------===//

  // TODO: Attributes.

  //===--------------------------------------------------------------------===//
  // Results
  //===--------------------------------------------------------------------===//

  size_t getNumResults() const { return numResults; }

  ArrayRef<AsyncValue *> getResults() const {
    return const_cast<PrimitiveFrame *>(this)->getResults();
  }

  MutableArrayRef<AsyncValue *> getResults() {
    AsyncValue **firstResult = &argsResultsAndAttrs.data()[numArguments].value;
    return llvm::makeMutableArrayRef(firstResult, getNumResults());
  }

  /// Set the result at the given index with the given AsyncValue.
  void setResult(size_t index, RCRef<AsyncValue> value) {
    assert(index < numResults && "invalid result number");
    AsyncValue *&result = argsResultsAndAttrs[numArguments + index].value;
    assert(!result && "result is already set");
    result = value.release();
  }

  // TODO: convenience methods for constructing results.

protected:
  Runtime &runtime;

  // numResults is set to ~0 so we can check that addAttribute() is called
  // after setNumResults.
  uint16_t numResults = uint16_t(~0U);
  uint16_t numArguments = 0;

  /// We store arguments, results, and attributes, and results in a single
  /// SmallVector for efficiency.
  union AsyncValueOrAttr {
    AsyncValue *value;
    const void *attr;
  };
  SmallVector<AsyncValueOrAttr, 8> argsResultsAndAttrs;
};

/// This class is used to construct a PrimitiveFrame object without exposing the
/// builder methods to the implementation functions.
///
/// When initializing a KernelFrame, this class requires that the client
/// performs the following actions in order:
///   1. Add argument values.
///   2. Set the number of results with `setNumResults`.
///   3. Add attributes values.
///
class PrimitiveFrameBuilder : public PrimitiveFrame {
public:
  using PrimitiveFrame::PrimitiveFrame;

  /// Clear all fields.
  void reset() {
    argsResultsAndAttrs.clear();
    numArguments = 0;
    numResults = uint16_t(~0U);
  }

  // Add a new argument to the KernelFrame.
  void addArg(AsyncValue *value) {
    assert(numResults == uint16_t(~0U) &&
           "Must call AddArg before calling SetNumResults");
    argsResultsAndAttrs.push_back({value});
    ++numArguments;
  }

  // Set the number of results expected, and allocate space to capture them.
  void setNumResults(size_t n) {
    assert(numArguments == argsResultsAndAttrs.size());
    assert(numResults == uint16_t(~0U));
    numResults = n;
    argsResultsAndAttrs.resize(argsResultsAndAttrs.size() + n);
  }

  AsyncValue *getResult(size_t index) const {
    assert(index < numResults && "invalid result #");
    return argsResultsAndAttrs[numArguments + index].value;
  }
};

} // namespace M

#endif // GRAPHRT_BEFEXECUTOR_PRIMITIVE_FRAME_H