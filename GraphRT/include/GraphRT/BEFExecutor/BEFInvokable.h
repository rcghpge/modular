//===- GraphRT/BEFExecutor/BEFInvokable.h ---------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_BEFEXECUTOR_INVOKABLE_H
#define GRAPHRT_BEFEXECUTOR_INVOKABLE_H

#include "LLCL/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"

#include <functional>

namespace M {
class ErrorOrSuccess;
class PrimitiveDescriptor;

/// This exposes an abstract interface that describes an "invokable", which is a
/// unit of computation with inputs and results.  It may be "invoked" which
/// triggers its bodies to start processing available dataflow values
/// asynchronously.
///
/// Invokables may be shared across many different runtimes.
///
class BEFInvokable {
public:
  virtual ~BEFInvokable() {}

  size_t getNumArgs() const { return numArgs; }
  size_t getNumResults() const { return numResults; }

  /// Reflect over the body of the invokable, calling the visitor callback for
  /// each invokable. The visitor callback is provided with a StringRef that
  /// contains the primitive's name, and the PrimitiveDescriptor. This uses a
  /// C++-style function pointer so that the user can have some state passed
  /// into the function to, for example, collect all the primitives of a certain
  /// kind. This is not recursive, it only walks the primitives contained
  /// directly within this invokable. It can fail if the visitor callback fails,
  /// but it can also fail due to parsing errors so it's important to always
  /// check the return value.
  virtual ErrorOrSuccess walkPrimitives(
      std::function<ErrorOrSuccess(StringRef, const PrimitiveDescriptor &)>
          visitor) const = 0;

  /// Invoke this using the specified runtime, passing the specified
  /// arguments. This fills in one AsyncValue for each result.  The caller
  /// is allowed to take from the `arguments` async values.
  virtual void invoke(MutableArrayRef<AnyAsyncValueRef> arguments,
                      MutableArrayRef<AnyAsyncValueRef> results,
                      Runtime &runtime) const = 0;

  /// Invoke this using the specified runtime, passing the specified
  /// arguments. This fills in one AsyncValue for each result.  The caller is
  /// allowed to take from the `arguments` async values.
  SmallVector<AnyAsyncValueRef, 1>
  invoke(MutableArrayRef<AnyAsyncValueRef> arguments, Runtime &runtime) const;

  /// Add a new reference to this object.  The lifetime of invokables are
  /// generally tied to the file they are loaded from (e.g. a BEFFile) so they
  /// don't need explicit ownership management in most cases, but some cases
  /// that can extend their lifetime (e.g. implementations of async
  /// control flow operations) should extend their lifetime.
  virtual void addRef() const = 0;

  /// Drop a reference to this object, potentially deallocating it.
  virtual void dropRef() const = 0;

protected:
  BEFInvokable(uint16_t numArgs, uint16_t numResults)
      : numArgs(numArgs), numResults(numResults) {}

private:
  const uint16_t numArgs, numResults;

  BEFInvokable(BEFInvokable &) = delete;
  BEFInvokable &operator=(BEFInvokable &) = delete;
  virtual void vtableAnchor();
};

} // namespace M

#endif // GRAPHRT_BEFEXECUTOR_INVOKABLE_H