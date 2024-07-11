//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_COMPILERSUPPORT_ASYNCSIDEEFFECTMAP_H
#define LLCL_COMPILERSUPPORT_ASYNCSIDEEFFECTMAP_H

#include "AsyncRT/Runtime/Algorithms.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/DenseMap.h"

namespace M::LLCL {

/// This class provides a mapping of Operation to AnyAsyncValueRef. Using async
/// functions to mutate IR often requires synchronizing state, or waiting for
/// any side effects to resolve - this class provides an easy way to handle
/// this. The semantics of this class are that if an operation is not found in
/// the map, there are no side effects in flight, so the AnyAsyncValueRef
/// returned is simply a ready chain.
class AsyncSideEffectMap {
public:
  explicit AsyncSideEffectMap(Runtime &runtime) : runtime(runtime) {}

  Runtime &getRuntime() { return runtime; }

  /// Map an Operation to a chain. This simply moves the provided AsyncValue
  /// into the map.
  void map(Operation *key, AnyAsyncValueRef chain) {
    chainMap[key] = std::move(chain);
  }

  /// Given an operation and a callable, run the provided callable chained on
  /// any side effects this operation may have pending.
  void
  mapChained(Operation *key,
             llvm::function_ref<AnyAsyncValueRef(AnyAsyncValueRef)> callable) {
    map(key, callable(getChain(key)));
  }

  /// Given an operation, add the provided callable to the `andThen` list of the
  /// operation's AsyncValue. If the operation is not found in the map, then the
  /// map assumes no side-effects are in flight and runs the callable.
  void andThenSync(Operation *key, llvm::function_ref<void(void)> callable);

  /// Given an operation, return the AsyncValue contained in the map that
  /// provides the side effect chain for that operation. If the operation is not
  /// in the map, returns a ready chain.
  AnyAsyncValueRef getChain(Operation *key);

  /// Await any side effects of the provided operation to complete. If the
  /// operation is not in the map, this method simply returns. This forwards to
  /// LLCL::await and therefore donates the current thread to completing work
  /// items. If the chain is resolved to an error, this takes ownership of the
  /// error and returns it.
  ErrorOrSuccess await(Operation *key) {
    auto found = chainMap.find(key);
    if (found != chainMap.end())
      LLCL::await(found->getSecond());

    if (found->getSecond().isError())
      return std::move(found->getSecond().takeDiagnostic().getMessage());

    return success();
  }

  /// Await all side effects for every operation in the map.
  void awaitAll() {
    for (auto &av : chainMap)
      LLCL::await(av.getSecond());
  }

private:
  /// The map needs a runtime reference to be able to create ready chains.
  Runtime &runtime;

  /// Store a map of Operation * -> AsyncValue to keep track of in-flight side
  /// effects.
  DenseMap<Operation *, AnyAsyncValueRef> chainMap;
};
} // namespace M::LLCL

#endif // LLCL_COMPILERSUPPORT_ASYNCSIDEEFFECTMAP_H
