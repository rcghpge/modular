//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/CompilerSupport/AsyncSideEffectMap.h"
#include "mlir/IR/Operation.h"

using namespace M::AsyncRT;

void AsyncSideEffectMap::andThenSync(Operation *key,
                                     llvm::function_ref<void(void)> callable) {
  auto found = chainMap.find(key);
  // If we couldn't find the operation in the map, just run the callable.
  if (found == chainMap.end())
    return callable();

  found->getSecond().andThenSync(callable);
}

AnyAsyncValueRef AsyncSideEffectMap::getChain(mlir::Operation *key) {
  auto found = chainMap.find(key);
  if (found == chainMap.end())
    return AsyncValueRef<Chain>::createReady(runtime);

  return found->getSecond().copy();
}
