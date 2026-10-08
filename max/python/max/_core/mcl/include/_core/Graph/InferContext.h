//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_GRAPHAPI_PYTHON_INFERCONTEXT_H
#define SDK_GRAPHAPI_PYTHON_INFERCONTEXT_H

#include "_core/Graph/Bindings.h"
#include "mlir/IR/MLIRContext.h"

namespace M::Graph::Python {

/// Decorator to infer the MLIRContext as the currently active Python MLIR
/// context. Returns a function which doesn't accept a context.
template <typename Return, typename Context, typename... Args,
          typename = std::enable_if_t<
              std::is_same_v<std::decay_t<Context>, mlir::MLIRContext *>, void>>
constexpr auto infer_context(Return (*func)(Context, Args...)) {
  return [=](Args... args) {
    return func(current_context(), std::forward<Args>(args)...);
  };
}

} // namespace M::Graph::Python

#endif // SDK_GRAPHAPI_PYTHON_INFERCONTEXT_H
