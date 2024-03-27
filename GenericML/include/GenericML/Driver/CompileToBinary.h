//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_COMPILETOBINARY_H
#define GENERICML_DRIVER_COMPILETOBINARY_H

#include "LLCL/Runtime/Runtime.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace M {

/// A context for the `M_compileToBinary()` driver method.
struct CompileToBinaryCtx {
  CompileToBinaryCtx(M::ContextRef ctx)
      : context(ctx.copy()), runtime(ctx->get<Runtime>()) {
    assert(runtime && "expected to have a runtime");
    registerContext(mlirCtx, context);

#ifdef MODULAR_PRODUCTION
    mlirCtx.printOpOnDiagnostic(false);
#endif // MODULAR_PRODUCTION
  }

  /// Here the default reverse-order-of-declaration destruction is important.
  ~CompileToBinaryCtx() = default;

  M::ContextRef context;
  LLCL::Runtime *runtime;
  mlir::MLIRContext mlirCtx;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

} // namespace M

#endif // GENERICML_DRIVER_COMPILETOBINARY_H
