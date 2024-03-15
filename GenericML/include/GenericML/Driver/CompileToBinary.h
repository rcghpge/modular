//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_COMPILETOBINARY_H
#define GENERICML_DRIVER_COMPILETOBINARY_H

#include "LLCL/Runtime/Runtime.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace M {

/// A context for the `M_compileToBinary()` driver method.
struct CompileToBinaryCtx {
  /// If one doesn't already exist, creates an owned runtime to use during
  /// compilation.
  /// It will convey global options to the passes that need them.
  CompileToBinaryCtx() : runtime(LLCL::createRuntimeIfNeeded()) {
#ifdef MODULAR_PRODUCTION
    mlirCtx.printOpOnDiagnostic(false);
#endif // MODULAR_PRODUCTION
  }

  /// Here the default reverse-order-of-declaration destruction is important.
  ~CompileToBinaryCtx() = default;

  // NOTE: This runtime is declared first so that it is destroyed last.
  // That is because `runtime` must outlive `mlirCtx` since `ConstantFoldPass`
  // creates new attributes owned by `mlirCtx`.
  ConditionallyOwnedPointer<LLCL::Runtime> runtime;
  mlir::MLIRContext mlirCtx;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

} // namespace M

#endif // GENERICML_DRIVER_COMPILETOBINARY_H
