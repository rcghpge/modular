//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_COMPILETOBINARY_H
#define GENERICML_DRIVER_COMPILETOBINARY_H

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "Support/Config.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/SourceMgr.h"

namespace M {

/// A context for the `M_compileToBinary()` driver method.
struct CompileToBinaryCtx {
  CompileToBinaryCtx(M::ContextRef ctx)
      : prevRuntimeProfilingTypeMask(Trace::kFullyEnabled),
        context(std::move(ctx)), runtime(context->get<Runtime>()),
        mlirCtx(mlir::MLIRContext::Threading::DISABLED),
        diagHandler(sourceMgr, &mlirCtx) {
    assert(runtime && "expected to have a runtime");
    registerContext(mlirCtx, context);
    configureMLIRContext(mlirCtx);

    std::optional<TimeTraceProfiler> &profilerOr = runtime->getProfiler();
    if (profilerOr) {
      // Cache the previous profiling type mask to restore it upon destruction.
      prevRuntimeProfilingTypeMask = profilerOr->runtimeProfilingTypeMask();
      // Disable `kCompiler` profiling events, which bloat the trace to several
      // GB during graph compilation.
      profilerOr->setRuntimeProfilingTypeMask(Trace::disableMaskType(
          prevRuntimeProfilingTypeMask, Trace::kCompiler));
    }
  }

  /// Here the default reverse-order-of-declaration destruction is important.
  ~CompileToBinaryCtx() {
    std::optional<TimeTraceProfiler> &profilerOr = runtime->getProfiler();
    if (profilerOr)
      profilerOr->setRuntimeProfilingTypeMask(prevRuntimeProfilingTypeMask);
  }

  uint64_t prevRuntimeProfilingTypeMask;
  M::ContextRef context;
  AsyncRT::Runtime *runtime;
  mlir::MLIRContext mlirCtx;
  mlir::OwningOpRef<mlir::ModuleOp> module;

  // Install a source manager diagnostic handler to properly display MLIR
  // diagnostics that can originate from downstream compilation errors.
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler diagHandler;
};

} // namespace M

#endif // GENERICML_DRIVER_COMPILETOBINARY_H
