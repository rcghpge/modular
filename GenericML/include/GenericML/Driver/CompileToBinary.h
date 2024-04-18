//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_COMPILETOBINARY_H
#define GENERICML_DRIVER_COMPILETOBINARY_H

#include "LLCL/CompilerSupport/Context.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/Config.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace M {

/// A context for the `M_compileToBinary()` driver method.
struct CompileToBinaryCtx {
  CompileToBinaryCtx(M::ContextRef ctx)
      : prevRuntimeProfilingTypeMask(Trace::kFullyEnabled),
        context(std::move(ctx)), runtime(context->get<Runtime>()),
        mlirCtx(mlir::MLIRContext::Threading::DISABLED) {
    assert(runtime && "expected to have a runtime");
    registerContext(mlirCtx, context);
    configureMLIRContext(mlirCtx);

    std::optional<TimeTraceProfiler> &profilerOr = runtime->getProfiler();
    if (profilerOr) {
      // Cache the previous profiling type mask to restore it upon destruction.
      prevRuntimeProfilingTypeMask = profilerOr->runtimeProfilingTypeMask();
      profilerOr->setRuntimeProfilingTypeMask(
          1 << Trace::typeBitshift(Trace::kOther));
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
  LLCL::Runtime *runtime;
  mlir::MLIRContext mlirCtx;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

} // namespace M

#endif // GENERICML_DRIVER_COMPILETOBINARY_H
