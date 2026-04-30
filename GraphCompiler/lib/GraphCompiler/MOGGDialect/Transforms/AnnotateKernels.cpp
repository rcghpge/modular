//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "mlir/IR/BuiltinOps.h"

// Forward declaration of the annotation function defined in
// MOGGDialect/Analysis/MojoLibraryAnalysis.cpp.
namespace M::MOGG {
mlir::LogicalResult runAnnotatePackage(mlir::ModuleOp moduleOp);
} // namespace M::MOGG

namespace M::MOGG {

#define GEN_PASS_DEF_ANNOTATEKERNELS
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct AnnotateKernelsPass
    : public impl::AnnotateKernelsBase<AnnotateKernelsPass> {
  void runOnOperation() override {
    if (failed(runAnnotatePackage(getOperation())))
      signalPassFailure();
  }
};

} // namespace

} // namespace M::MOGG
