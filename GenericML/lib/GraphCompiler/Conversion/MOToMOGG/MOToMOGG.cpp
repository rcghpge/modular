//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Converts MO ops to MGP and MEF ops.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOToMOGG.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MOToMOGGUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
using namespace M;
using namespace mlir;

namespace M::MO {

#define GEN_PASS_DEF_MOTOMOGG
#include "GenericML/GraphCompiler/Conversion/MOToMOGG.h.inc"

} // namespace M::MO

namespace {

struct MOToMOGGPass : public M::MO::impl::MOToMOGGBase<MOToMOGGPass> {
  using MOToMOGGBase::MOToMOGGBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns{ctx};

    auto &analysis = getAnalysis<MOGG::LoadMojoAnalysisPass>();
    ErrorOr<MOGG::MojoLibraryAnalysis *> errOr =
        analysis.initIfNeeded(ctx, kernelLibraryPaths);
    if (errOr.isError()) {
      emitError(getOperation().getLoc(),
                "[MO_TO_MOGG] Failed to load MOJO analysis.");
      return signalPassFailure();
    }
    MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis = *errOr.takeValue();

    // Run MOGG lowering on any valid operation
    auto result = getOperation().walk([&](Operation *op) {
      OpBuilder insertPoint{op};
      auto maybeOp =
          MOToMOGGUtils::lowerOpToMOGG(op, insertPoint, mojoLibraryAnalysis,
                                       /*forceTrivial=*/false, forceSync);

      // Expected no lowering.
      if (!maybeOp.has_value())
        return WalkResult::advance();

      ErrorOr<Operation *> &results = maybeOp.value();
      if (results.isError()) {
        // Lowering error, abort pass!
        op->emitOpError("[MO_TO_MOGG] ") << results.getError();
        return WalkResult::interrupt();
      } else {
        op->replaceAllUsesWith(results.takeValue()->getResults());
        op->erase();
        return WalkResult::advance();
      }
    });

    if (result.wasInterrupted())
      return signalPassFailure();

    // Run canonicalization to kill dead code
    mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(
        ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(mlir::createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      emitError(getOperation().getLoc(),
                "[MO_TO_MOGG] Failed to run post-pass canonicalization.");
      return signalPassFailure();
    }
  }
};

} // namespace
