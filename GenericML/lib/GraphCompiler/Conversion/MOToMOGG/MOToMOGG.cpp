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

  // Replaces almost all rebinds which do not operate with unknown shapes.
  // Returns whether we were successful.
  bool replaceMORebinds() {
    auto res = getOperation().walk([&](MO::RebindOp op) {
      // Don't mess any rebind that deals with `?`. When this pass is run
      // this exists due to PT, we need the rebind to give dimensions a proper
      // name.
      MO::TensorType inTy = op.getInput().getType();
      MO::TensorType outTy = op.getResult().getType();
      if (inTy.hasUnknownShape() || outTy.hasUnknownShape())
        return WalkResult::advance();

      // Preserve layout attributes which are needed for best matmul/conv impl.
      if (outTy.getLayoutAttr() || inTy.getLayoutAttr())
        return WalkResult::advance();

      // Don't mess up the signature of ops which are not lowered to MOGG.
      // This is to ensure things still type-check since we keep the old op IR.
      // The exception are rebinds which we need to propagate information.
      llvm::SmallPtrSet<Operation *, 2> replacementExceptions;
      for (Operation *user : op->getUsers()) {
        if (isa<MO::RebindOp>(user))
          continue;

        auto errOrHasLowering = M::MOToMOGGUtils::hasMOGGLowering(user);
        if (errOrHasLowering.isError()) {
          op->emitError(errOrHasLowering.getError());
          return WalkResult::interrupt();
        }
        if (!errOrHasLowering.get())
          replacementExceptions.insert(user);
      }

      replacementExceptions.insert(op);
      op.getResult().replaceAllUsesExcept(op.getInput(), replacementExceptions);
      return WalkResult::advance();
    });

    return !res.wasInterrupted();
  }

  bool moToMOGGKernels() {
    // Load mojo library analysis
    auto &analysis = getAnalysis<MOGG::LoadMojoAnalysisPass>();
    ErrorOr<MOGG::MojoLibraryAnalysis *> errOr =
        analysis.initIfNeeded(&getContext(), kernelLibraryPaths);
    if (errOr.isError()) {
      emitError(getOperation().getLoc(),
                Twine("[MO_TO_MOGG] Failed to load MOJO analysis: ") +
                    errOr.getError());
      return false;
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

    return !result.wasInterrupted();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns{ctx};

    // Replace almost all rebinds without unknown shapes with their original
    // inputs. This removes most rebinds in the graph and is done to prevent
    // rebinds from blocking fusion.
    bool replacementSuccess = replaceMORebinds();
    if (!replacementSuccess)
      return signalPassFailure();

    // Run MOGG lowering on any valid operation
    bool moggLoweringSuccess = moToMOGGKernels();
    if (!moggLoweringSuccess)
      return signalPassFailure();
  }
};

} // namespace
