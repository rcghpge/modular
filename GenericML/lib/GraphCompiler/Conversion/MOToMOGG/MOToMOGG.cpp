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
#include "GenericML/GraphCompiler/MOGGSupport/MOToMOGGUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace M;
using namespace mlir;

namespace M::MO {

#define GEN_PASS_DEF_MOTOMOGG
#include "GenericML/GraphCompiler/Conversion/MOToMOGG.h.inc"

} // namespace M::MO

namespace {

struct MOOpPattern : public RewritePattern {
  MOOpPattern(MLIRContext *ctx, MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis)
      : RewritePattern(MatchAnyOpTypeTag{}, 1, ctx),
        mojoLibraryAnalysis(mojoLibraryAnalysis) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    OpBuilder insertPoint{op};
    auto maybeOp =
        MOToMOGGUtils::lowerOpToMOGG(op, insertPoint, mojoLibraryAnalysis);
    if (!maybeOp.has_value())
      return failure();

    ErrorOr<Operation *> &results = maybeOp.value();
    if (results.isError()) {
      op->emitOpError("[MO_TO_MOGG] ") << results.getError();
      return failure();
    } else {
      rewriter.replaceOp(op, results.takeValue()->getResults());
      return mlir::success();
    }
  }

private:
  MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis;
};

struct MOToMOGGPass : public M::MO::impl::MOToMOGGBase<MOToMOGGPass> {
  using MOToMOGGBase::MOToMOGGBase;

  void runOnOperation() override {

    auto *ctx = &getContext();
    RewritePatternSet patterns{ctx};
    auto mojoLibraryAnalysis =
        MOGG::MojoLibraryAnalysis::get(ctx, extraLibPaths);
    if (mojoLibraryAnalysis.isError())
      return signalPassFailure();
    patterns.add<MOOpPattern>(ctx, *mojoLibraryAnalysis);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
