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
#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGSupport/LoweringHelper.h"
#include "GenericML/GraphCompiler/MOGGSupport/MOToMOGGUtils.h"
#include "GenericML/GraphCompiler/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

using namespace M;
using namespace mlir;

namespace M::MO {

#define GEN_PASS_DEF_MOTOMOGG
#include "GenericML/GraphCompiler/Conversion/MOToMOGG.h.inc"

} // namespace M::MO

namespace {

struct MOOpPattern : public RewritePattern {
  MOOpPattern(MLIRContext *ctx, MOToMOGGUtils::MoToKernelMapTy &moToMojoKernels,
              MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis)
      : RewritePattern(MatchAnyOpTypeTag{}, 1, ctx),
        moToMojoKernels(moToMojoKernels),
        mojoLibraryAnalysis(mojoLibraryAnalysis) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto maybeResults =
        MOToMOGGUtils::lowerOpToMOGG(op, moToMojoKernels, mojoLibraryAnalysis);
    if (!maybeResults.has_value())
      return failure();

    ErrorOr<ValueRange> &results = maybeResults.value();
    if (results.isError()) {
      op->emitOpError("[MO_TO_MOGG] ") << results.getError();
      return failure();
    } else {
      rewriter.replaceOp(op, results.takeValue());
      return mlir::success();
    }
  }

  MOToMOGGUtils::MoToKernelMapTy &moToMojoKernels;

  MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis;
};

struct MOToMOGGPass : public M::MO::impl::MOToMOGGBase<MOToMOGGPass> {
  using MOToMOGGBase::MOToMOGGBase;

  void runOnOperation() override {

    // A map from MO op type ids to a record describing how to lower them for
    // 'cpu' devices.
    MOToMOGGUtils::MoToKernelMapTy moToMojoKernels =
        MOToMOGGUtils::getHardCodedOpMapping();
    auto *ctx = &getContext();
    RewritePatternSet patterns{ctx};
    auto mojoLibraryAnalysis =
        MOGG::MojoLibraryAnalysis::get(ctx, extraLibPaths);
    if (mojoLibraryAnalysis.isError())
      return signalPassFailure();
    patterns.add<MOOpPattern>(ctx, moToMojoKernels, *mojoLibraryAnalysis);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
