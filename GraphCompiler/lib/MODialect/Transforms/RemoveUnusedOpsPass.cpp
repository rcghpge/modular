//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace M;
using namespace MO;

namespace M::MO {

#define GEN_PASS_DEF_REMOVEUNUSEDOPS
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

namespace {

static bool isUnusedOp(Operation *op) {
  // Only MO ops without regions and without users.
  // Skip InvokeShapeFunc and Terminator ops.
  return isa<MO::MODialect>(op->getDialect()) && op->getNumRegions() == 0 &&
         !op->hasTrait<mlir::OpTrait::IsTerminator>() &&
         !isa<MO::InvokeShapeFuncOp>(op) && op->use_empty();
}

static bool isSafeToErase(Operation *op) {
  return isa<MO::AllReduceSumOp, MO::DistributedBroadcastOp,
             MO::DistributedScatterOp>(op);
}

struct EraseDeadOpPattern : RewritePattern {
  EraseDeadOpPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isUnusedOp(op))
      return failure();

    // TODO(GEX-3289): Even if the op is unused, it might still have some side
    // effects expected by the user. In theory the output chain should be
    // propagated correctly, and no uses should mean dead. But currently we
    // never pass chains as outputs to the graph, because the Graph / Executor
    // Python API doesn't support it. So we can't know if an unused op is safe
    // to delete or not. A few ops like AllReduce are exceptions (eg it has side
    // effects like mutating the signal buffers, but if the output tensor is
    // unused it's still safe to remove). Once this is fixed, remove this
    // condition.
    if (!isSafeToErase(op))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

struct RemoveUnusedOpsPass
    : public M::MO::impl::RemoveUnusedOpsBase<RemoveUnusedOpsPass> {
  using RemoveUnusedOpsBase::RemoveUnusedOpsBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<EraseDeadOpPattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
