//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Converts MO ops to MGP and BEF ops.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Conversion/MOToMOGG.h"
#include "GenericML/MODialect/MOOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace M;

namespace M::MO {

#define GEN_PASS_DEF_MOTOMOGG
#include "GenericML/Conversion/MOToMOGG.h.inc"

} // namespace M::MO

struct MOOpPattern : public RewritePattern {
  MOOpPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // skip non-mo ops and container/constant/output mo ops
    if (!isa<MO::MODialect>(op->getDialect()) || isa<MO::GraphOp>(op) ||
        isa<MO::ConstantOp>(op) || isa<MO::OutputOp>(op))
      return failure();

    // TODO: The current lowering assumes:
    // mogg.ext.op + hard coded kgen shims and does not use the ext_op body
    // This assumption clearly needs to be relaxed into a search going forward
    // TODO: Decide if we need to bufferize and hold an explicit MGP graph in
    // the body
    OpBuilder b(op);

    // TODO: Relax this, lower to kgen generators, do a search here, etc
    std::string opName = "mop.kernels." + op->getName().stripDialect().str();

    // Create the mogg_ext_op
    b.setInsertionPoint(op);
    auto mogg_op = b.create<MOGG::ExtOp>(op->getLoc(),
                                         /*outputTypes=*/op->getResultTypes(),
                                         /*op_name=*/opName,
                                         /*inputOperands=*/op->getOperands());

    // Set up the block in the body region
    // TODO: lower Bufferized version directly in here?
    mogg_op.getBody().emplaceBlock();
    b.setInsertionPointToEnd(&mogg_op.getBody().front());
    b.create<MOGG::OutputOp>(mogg_op->getLoc(), mlir::ValueRange({}));

    rewriter.replaceOp(op, mogg_op.getResults());
    return success();
  }
};

namespace {
struct MOToMOGGPass : public M::MO::impl::MOToMOGGBase<MOToMOGGPass> {
  using MOToMOGGBase::MOToMOGGBase;

  void runOnOperation() override;
};
} // namespace

void MOToMOGGPass::runOnOperation() {
  auto *ctx = &getContext();

  RewritePatternSet patterns{ctx};
  patterns.add<MOOpPattern>(ctx);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}
