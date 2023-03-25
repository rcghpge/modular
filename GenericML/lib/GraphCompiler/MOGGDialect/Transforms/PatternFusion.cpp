//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGMatchers.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/IR/IRMapping.h"

using namespace M;
using namespace mlir;

namespace M::MOGG {

#define GEN_PASS_DEF_PATTERNFUSION
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct GatherSumPattern : public RewritePattern {
  GatherSumPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag{}, 2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    OpBuilder b(op);
    b.setInsertionPoint(op);
    Value input, indices, gatherAxis, reduceAxis;
    Operation *gather, *sum;
    // TODO: Simplify this once we have parameterized shape calculations
    auto reduceAxisPattern = m_ScalarConstantInt(&reduceAxis, 1);
    auto sumPattern = m_Op<MO::ReduceAddOp>(
        &sum,
        m_Op<MO::GatherOp>(&gather, m_MOMatrix(&input), m_MOMatrix(&indices),
                           m_ScalarConstantInt(&gatherAxis, 0)),
        reduceAxisPattern);
    auto any = matchers::m_Any();
    auto zero = m_ScalarConstantInt(0);
    auto one = m_ScalarConstantInt(1);
    auto sumShapePattern = m_Op<MO::ShapeOfOp>(sumPattern);
    auto squeezeShapePattern = m_Op<MO::ConcatOp>(
        m_Op<MO::GatherOp>(sumShapePattern, m_Op<MO::RangeOp>(any, any, one),
                           zero),
        m_Op<MO::GatherOp>(sumShapePattern, m_Op<MO::RangeOp>(any, any, one),
                           zero),
        zero);

    if (!m_Op<MO::ReshapeOp>(sumPattern, squeezeShapePattern).match(op))
      return failure();

    auto results =
        b.create<MOGG::ExtOp>(
             op->getLoc(),
             /*outputTypes=*/op->getResultTypes(),
             /*op_name=*/"mop.kernels.gather_sum",
             /*inputOperands=*/
             SmallVector<Value>{input, indices, gatherAxis, reduceAxis})
            .getResults();

    rewriter.replaceOp(op, results);

    return mlir::success();
  }
};

struct PatternFusionPass : public impl::PatternFusionBase<PatternFusionPass> {
  using PatternFusionBase::PatternFusionBase;

  void runOnOperation() override;
};

} // namespace

void PatternFusionPass::runOnOperation() {
  auto *ctx = &getContext();

  RewritePatternSet patterns{ctx};
  patterns.add<GatherSumPattern>(ctx);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

} // namespace M::MOGG
