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
using namespace mlir::matchers;

namespace M::MOGG {

#define GEN_PASS_DEF_PATTERNFUSION
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct GatherSumPattern : public RewritePattern {
  GatherSumPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag{}, 2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value input, indices, gatherAxis, reduceAxis;
    Operation *gather, *sum, *staticReshape;
    // TODO: Simplify this once we have parameterized shape calculations
    auto reduceAxisPattern = m_ScalarConstantInt(&reduceAxis, 1);
    auto sumPattern = m_Op<MO::ReduceAddOp>(
        &sum,
        m_Op<MO::GatherOp>(&gather, m_MOMatrix(m_Any(&input)),
                           m_MOMatrix(m_Any(&indices)),
                           m_ScalarConstantInt(&gatherAxis, 0)),
        reduceAxisPattern);

    auto any = m_Any();
    auto zero = m_ScalarConstantInt(0);
    auto one = m_ScalarConstantInt(1);
    auto sumShapePattern = m_Op<MO::ShapeOfOp>(sumPattern);
    auto squeezeShapePattern = m_Op<MO::ConcatOp>(
        m_Op<MO::GatherOp>(sumShapePattern, m_Op<MO::RangeOp>(any, any, one),
                           zero),
        m_Op<MO::GatherOp>(sumShapePattern, m_Op<MO::RangeOp>(any, any, one),
                           zero),
        zero);

    if (m_Op<MO::Static::ReshapeOp>(&staticReshape, sumPattern).match(op)) {
      MO::ShapeAttr outShape =
          cast<MO::TensorType>(staticReshape->getResultTypes()[0])
              .getTypedShapeAttr();
      MO::ShapeAttr inShape =
          cast<MO::TensorType>(staticReshape->getOperand(0).getType())
              .getTypedShapeAttr();
      ArrayRef<TypedAttr> inValues = inShape.getValues();
      ArrayRef<TypedAttr> outValues = outShape.getValues();

      // This kernel is assuming the reshape is a squeeze on axis 1, so we
      // check that the input and output shapes match that assumption
      if (inShape.getRank() != 3 || inShape.getIntDim(1) != 1 ||
          outValues[0] != inValues[0] || outValues[1] != inValues[2])
        return rewriter.notifyMatchFailure(op, "failed to match squeeze shape");
    } else if (!m_Op<MO::ReshapeOp>(sumPattern, squeezeShapePattern)
                    .match(op)) {
      return rewriter.notifyMatchFailure(op, "failed to match pattern");
    }

    if (!input || !indices || !gatherAxis || !reduceAxis)
      return rewriter.notifyMatchFailure(op, "null input value");

    rewriter.replaceOpWithNewOp<MOGG::ExtOp>(
        op,
        /*outputTypes=*/op->getResultTypes(),
        /*op_name=*/"mop.kernels.gather_sum",
        /*inputOperands=*/
        SmallVector<Value>{input, indices, gatherAxis, reduceAxis});

    return mlir::success();
  }
};

struct MatmulConstBPattern : public RewritePattern {
  MatmulConstBPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag{}, 2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value input, weight;
    if (!m_Op<MO::MatmulOp>(m_Any(&input), m_Any(&weight)).match(op))
      return failure();

    if (!input || !isa_and_nonnull<MO::ConstantOp>(weight.getDefiningOp()))
      return failure();

    MO::TensorType weightsType = cast<MO::TensorType>(weight.getType());
    SmallVector<uint64_t> weightsDims;
    for (int64_t dim : weightsType.getIntDims()) {
      // NOTE this (constants with unknown shape) may occur as a result of
      // constant folding. Once we have shape inference on by default, we won't
      // have this anymore.
      if (dim < 0)
        return failure();
      weightsDims.push_back(dim);
    }

    if (!weightsType.hasStaticShape() || weightsType.hasUnknownShape())
      return failure();

    //   res = matmul(any, const_weight) : [_, _]
    // ===>
    //   packed_weight = pack(const_weight) : [?, ?]
    //   res = matmul_packb(any, packed_weight) : [_, _]
    // where `_` could be a parameter, integer, or `?`
    MO::TensorType packedWeightsType = weightsType;
    if (weightsType.hasRank()) {
      SmallVector<int64_t> unknownDims(weightsType.getRank(),
                                       mlir::ShapedType::kDynamic);
      Type dtype = weightsType.getElementType();
      packedWeightsType = weightsType.cloneWith(unknownDims, dtype);
    }
    Value packedWeights =
        rewriter
            .create<MOGG::ExtOp>(op->getLoc(),
                                 /*outputTypes=*/packedWeightsType,
                                 /*op_name=*/"mop.kernels.pack_matmul_b",
                                 /*inputOperands=*/SmallVector<Value>{weight})
            .getResult(0);

    rewriter.replaceOpWithNewOp<MOGG::ExtOp>(
        op, op->getResultTypes(), "mop.kernels.matmul_packb",
        SmallVector<Value>{input, packedWeights}, weightsDims);

    return mlir::success();
  }
};

struct MatmulTranspBPattern : public RewritePattern {
  MatmulTranspBPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag{}, 2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    OpBuilder b(op);
    b.setInsertionPoint(op);
    Value input, weight, axes;
    if (!m_Op<MO::MatmulOp>(
             m_Any(&input),
             m_Op<MO::TransposeOp>(m_Any(&weight), m_MOVector(m_Any(&axes))))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !weight || !axes)
      return rewriter.notifyMatchFailure(op, "null input value");

    std::optional<SmallVector<int64_t>> maybeAxes =
        MO::tryGetConstantIntVector(cast<MO::TensorValue>(axes));

    if (!maybeAxes.has_value())
      return rewriter.notifyMatchFailure(op, "axes not constant");

    SmallVector<int64_t> &axesValues = maybeAxes.value();

    if (axesValues[0] != 1 || axesValues[1] != 0)
      return rewriter.notifyMatchFailure(op, "axes not transposing");

    auto results =
        b.create<MOGG::ExtOp>(op->getLoc(),
                              /*outputTypes=*/op->getResultTypes(),
                              /*op_name=*/"mop.kernels.matmul_transpb",
                              /*inputOperands=*/
                              SmallVector<Value>{input, weight})
            .getResults();

    rewriter.replaceOp(op, results);

    return mlir::success();
  }
};

struct BatchMatmulAdjBPattern : public RewritePattern {
  BatchMatmulAdjBPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag{}, 2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    OpBuilder b(op);
    b.setInsertionPoint(op);
    Value input, weight, axes;
    if (!m_Op<MO::BatchMatmulOp>(
             m_Any(&input),
             m_Op<MO::TransposeOp>(m_Any(&weight), m_MOVector(m_Any(&axes))))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !weight || !axes)
      return rewriter.notifyMatchFailure(op, "null input value");

    std::optional<SmallVector<int64_t>> maybeAxes =
        MO::tryGetConstantIntVector(cast<MO::TensorValue>(axes));

    if (!maybeAxes.has_value())
      return rewriter.notifyMatchFailure(op, "axes not constant");

    SmallVector<int64_t> &axesValues = maybeAxes.value();

    int64_t rank = axesValues.size();

    if (axesValues[rank - 1] != rank - 2 || axesValues[rank - 2] != rank - 1)
      return rewriter.notifyMatchFailure(op, "axes not adjoint");

    for (int64_t i = 0, e = rank - 2; i < e; ++i)
      if (axesValues[i] != i)
        return rewriter.notifyMatchFailure(op, "axes not adjoint");

    auto results =
        b.create<MOGG::ExtOp>(op->getLoc(),
                              /*outputTypes=*/op->getResultTypes(),
                              /*op_name=*/"mop.kernels.batch_matmul_adjb",
                              /*inputOperands=*/
                              SmallVector<Value>{input, weight})
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
  patterns.add<BatchMatmulAdjBPattern>(ctx);
  patterns.add<MatmulConstBPattern>(ctx);
  patterns.add<MatmulTranspBPattern>(ctx);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

} // namespace M::MOGG
