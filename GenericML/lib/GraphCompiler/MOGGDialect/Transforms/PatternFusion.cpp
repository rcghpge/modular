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

    if (!m_Op<MO::ReduceAddOp>(
             m_Op<MO::GatherOp>(m_MOMatrix(m_Any(&input)),
                                m_MOMatrix(m_Any(&indices)),
                                m_ScalarConstantInt(&gatherAxis, 0)),
             m_ScalarConstantInt(&reduceAxis, 1))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

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

struct FullyConnectedPattern : public RewritePattern {
  FullyConnectedPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag{}, 2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    OpBuilder b(op);
    b.setInsertionPoint(op);

    Value input, weight, bias;
    auto matmulP = m_Op<MO::MatmulOp>(m_MOMatrix(m_Any(&input)),
                                      m_MOMatrix(m_Any(&weight)));
    auto biasP = m_Ranked(1, m_Typed<M::MO::TensorType>(m_Any(&bias)));
    if (!m_Op<MO::AddOp>(
             m_Optional(matmulP, m_Op<MO::BroadcastToOp>(matmulP, m_Any())),
             m_Optional(biasP, m_Op<MO::BroadcastToOp>(biasP, m_Any())))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !weight || !bias)
      return rewriter.notifyMatchFailure(op, "null input value");

    rewriter.replaceOpWithNewOp<MOGG::ExtOp>(
        op, op->getResultTypes(), "mop.kernels.fully_connected",
        SmallVector<Value>{input, weight, bias});

    return mlir::success();
  }
};
// TODO(#11975): move this logic to an interface on MatrixBandPartOp.
struct CommuteBroadcastWithMatBandPart
    : OpRewritePattern<MO::MatrixBandPartOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MO::MatrixBandPartOp bandPartOp,
                                PatternRewriter &rewriter) const override {
    Location loc = bandPartOp.getLoc();

    MO::TensorType resTy = bandPartOp.getType();
    if (!resTy.hasRank())
      return rewriter.notifyMatchFailure(loc, "not statically ranked");

    MO::TensorValue inputVal = bandPartOp.getInput();
    auto bcastOp = inputVal.getDefiningOp<MO::Static::BroadcastToOp>();
    if (!bcastOp)
      return rewriter.notifyMatchFailure(loc, "not preceded by broadcast");

    MO::TensorValue bcastInput = bcastOp.getInput();
    MO::TensorType bcastInputTy = bcastInput.getType();
    if (!bcastInputTy.hasRank()) {
      return rewriter.notifyMatchFailure(
          loc, "preceding broadcast has unranked input");
    }

    MO::TensorType inputTy = inputVal.getType();
    SmallVector<TypedAttr> newBcastResDims;
    if (bcastInputTy.getRank() > 2) {
      // If broadcasting in the batch dimensions, we check their equality.
      ArrayRef<TypedAttr> inBatchDims = bcastInputTy.getDims().drop_back(2);
      ArrayRef<TypedAttr> outBatchDims = inputTy.getDims().drop_back(2);
      if (inBatchDims.equals(outBatchDims)) {
        return rewriter.notifyMatchFailure(
            loc, "preceding broadcast with batch dims canonical");
      }
      newBcastResDims.append(inBatchDims.begin(), inBatchDims.end());
    } else {
      // We are broadcasting from something less than rank-2, so it's enough to
      // ensure that we are broadcasting to rank-2.
      if (inputTy.getRank() == 2) {
        return rewriter.notifyMatchFailure(
            loc, "preceding broadcast without batch dims canonical");
      }
    }

    // We will inject a new broadcast before the band part op that does not
    // broadcast in the batch dimensions. If there is no broadcasting in the
    // trailing two dimensions (e.g. because they are static and equal) the
    // broadcast canonicalizer will erase the op.
    ArrayRef<TypedAttr> matDims = inputTy.getDims().take_back(2);
    newBcastResDims.append(matDims.begin(), matDims.end());

    auto shapeTy = rewriter.getType<MO::ShapeType>();
    MO::TensorType newBcastResTy =
        inputTy.cloneWith(MO::ShapeAttr::get(newBcastResDims, shapeTy));
    auto newBcastOp = rewriter.create<MO::Static::BroadcastToOp>(
        loc, newBcastResTy, bcastInput);

    // We replace the bandpart op using a new input and result type.
    SmallVector<Value> newOperands(bandPartOp.getOperands());
    newOperands[0] = newBcastOp;
    auto newBandPartOp = rewriter.create<MO::MatrixBandPartOp>(
        loc, newBcastResTy, newOperands,
        bandPartOp->getAttrDictionary().getValue());

    // Note that the original result type is used
    rewriter.replaceOpWithNewOp<MO::Static::BroadcastToOp>(bandPartOp, resTy,
                                                           newBandPartOp);
    return success();
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
  patterns.add<GatherSumPattern, BatchMatmulAdjBPattern, MatmulTranspBPattern,
               FullyConnectedPattern, CommuteBroadcastWithMatBandPart>(ctx);
  if (useMatmulPacking.getValue())
    patterns.add<MatmulConstBPattern>(ctx);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

} // namespace M::MOGG
