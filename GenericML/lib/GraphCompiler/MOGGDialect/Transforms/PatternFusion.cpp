//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h"
#include "mlir/IR/BuiltinOps.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

#include "mlir/IR/IRMapping.h"

using namespace M;
using namespace mlir;
using namespace mlir::matchers;

namespace M::MOGG {

#define GEN_PASS_DEF_PATTERNFUSION
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct GatherSumPattern : public OpRewritePattern<MO::ReduceAddOp> {
  GatherSumPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::ReduceAddOp op,
                                PatternRewriter &rewriter) const override {
    Value input, indices, gatherAxis, reduceAxis;

    if (!m_Op<MO::ReduceAddOp>(
             m_Op<MO::GatherOp>(m_MOMatrix(m_Any(&input)),
                                m_MOMatrix(m_Any(&indices)),
                                m_ScalarConstant(&gatherAxis, 0)),
             m_ScalarConstant(&reduceAxis, 1))
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

struct MatmulConstBPattern : public OpRewritePattern<MO::MatmulOp> {
  MatmulConstBPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInputA();
    Value weight = op.getInputB();

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

struct MatmulTranspBPattern : public OpRewritePattern<MO::MatmulOp> {
  MatmulTranspBPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::MatmulOp op,
                                PatternRewriter &rewriter) const override {
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

    rewriter.replaceOpWithNewOp<MOGG::ExtOp>(
        op,
        /*outputTypes=*/op->getResultTypes(),
        /*op_name=*/"mop.kernels.matmul_transpb",
        /*inputOperands=*/
        SmallVector<Value>{input, weight});

    return mlir::success();
  }
};

struct BatchMatmulAdjBPattern : public OpRewritePattern<MO::BatchMatmulOp> {
  BatchMatmulAdjBPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
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

    rewriter.replaceOpWithNewOp<MOGG::ExtOp>(
        op,
        /*outputTypes=*/op->getResultTypes(),
        /*op_name=*/"mop.kernels.batch_matmul_adjb",
        /*inputOperands=*/
        SmallVector<Value>{input, weight});

    return mlir::success();
  }
};

struct FullyConnectedPattern : public RewritePattern {
  FullyConnectedPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag{}, 3, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    OpBuilder b(op);
    b.setInsertionPoint(op);

    Value input, weight, bias;
    Operation *matmul, *reshape;
    auto matmulP = m_Op<MO::MatmulOp>(&matmul, m_MOMatrix(m_Any(&input)),
                                      m_MOMatrix(m_Any(&weight)));
    auto broadcastP =
        m_Optional(matmulP, m_Op<MO::Static::BroadcastToOp>(matmulP));
    auto lhs = m_Optional(broadcastP,
                          m_Op<MO::Static::ReshapeOp>(&reshape, broadcastP));
    auto rhs = m_Op<MO::Static::BroadcastToOp>(
        m_Ranked(1, m_Typed<M::MO::TensorType>(m_Any(&bias))));
    if (isa<MO::AddOp>(op) &&
        !m_Optional(m_Op<MO::AddOp>(lhs, rhs), m_Op<MO::AddOp>(rhs, lhs))
             .match(op->getResult(0)))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !weight || !bias)
      return rewriter.notifyMatchFailure(op, "null input value");
    Value fc;
    if (!isa_and_nonnull<MO::ConstantOp>(weight.getDefiningOp())) {
      fc = rewriter
               .create<MOGG::ExtOp>(op->getLoc(), matmul->getResultTypes(),
                                    "mop.kernels.fully_connected",
                                    SmallVector<Value>{input, weight, bias})
               .getResult(0);
    } else {
      MO::TensorType weightsType = cast<MO::TensorType>(weight.getType());
      SmallVector<uint64_t> weightsDims;
      for (int64_t dim : weightsType.getIntDims()) {
        // NOTE this (constants with unknown shape) may occur as a result of
        // constant folding. Once we have shape inference on by default, we
        // won't have this anymore.
        if (dim < 0)
          return failure();
        weightsDims.push_back(dim);
      }
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

      fc = rewriter
               .create<MOGG::ExtOp>(
                   op->getLoc(), matmul->getResultTypes()[0],
                   "mop.kernels.fully_connected_packb",
                   SmallVector<Value>{input, packedWeights, bias}, weightsDims)
               .getResult(0);
    }

    if (reshape) {
      rewriter.replaceOpWithNewOp<MO::Static::ReshapeOp>(
          op, op->getResultTypes()[0], fc);
    } else {
      rewriter.replaceOp(op, fc);
    }

    return mlir::success();
  }
};

struct LiftTransposesPattern : public OpRewritePattern<MO::TransposeOp> {
  LiftTransposesPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    // input perm1            input  perm1
    //   \    /                 \      /
    //  transpose1             transpose1   perm2
    //     |            ===>        \        /
    //   unary  perm2               transpose2 (updated)
    //      \    /                       |
    //    transpose2 (op)              unary (updated)
    Value input, perm1, perm2;
    Operation *unaryOp;
    auto transpose1P = m_Op<MO::TransposeOp>(m_Any(&input), m_Any(&perm1));
    auto transpose2P = m_Op<MO::TransposeOp>(
        m_Op<MO::ElementWiseUnary>(&unaryOp, transpose1P), m_Any(&perm2));
    if (!transpose2P.match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !perm1 || !perm2)
      return rewriter.notifyMatchFailure(op, "null input or permutation value");

    // If unary op has multiple uses, we don't lift the transposes because it
    // may create extra computation.
    // TODO add [get|set]Input method to `ElementWiseUnary` interface
    Value unary = unaryOp->getResult(0);
    if (!unary.hasOneUse())
      return rewriter.notifyMatchFailure(op, "unary op has multiple uses");

    auto transpose2Op = cast<MO::TransposeOp>(op);
    transpose2Op.replaceAllUsesWith(unary);
    transpose2Op.getInputMutable().assign(unaryOp->getOperand(0));
    unaryOp->setOperand(0, transpose2Op.getResult());
    unary.setType(transpose2Op.getType());
    unaryOp->moveAfter(transpose2Op);

    return mlir::success();
  }
};

// TODO: Put this in a MO Transform Pass?
// Need to move pattern matching to a more generic place
struct ConvertBatchMatmulToMatmulPattern
    : public OpRewritePattern<MO::BatchMatmulOp> {
  ConvertBatchMatmulToMatmulPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx, 4) {}

  LogicalResult matchAndRewrite(MO::BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value input, weight;
    Operation *broadcast;
    if (!m_Op<MO::BatchMatmulOp>(m_Any(&input),
                                 m_Op<MO::Static::BroadcastToOp>(
                                     &broadcast, m_MOMatrix(m_Any(&weight))))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !weight)
      return rewriter.notifyMatchFailure(op, "null input value");

    auto inputTy = cast<MO::TensorType>(input.getType());
    auto weightTy = cast<MO::TensorType>(weight.getType());
    auto broadcastTy = cast<MO::TensorType>(broadcast->getResult(0).getType());
    auto bmmTy = cast<MO::TensorType>(op->getResult(0).getType());

    if (inputTy.hasUnknownShape() || weightTy.hasUnknownShape() ||
        bmmTy.hasUnknownShape() || !bmmTy.hasRank()) {
      return rewriter.notifyMatchFailure(
          op, "ops aren't parameterized, can't rewrite");
    }

    ArrayRef<TypedAttr> outDims =
        broadcastTy.getDims().drop_front(broadcastTy.getRank() - 2);
    ArrayRef<TypedAttr> inDims = weightTy.getDims();

    if (!inDims.equals(outDims)) {
      return rewriter.notifyMatchFailure(op, "Broadcast not on batch axis");
    }

    ArrayRef<TypedAttr> oldShape = inputTy.getDims();
    ArrayRef<TypedAttr> kVec = oldShape.take_back();
    TypedAttr batch =
        KGEN::ParamOperatorAttr::get(KGEN::POC::MulNuw, oldShape.drop_back(1));
    SmallVector<TypedAttr> newShape{batch, kVec[0]};

    auto shapeTy = rewriter.getType<MO::ShapeType>();
    MO::TensorType reshapeTy =
        inputTy.cloneWith(MO::ShapeAttr::get(newShape, shapeTy));
    Value reshape =
        rewriter.create<MO::Static::ReshapeOp>(op->getLoc(), reshapeTy, input)
            .getResult();

    ArrayRef<TypedAttr> weightShape = weightTy.getDims();
    MO::TensorType matmulTy = inputTy.cloneWith(
        MO::ShapeAttr::get({newShape[0], weightShape[1]}, shapeTy));

    Value matmul =
        rewriter.create<MO::MatmulOp>(op->getLoc(), matmulTy, reshape, weight)
            .getResult();

    rewriter.replaceOpWithNewOp<MO::Static::ReshapeOp>(op, bmmTy, matmul);

    return mlir::success();
  }
};

// TODO: Move this to an algebraic simplification pass after refactoring
// patterns out of MOGG
struct Pow2Pattern : public OpRewritePattern<MO::PowOp> {
  Pow2Pattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::PowOp op,
                                PatternRewriter &rewriter) const override {
    Value input;

    if (!m_Op<MO::PowOp>(m_Any(&input),
                         m_Op<MO::Static::BroadcastToOp>(m_ScalarConstant(2)))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input)
      return rewriter.notifyMatchFailure(op, "null input value");

    rewriter.replaceOpWithNewOp<MO::MulOp>(op, input, input);

    return mlir::success();
  }
};

// TODO: Move this to an algebraic simplification pass after refactoring
// patterns out of MOGG
struct GeluPattern : public OpRewritePattern<MO::MulOp> {
  GeluPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 4) {}

  LogicalResult matchAndRewrite(MO::MulOp op,
                                PatternRewriter &rewriter) const override {
    Value input;
    // gelu = x * 1/2 * (1 + erf(x/sqrt(2)))
    auto sq2 = m_Op<MO::Static::BroadcastToOp>(
        m_ScalarConstant(llvm::numbers::sqrt2f));
    auto divsq2 = m_Op<MO::Static::BroadcastToOp>(
        m_ScalarConstant(llvm::numbers::inv_sqrt2f));
    auto x = m_Any(&input);
    auto erf = m_Op<MO::ErfOp>(m_Optional(
        m_Op<MO::DivOp>(x, sq2), m_BinaryCommutativeOp<MO::MulOp>(x, divsq2)));
    auto one = m_Op<MO::Static::BroadcastToOp>(m_ScalarConstant(1.0f));
    auto add = m_BinaryCommutativeOp<MO::AddOp>(one, erf);

    auto half = m_Op<MO::Static::BroadcastToOp>(m_ScalarConstant(0.5f));
    auto x2 = m_Val(input);
    // TODO: write an associative commutative matcher?
    auto gelu =
        m_Optional(m_BinaryCommutativeOp<MO::MulOp>(
                       m_BinaryCommutativeOp<MO::MulOp>(x2, half), add),
                   m_BinaryCommutativeOp<MO::MulOp>(
                       half, m_BinaryCommutativeOp<MO::MulOp>(x2, add)));

    if (isa<MO::MulOp>(op) && gelu.match(op->getResult(0)))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input)
      return rewriter.notifyMatchFailure(op, "null input value");

    rewriter.replaceOpWithNewOp<MO::GeluOp>(op, input);

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
  patterns.add<GatherSumPattern, BatchMatmulAdjBPattern, MatmulTranspBPattern,
               LiftTransposesPattern, Pow2Pattern,
               ConvertBatchMatmulToMatmulPattern, FullyConnectedPattern,
               GeluPattern>(ctx);
  if (useMatmulPacking.getValue())
    patterns.add<MatmulConstBPattern>(ctx);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

} // namespace M::MOGG
