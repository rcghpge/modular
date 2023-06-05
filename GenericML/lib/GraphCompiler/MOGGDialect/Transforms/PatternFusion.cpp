//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GenericML/GraphCompiler/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

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

static LogicalResult validate2DTranspose(Value perm) {
  std::optional<SmallVector<int64_t>> maybePerm =
      MO::tryGetConstantIntVector(cast<MO::TensorValue>(perm));

  if (!maybePerm.has_value())
    return failure();

  SmallVector<int64_t> &permValues = maybePerm.value();

  if (permValues[0] != 1 || permValues[1] != 0)
    return failure();
  return success();
}

struct MatmulConstBPattern : public OpRewritePattern<MO::MatmulOp> {
  MatmulConstBPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 3) {}

  LogicalResult matchAndRewrite(MO::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInputA();
    Value weight, perm;
    Operation *transpose = nullptr;
    auto weightP = m_MOMatrix(m_Any(&weight));
    if (!m_Optional(m_Op<MO::TransposeOp>(&transpose, weightP,
                                          m_MOVector(m_Any(&perm))),
                    weightP)
             .match(op.getInputB()))
      return failure();

    if (!input || !isa_and_nonnull<MO::ConstantOp>(weight.getDefiningOp()))
      return failure();

    MO::TensorType weightsType = cast<MO::TensorType>(weight.getType());
    if (!weightsType.hasStaticShape() || weightsType.hasUnknownShape())
      return failure();

    std::string packing_kernel = "mop.kernels.pack_matmul_b";
    if (transpose) {
      if (!perm || validate2DTranspose(perm).failed())
        return rewriter.notifyMatchFailure(op, "invalid transpose");
      packing_kernel = "mop.kernels.pack_matmul_b.transp_b";
    }
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
                                 /*op_name=*/packing_kernel,
                                 /*inputOperands=*/SmallVector<Value>{weight})
            .getResult(0);

    rewriter.replaceOpWithNewOp<MOGG::ExtOp>(
        op, op->getResultTypes(), "mop.kernels.matmul_packb",
        SmallVector<Value>{input, packedWeights});

    return mlir::success();
  }
};

struct MatmulTranspBPattern : public OpRewritePattern<MO::MatmulOp> {
  MatmulTranspBPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value input, weight, perm;
    if (!m_Op<MO::MatmulOp>(
             m_Any(&input),
             m_Op<MO::TransposeOp>(m_Any(&weight), m_MOVector(m_Any(&perm))))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !weight || !perm)
      return rewriter.notifyMatchFailure(op, "null input value");
    if (validate2DTranspose(perm).failed())
      return rewriter.notifyMatchFailure(op, "invalid transpose");
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
    Value input, weight, perm;
    if (!m_Op<MO::BatchMatmulOp>(
             m_Any(&input),
             m_Op<MO::TransposeOp>(m_Any(&weight), m_MOVector(m_Any(&perm))))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !weight || !perm)
      return rewriter.notifyMatchFailure(op, "null input value");

    std::optional<SmallVector<int64_t>> maybePerm =
        MO::tryGetConstantIntVector(cast<MO::TensorValue>(perm));

    if (!maybePerm.has_value())
      return rewriter.notifyMatchFailure(op, "perm not constant");

    SmallVector<int64_t> &permValues = maybePerm.value();

    int64_t rank = permValues.size();

    if (permValues[rank - 1] != rank - 2 || permValues[rank - 2] != rank - 1)
      return rewriter.notifyMatchFailure(op, "perm not adjoint");

    for (int64_t i = 0, e = rank - 2; i < e; ++i)
      if (permValues[i] != i)
        return rewriter.notifyMatchFailure(op, "perm not adjoint");

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

  // Return null if we fail to get a rank-1 shape after squeezing `ty`'s shape
  static MO::TensorType tryGetVectorTyAfterSqueeze(MO::TensorType ty) {
    if (!ty.hasRank())
      return {};

    MOSH::ShapeAttr shapeAttr = ty.getTypedShapeAttr();
    SmallVector<TypedAttr> dimsAfterSqueeze;
    for (TypedAttr dim : shapeAttr.getValues()) {
      auto intAttrDim = dyn_cast<IntegerAttr>(dim);
      // Squeeze only if we see a static 1
      if (intAttrDim && intAttrDim.getInt() == 1)
        continue;
      dimsAfterSqueeze.push_back(dim);
    }

    if (dimsAfterSqueeze.size() == 1) {
      return ty.cloneWith(
          MOSH::ShapeAttr::get(dimsAfterSqueeze, shapeAttr.getType()));
    }
    return {};
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value input, weight, bias;
    Operation *matmul = nullptr, *add = nullptr, *reshape = nullptr,
              *transpose = nullptr, *gelu_approx_sigmoid = nullptr,
              *gelu = nullptr, *relu = nullptr, *tanh = nullptr,
              *sigmoid = nullptr;

    // Pattern For the Matmul including possible transpose on the weight
    // and broadcast after the matmul
    auto weightP = m_MOMatrix(m_Any(&weight));
    auto matmulP = m_Op<MO::MatmulOp>(
        &matmul, m_MOMatrix(m_Any(&input)),
        m_Optional(m_Op<MO::TransposeOp>(&transpose, weightP, m_Any()),
                   weightP));

    // Pattern for the Bias Add  including possible reshaping of the matmul
    // and broadcasting of the bias vector
    auto broadcastP =
        m_Optional(matmulP, m_Op<MO::Static::BroadcastToOp>(matmulP));
    auto biasP = m_Typed<MO::TensorType>(m_Any(&bias));
    auto bias_add = m_BinaryCommutativeOp<MO::AddOp>(
        &add,
        m_Optional(broadcastP,
                   m_Op<MO::Static::ReshapeOp>(&reshape, broadcastP)),
        m_Optional(m_Op<MO::Static::BroadcastToOp>(biasP), biasP));

    // Patterns for Activation Functions
    auto approxGeluP = m_BinaryCommutativeOp<MO::MulOp>(
        &gelu_approx_sigmoid, bias_add,
        m_Op<MO::SigmoidOp>(m_BinaryCommutativeOp<MO::MulOp>(
            bias_add,
            m_Op<MO::Static::BroadcastToOp>(m_ScalarConstant(1.702f)))));
    auto geluP = m_Op<MO::GeluOp>(&gelu, bias_add);
    auto reluP = m_Op<MO::ReluOp>(&relu, bias_add);
    auto tanhP = m_Op<MO::TanhOp>(&tanh, bias_add);
    auto sigmoidP = m_Op<MO::SigmoidOp>(&sigmoid, bias_add);

    auto fully_connected =
        m_Optional(approxGeluP, geluP, reluP, tanhP, sigmoidP, bias_add);
    // TODO: Make the Optional Matcher work with Operation* directly
    if (op->getNumResults() > 0 && !fully_connected.match(op->getResult(0)))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !weight || !bias)
      return rewriter.notifyMatchFailure(op, "null input value");

    // NOTE we need this because sometimes the bias comes in the form [1, N],
    // so we must match such cases.
    MO::TensorType vecBiasTy =
        tryGetVectorTyAfterSqueeze(cast<MO::TensorValue>(bias).getType());
    if (!vecBiasTy)
      return rewriter.notifyMatchFailure(op, "bias is not a vector");

    // NOTE The FC kernel requires bias to be rank-1, so we reshape if needed.
    Location loc = op->getLoc();
    if (bias.getType() != vecBiasTy)
      bias = rewriter.create<MO::Static::ReshapeOp>(loc, vecBiasTy, bias);

    Value fc;
    if (!isa_and_nonnull<MO::ConstantOp>(weight.getDefiningOp())) {
      if (gelu_approx_sigmoid || gelu || relu || tanh || sigmoid)
        return rewriter.notifyMatchFailure(
            op, "haven't implemented activation fusions with out packb");
      fc = rewriter
               .create<MOGG::ExtOp>(loc,
                                    reshape ? matmul->getResultTypes()[0]
                                            : add->getResultTypes()[0],
                                    "mop.kernels.fully_connected",
                                    SmallVector<Value>{input, weight, bias})
               .getResult(0);
    } else {
      Value weights = transpose ? transpose->getResult(0) : weight;
      MO::TensorType weightsType = cast<MO::TensorType>(weights.getType());
      MO::TensorType packedWeightsType = weightsType;
      if (weightsType.hasRank()) {
        SmallVector<int64_t> unknownDims(weightsType.getRank(),
                                         mlir::ShapedType::kDynamic);
        Type dtype = weightsType.getElementType();
        packedWeightsType = weightsType.cloneWith(unknownDims, dtype);
      }
      Value packedWeights =
          rewriter
              .create<MOGG::ExtOp>(
                  loc,
                  /*outputTypes=*/packedWeightsType,
                  /*op_name=*/"mop.kernels.pack_matmul_b",
                  /*inputOperands=*/SmallVector<Value>{weights})
              .getResult(0);

      std::string kernel;
      if (gelu_approx_sigmoid)
        kernel = "mop.kernels.fully_connected.packb.gelu_approx_sigmoid";
      else if (gelu)
        kernel = "mop.kernels.fully_connected.packb.gelu";
      else if (sigmoid)
        kernel = "mop.kernels.fully_connected.packb.sigmoid";
      else if (relu)
        kernel = "mop.kernels.fully_connected.packb.relu";
      else if (tanh)
        kernel = "mop.kernels.fully_connected.packb.tanh";
      else
        kernel = "mop.kernels.fully_connected.packb";

      fc = rewriter
               .create<MOGG::ExtOp>(
                   loc,
                   reshape ? matmul->getResultTypes()[0]
                           : add->getResultTypes()[0],
                   kernel, SmallVector<Value>{input, packedWeights, bias})
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

struct ConcatZeroAxisPattern : public RewritePattern {
  ConcatZeroAxisPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag{}, 2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Doing this manually, I don't think the patterns can support variadic
    // inputs yet
    auto concat = dyn_cast<MO::ConcatOp>(op);
    if (!concat)
      return rewriter.notifyMatchFailure(op, "Not a Concat Op");

    std::optional<int64_t> axis =
        MO::tryGetConstantInt(cast<MO::TensorValue>(concat.getAxis()));
    if (!axis.has_value() || axis.value() != 0)
      return rewriter.notifyMatchFailure(op, "axis not constant 0");

    for (Value input : concat.getInputs()) {
      // ensure inputs aren't constant or used anywhere else
      if (mlir::Operation *iop = input.getDefiningOp()) {
        if (isa<MO::ConstantOp>(iop))
          return rewriter.notifyMatchFailure(iop, "is constant");
        if (isa<MO::ConcatOp>(iop) || isa<MOGG::ConcatOp>(iop))
          return rewriter.notifyMatchFailure(
              iop, "don't currently support nesting in-place concats");
      }
      if (!input.hasOneUse())
        return rewriter.notifyMatchFailure(input.getLoc(), "has extra users");
    }

    rewriter.replaceOpWithNewOp<MOGG::ConcatOp>(
        op,
        /*outputTypes=*/op->getResultTypes(),
        /*inputOperands=*/concat.getInputs());

    return mlir::success();
  }
};

struct SplitZeroAxisPattern : public OpRewritePattern<MO::SplitOp> {
  SplitZeroAxisPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::SplitOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<int64_t> axis =
        MO::tryGetConstantInt(cast<MO::TensorValue>(op.getAxis()));
    if (!axis.has_value() || axis.value() != 0)
      return rewriter.notifyMatchFailure(op, "axis not constant 0");

    // Ensure input isn't constant
    if (op.getInput().getDefiningOp<MO::ConstantOp>())
      return rewriter.notifyMatchFailure(op, "input is a constant");

    rewriter.replaceOpWithNewOp<MOGG::SplitOp>(
        op,
        /*outputTypes=*/op->getResultTypes(),
        /*input=*/op.getInput(),
        /*splitSizes=*/op.getSplitSizes());

    return mlir::success();
  }
};

struct LayerNormPattern : public OpRewritePattern<MO::AddOp> {
  LayerNormPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 6) {}

  LogicalResult matchAndRewrite(MO::AddOp op,
                                PatternRewriter &rewriter) const override {
    // x = input + skip
    // layernorm = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
    Value gamma, beta, eps;
    auto gam = m_Op<MO::Static::BroadcastToOp>(m_Any(&gamma));
    auto bet = m_Op<MO::Static::BroadcastToOp>(m_Any(&beta));
    auto ep = m_Op<MO::Static::BroadcastToOp>(m_Any(&eps));

    auto last_axis = m_Optional(m_ScalarConstant(2), m_ScalarConstant(-1));
    // TODO: Current capture system can't distinguish multiple uses of the same
    // value in the pattern, need to fix, this might be slightly risky
    Value input, skip;
    auto x =
        m_Optional(m_BinaryCommutativeOp<MO::AddOp>(m_Ranked(3, m_Any(&input)),
                                                    m_Ranked(3, m_Any(&skip))),
                   m_Ranked(3, m_Any(&input)));
    auto mean = m_Op<MO::Static::BroadcastToOp>(m_Op<MO::MeanOp>(x, last_axis));
    auto diff = m_Op<MO::SubOp>(x, mean);
    auto var = m_Op<MO::MeanOp>(m_Op<MO::MulOp>(diff, diff), last_axis);
    // pytorch specific pattern
    auto p_ratio =
        m_Op<MO::DivOp>(diff, m_Op<MO::Static::BroadcastToOp>(m_Op<MO::SqrtOp>(
                                  m_BinaryCommutativeOp<MO::AddOp>(var, ep))));
    auto p_layernorm = m_BinaryCommutativeOp<MO::AddOp>(
        m_BinaryCommutativeOp<MO::MulOp>(gam, p_ratio), bet);

    // TF specific Pattern
    auto t_rDenom = m_BinaryCommutativeOp<MO::MulOp>(
        gam, m_Op<MO::Static::BroadcastToOp>(
                 m_Op<MO::RsqrtOp>(m_BinaryCommutativeOp<MO::AddOp>(var, ep))));
    auto t_term1 = m_BinaryCommutativeOp<MO::MulOp>(x, t_rDenom);
    auto t_term2 = m_BinaryCommutativeOp<MO::MulOp>(mean, t_rDenom);
    auto t_term3 = m_Op<MO::SubOp>(bet, t_term2);
    auto t_layernorm = m_BinaryCommutativeOp<MO::AddOp>(t_term1, t_term3);

    if (!t_layernorm.match(op.getResult()) &&
        !p_layernorm.match(op.getResult()))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !beta || !gamma || !eps)
      return rewriter.notifyMatchFailure(op, "null input value");
    if (!skip) {
      rewriter.replaceOpWithNewOp<MOGG::ExtOp>(
          op,
          /*outputTypes=*/op->getResultTypes(),
          /*op_name=*/"mop.kernels.layer_norm",
          /*inputOperands=*/
          SmallVector<Value>{input, gamma, beta, eps});
    } else {
      rewriter.replaceOpWithNewOp<MOGG::ExtOp>(
          op,
          /*outputTypes=*/op->getResultTypes(),
          /*op_name=*/"mop.kernels.layer_norm.skip",
          /*inputOperands=*/
          SmallVector<Value>{input, gamma, beta, eps, skip});
    }
    return mlir::success();
    return failure();
  }
};

struct MultiHeadAttentionPattern : public OpRewritePattern<MO::BatchMatmulOp> {
  MultiHeadAttentionPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  // Resolve any specified `mulScalar` or `divScalar` to be a multiplicative f32
  // scale, or return null if such conversion failed. If none of these scalars
  // are specified (i.e., both null), return a new scalar 1 of type f32.
  static Value resolveScale(Value mulScalar, Value divScalar, Location loc,
                            RewriterBase &rewriter) {
    auto tensorMulScalar = dyn_cast_or_null<MO::TensorValue>(mulScalar);
    auto tensorDivScalar = dyn_cast_or_null<MO::TensorValue>(divScalar);

    // TODO support non-f32 scalar
    if (tensorMulScalar) {
      if (tensorMulScalar.getType().getElementType().isF32())
        return mulScalar;

      return {};
    }

    if (tensorDivScalar) {
      auto scalarType = tensorDivScalar.getType();
      if (scalarType.getElementType().isF32()) {
        auto one = rewriter.create<MO::ConstantOp>(loc, 1.0f);
        return rewriter.create<MO::DivOp>(loc, scalarType, one, divScalar);
      }

      return {};
    }

    return rewriter.create<MO::ConstantOp>(loc, 1.0f);
  }

  LogicalResult matchAndRewrite(MO::BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
    //   bmm1 = mo.batch_matmul(query, keyT)
    // * bmm1 = (bmm1 * scalar) or (bmm1 / scalar)
    // * bmm1 = bmm1 + mask
    //   softmax = mo.softmax(bmm1)
    // * broadcast = mo.static.broadcast_to(softmax)
    //   res = mo.batch_matmul(broadcast, value)
    //
    // ==>
    //
    //   res = "mop.kernels.multi_head_attention"(query, keyT, value, scale)
    // or
    //   res = "mop.kernels.multi_head_attention_masked"(
    //          query, keyT, value, scale, mask)
    //
    // NOTE `*` means optional
    Value query, keyT, value, mulScalar, divScalar, mask;
    Operation *bmm1Op = nullptr, *softmaxOp = nullptr, *broadcastOp = nullptr;
    auto queryP = m_Any(&query);
    auto keyTP = m_Any(&keyT);
    auto bmm1P = m_Op<MO::BatchMatmulOp>(&bmm1Op, queryP, keyTP);
    auto mulBmm1P = m_BinaryCommutativeOp<MO::MulOp>(
        bmm1P, m_Op<MO::Static::BroadcastToOp>(m_Any(&mulScalar)));
    auto divBmm1P = m_Op<MO::DivOp>(
        bmm1P, m_Op<MO::Static::BroadcastToOp>(m_Any(&divScalar)));
    auto scaledBmm1P = m_Optional(bmm1P, mulBmm1P, divBmm1P);
    // TODO
    // 1. eliminate broadcast on `scaledBmm1P` via more aggressive shape
    // inference
    // 2. fuse in the broadcast on `mask`
    auto maskedBmm1P =
        m_Optional(scaledBmm1P,
                   m_BinaryCommutativeOp<MO::AddOp>(
                       m_Optional(scaledBmm1P,
                                  m_Op<MO::Static::BroadcastToOp>(scaledBmm1P)),
                       m_Any(&mask)));
    auto softmaxP = m_Op<MO::SoftmaxOp>(&softmaxOp, maskedBmm1P);
    // TODO remove this once shape inference can eliminate these broadcasts in
    // top models.
    auto broadcastP = m_Optional(
        softmaxP, m_Op<MO::Static::BroadcastToOp>(&broadcastOp, softmaxP));
    auto valueP = m_Any(&value);
    auto bmm2P = m_Op<MO::BatchMatmulOp>(broadcastP, valueP);

    if (!bmm2P.match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    // Don't fuse if intermediate results are used elsewhere
    if (!bmm1Op->hasOneUse())
      return rewriter.notifyMatchFailure(op, "first bmm has multiple users");

    if (!softmaxOp->hasOneUse())
      return rewriter.notifyMatchFailure(op, "softmax has multiple users");

    if (broadcastOp && !broadcastOp->hasOneUse())
      return rewriter.notifyMatchFailure(op, "broadcast has multiple users");

    Value scale = resolveScale(mulScalar, divScalar, op.getLoc(), rewriter);
    if (!scale)
      return rewriter.notifyMatchFailure(op, "failed to resolve scale");

    // Prepare kernell name and arguments
    StringRef kernelName = "mop.kernels.multi_head_attention";
    SmallVector<Value> kernelArgs{query, keyT, value, scale};
    if (mask) {
      kernelName = "mop.kernels.multi_head_attention_masked";
      kernelArgs.push_back(mask);
    }

    // Replace with fused kernel call
    rewriter.replaceOpWithNewOp<MOGG::ExtOp>(op, op.getResult().getType(),
                                             kernelName, kernelArgs);

    return mlir::success();
  }
};

} // namespace

struct PatternFusionPass : public impl::PatternFusionBase<PatternFusionPass> {
  using PatternFusionBase::PatternFusionBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns{ctx};
    patterns
        .add<GatherSumPattern, BatchMatmulAdjBPattern, MatmulTranspBPattern,
             FullyConnectedPattern, ConcatZeroAxisPattern, SplitZeroAxisPattern,
             LayerNormPattern, MatmulConstBPattern, MultiHeadAttentionPattern>(
            ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace M::MOGG
