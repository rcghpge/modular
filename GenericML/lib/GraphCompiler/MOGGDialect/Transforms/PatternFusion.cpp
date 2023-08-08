//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOAttrs.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GenericML/GraphCompiler/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>

using namespace M;
using namespace mlir;
using namespace mlir::matchers;

namespace M::MOGG {

#define GEN_PASS_DEF_PATTERNFUSION
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// The size which we regard as too small to run the special matmuls.
constexpr int64_t kTinyMatmulFlops = 5000;

bool isMatmulSmall(MO::MatmulOp matmul) {
  MO::TensorType aType = matmul.getInputA().getType();
  MO::TensorType bType = matmul.getInputB().getType();

  if (!aType.hasStaticShape() || !bType.hasStaticShape())
    return false;

  SmallVector<int64_t> ADims = aType.getIntDims();
  SmallVector<int64_t> BDims = bType.getIntDims();

  // M * K * N
  int64_t approxFlops = ADims[0] * ADims[1] * BDims[1];
  return approxFlops < kTinyMatmulFlops;
}

struct PackParamCache {
  PackParamCache(MLIRContext *ctx) : ctx(ctx) {}

  MLIRContext *ctx;

  // Each packed function will return the same weight type for each MO tensor
  // (shape + element type). By preserving this information we get better
  // performance as we are able to recognise and DCE / CSE matching shape / pack
  // funcs.
  DenseMap<MO::TensorType, MO::TensorType> knownTypes, knownTypesTransp;

  // Each packed shape function should return a unique shape, we use this atomic
  // to ensure each gets a unique parameter name.
  std::atomic<int64_t> paramId = 0;
};

Value lowerMatmulPackingFunc(Location loc, PatternRewriter &rewriter,
                             PackParamCache &cache, Value weight,
                             bool transpose) {
  MO::TensorType weightsType = cast<MO::TensorType>(weight.getType());
  MO::TensorType packedWeightsType;

  // First we check the cache. If we've already lowered this same tensor &
  // transpose combination then we will want to lower to the exact same
  // parameter type.
  if (transpose) {
    auto itr = cache.knownTypesTransp.find(weightsType);
    if (itr != cache.knownTypesTransp.end())
      packedWeightsType = itr->second;
  } else {
    auto itr = cache.knownTypes.find(weightsType);
    if (itr != cache.knownTypes.end())
      packedWeightsType = itr->second;
  }

  // If we haven't lowered the shape function, lower it now so we have
  // parameterized shapes.
  if (!packedWeightsType) {
    Type paramType = rewriter.getIndexType();

    // Create a pair of parameters for the new type.
    auto newParams = KGEN::ParamDeclArrayAttr::get(
        cache.ctx,
        {KGEN::ParamDeclAttr::get(("MMPackSh" + Twine(cache.paramId++)).str(),
                                  paramType),
         KGEN::ParamDeclAttr::get(("MMPackSh" + Twine(cache.paramId++)).str(),
                                  paramType)});

    // Create the packed B shape function.
    // TODO(#18770) Find a better registration mechanism.
    SmallVector<mlir::NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("transpose_in_0",
                                          rewriter.getBoolAttr(transpose)));
    mlir::DictionaryAttr kgenParams = rewriter.getDictionaryAttr(attrs);
    rewriter.create<MO::InvokeShapeFuncOp>(loc, SmallVector<Value>{weight},
                                           "pack_matmul_b_shape_func",
                                           newParams, kgenParams, ArrayAttr{});

    SmallVector<TypedAttr> outputShapeDims;
    for (KGEN::ParamDeclAttr outputParamDecl : newParams)
      outputShapeDims.push_back(KGEN::ParamDeclRefAttr::get(outputParamDecl));

    auto moshShapeTy = MOSH::ShapeType::get(cache.ctx);
    auto shapeAttr = MOSH::ShapeAttr::get(outputShapeDims, moshShapeTy);
    auto packedWeightsLayout = rewriter.getAttr<MO::LayoutAttr>("KNkni");
    packedWeightsType =
        weightsType.cloneWith(shapeAttr).cloneWith(packedWeightsLayout);

    // Cache the result so the next type has a hit too.
    if (transpose)
      cache.knownTypesTransp[weightsType] = packedWeightsType;
    else
      cache.knownTypes[weightsType] = packedWeightsType;
  }

  // Pack the mm weights.
  return rewriter.create<MO::LayoutTransformOp>(
      loc, /*outputTypes=*/packedWeightsType, weight);
}

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
  MatmulConstBPattern(MLIRContext *ctx, PackParamCache &cache)
      : OpRewritePattern(ctx, 3), cache(cache) {}

  LogicalResult matchAndRewrite(MO::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInputA();
    Value weight, perm;
    Operation *transpose = nullptr;

    // If all matmul sizes kTinyMatmulSize.
    if (isMatmulSmall(op))
      return rewriter.notifyMatchFailure(op, "Matmul is too small");

    auto weightP = m_MOMatrix(m_Any(&weight));
    auto weightTransP =
        m_Op<MO::TransposeOp>(&transpose, weightP, m_MOVector(m_Any(&perm)));
    if (!m_Optional(weightTransP, weightP).match(op.getInputB()))
      return failure();

    MO::TensorType weightsType = cast<MO::TensorType>(weight.getType());
    MO::LayoutAttr weightsLayout = op.getInputB().getType().getLayoutAttr();
    if (weightsLayout.getFormat() == "KNkni")
      return rewriter.notifyMatchFailure(op, "Matmul is already packed");

    if (weightsLayout.getFormat() == "NK")
      return rewriter.notifyMatchFailure(op,
                                         "Transposed op has already been run");

    if (weightsType.hasUnknownShape())
      return rewriter.notifyMatchFailure(op, "Weights have unknown shape");

    if (!input || !isa_and_nonnull<MO::ConstantOp>(weight.getDefiningOp()))
      return failure();

    if (transpose) {
      if (!perm || validate2DTranspose(perm).failed())
        return rewriter.notifyMatchFailure(op, "invalid transpose");
    }

    // TODO once we have layout inference across transpose op, we can remove
    // this check and layout annotation; or we might remove this `transpose`
    // flag from the packing shape function altogether.
    if (transpose) {
      if (!weight.hasOneUse())
        return rewriter.notifyMatchFailure(
            op, "pre-transposed weights have multiple uses");
      auto preTranspWeightsLayout = rewriter.getAttr<MO::LayoutAttr>("NK");
      weight.setType(weightsType.cloneWith(preTranspWeightsLayout));
    }

    Value packedWeights = lowerMatmulPackingFunc(op->getLoc(), rewriter, cache,
                                                 weight, transpose);

    op.setOperand(1, packedWeights);
    return mlir::success();
  }

private:
  PackParamCache &cache;
};

struct MatmulTranspBPattern : public OpRewritePattern<MO::MatmulOp> {
  MatmulTranspBPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value input, weight, perm;

    auto weightTransP =
        m_Op<MO::TransposeOp>(m_Any(&weight), m_MOVector(m_Any(&perm)));
    if (!m_Op<MO::MatmulOp>(
             m_Any(&input),
             m_Optional(m_Op<MO::RebindOp>(weightTransP), weightTransP))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !weight || !perm)
      return rewriter.notifyMatchFailure(op, "null input value");
    if (validate2DTranspose(perm).failed())
      return rewriter.notifyMatchFailure(op, "invalid transpose");

    MO::LayoutAttr weightsLayout = op.getInputB().getType().getLayoutAttr();
    if (weightsLayout.getFormat() == "NK")
      return rewriter.notifyMatchFailure(op, "Matmul is already transposed");

    if (weightsLayout.getFormat() == "KNkni")
      return rewriter.notifyMatchFailure(
          op, "The packed optimization has been run already");

    // TODO once we have layout inference across transpose op, we can remove
    // this check and layout annotation.
    if (!weight.hasOneUse())
      return rewriter.notifyMatchFailure(op, "weights have multiple uses");
    // Take the non-transposed weight.
    op.setOperand(1, weight);
    auto preTranspWeightsLayout = rewriter.getAttr<MO::LayoutAttr>("NK");
    weight.setType(cast<MO::TensorType>(weight.getType())
                       .cloneWith(preTranspWeightsLayout));

    return mlir::success();
  }
};

struct BatchMatmulAdjBPattern : public OpRewritePattern<MO::BatchMatmulOp> {
  BatchMatmulAdjBPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value input, weight, perm;
    auto weightTransP =
        m_Op<MO::TransposeOp>(m_Any(&weight), m_MOVector(m_Any(&perm)));
    if (!m_Op<MO::BatchMatmulOp>(
             m_Any(&input),
             m_Optional(m_Op<MO::RebindOp>(weightTransP), weightTransP))
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

    // Take the non-transposed weight.
    op.setOperand(1, weight);

    // Mark this matmul as being a tranposed kernel.
    op.setTransposeBAttr(rewriter.getBoolAttr(true));

    return mlir::success();
  }
};

struct ConcatZeroAxisPattern : public OpRewritePattern<MO::ConcatOp> {
  ConcatZeroAxisPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::ConcatOp concat,
                                PatternRewriter &rewriter) const override {
    // Doing this manually, I don't think the patterns can support variadic
    // inputs yet
    std::optional<int64_t> axis =
        MO::tryGetConstantInt(cast<MO::TensorValue>(concat.getAxis()));
    if (!axis.has_value() || axis.value() != 0)
      return rewriter.notifyMatchFailure(concat, "axis not constant 0");

    Block *concatBlock = concat.getResult().getParentBlock();
    for (Value input : concat.getInputs()) {
      if (!input.hasOneUse())
        return rewriter.notifyMatchFailure(input.getLoc(), "has extra users");
      if (input.getParentBlock() != concatBlock) {
        return rewriter.notifyMatchFailure(
            input.getLoc(), "concat in a different block than inputs");
      }
      // ensure inputs aren't constant or used anywhere else
      if (mlir::Operation *iop = input.getDefiningOp()) {
        if (isa<MO::ConstantOp>(iop))
          return rewriter.notifyMatchFailure(iop, "is constant");
        if (isa<MO::ConcatOp>(iop) || isa<MOGG::ConcatOp>(iop)) {
          return rewriter.notifyMatchFailure(
              iop, "don't currently support nesting in-place concats");
        }
      } else {
        return rewriter.notifyMatchFailure(input.getLoc(),
                                           "input is a block argument");
      }
    }

    rewriter.replaceOpWithNewOp<MOGG::ConcatOp>(
        concat,
        /*outputTypes=*/concat->getResultTypes(),
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

    // Handle Scalar beta and gamma
    auto betaType = beta.getType().cast<MO::TensorType>();
    if (betaType && betaType.getRank() == 0) {
      beta = rewriter.create<MO::Static::BroadcastToOp>(
          op.getLoc(),
          MO::TensorType::get(SmallVector<TypedAttr>{op.getType().getDim(2)},
                              betaType.getElementType()),
          beta);
    }

    auto gammaType = gamma.getType().cast<MO::TensorType>();
    if (gammaType && gammaType.getRank() == 0) {
      gamma = rewriter.create<MO::Static::BroadcastToOp>(
          op.getLoc(),
          MO::TensorType::get(SmallVector<TypedAttr>{op.getType().getDim(2)},
                              gammaType.getElementType()),
          beta);
    }

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

} // namespace

struct PatternFusionPass : public impl::PatternFusionBase<PatternFusionPass> {
  using PatternFusionBase::PatternFusionBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns{ctx};

    PackParamCache cache{ctx};
    patterns.add<MatmulConstBPattern>(ctx, cache);

    patterns.add<GatherSumPattern, BatchMatmulAdjBPattern, MatmulTranspBPattern,
                 ConcatZeroAxisPattern, SplitZeroAxisPattern, LayerNormPattern>(
        ctx);

    MO::GraphOp graphOp = getOperation();
    if (failed(applyPatternsAndFoldGreedily(graphOp, std::move(patterns))))
      return signalPassFailure();
    MO::synchronizeMetadataUpdates(graphOp);
  }
};

} // namespace M::MOGG
