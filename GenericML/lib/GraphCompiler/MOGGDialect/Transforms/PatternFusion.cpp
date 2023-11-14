//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOAttrs.h"
#include "GenericML/GraphCompiler/MODialect/MODevices.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GenericML/GraphCompiler/Matchers.h"
#include "GenericML/Support/DataflowUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

#include <mlir/IR/BuiltinAttributes.h>
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
constexpr int64_t kTinyBMMatmulFlops = 512;

static bool isMatmulSmall(MO::MatmulOp matmul) {
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

static bool isBMMSmall(MO::BatchMatmulOp matmul) {
  MO::TensorType aType = matmul.getInputA().getType();
  MO::TensorType bType = matmul.getInputB().getType();

  if (!aType.hasStaticShape() || !bType.hasStaticShape())
    return false;

  SmallVector<int64_t> ADims = aType.getIntDims();
  SmallVector<int64_t> BDims = bType.getIntDims();

  size_t rank = aType.getRank();
  int64_t batch_size = 1;
  for (size_t i = 0; i < ADims.size() - 2; ++i)
    batch_size *= ADims[i];

  // M * K * N
  int64_t approxFlops =
      batch_size * (ADims[rank - 2] * ADims[rank - 1] * BDims[rank - 1]);
  return approxFlops < kTinyBMMatmulFlops;
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

Value lowerMatmulPackingFunc(Location loc, MO::DeviceAwareRewriter &rewriter,
                             PackParamCache &cache, Value weight,
                             Type inputDtype, Type resultDtype,
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

  SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("a_type", TypeAttr::get(inputDtype)));
  attrs.push_back(rewriter.getNamedAttr("c_type", TypeAttr::get(resultDtype)));

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
    SmallVector<mlir::NamedAttribute> shapeFuncAttrs = attrs;
    shapeFuncAttrs.push_back(rewriter.getNamedAttr(
        "transpose_in_0", rewriter.getBoolAttr(transpose)));
    mlir::DictionaryAttr kgenParams =
        rewriter.getDictionaryAttr(shapeFuncAttrs);
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

  mlir::DictionaryAttr kgenParams = rewriter.getDictionaryAttr(attrs);
  // Pack the mm weights.
  return rewriter.create<MO::LayoutTransformOp>(
      loc, /*outputTypes=*/packedWeightsType, weight, kgenParams);
}

struct GatherSumPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern, MO::ReduceAddOp> {
  GatherSumPattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::ReduceAddOp op,
                             MO::DeviceAwareRewriter &rewriter) const override {
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

    rewriter.replaceOpWithNewOp<MO::GatherSumOp>(
        op, op->getResultTypes(), SmallVector<Value>{input, indices});

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

struct MatmulConstBPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern, MO::MatmulOp> {
  MatmulConstBPattern(MLIRContext *ctx, PackParamCache &cache)
      : DeviceAwareRewritePattern(ctx, 3), cache(cache) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::MatmulOp op,
                             MO::DeviceAwareRewriter &rewriter) const override {
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

    if (!input ||
        !dependsOnConstantsOnly<MO::ConstantOp>(weight.getDefiningOp()))
      return failure();

    if (transpose) {
      if (!perm || validate2DTranspose(perm).failed())
        return rewriter.notifyMatchFailure(op, "invalid transpose");
    }

    if (!rewriter.addOps({transpose})) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    if (!rewriter.deviceAnnots.isConsistentWith(MO::getHostDeviceAny(op))) {
      // TODO(#25114): Support outside of host.
      return rewriter.notifyMatchFailure(
          op, "packing is only supported on the host device");
    }

    // TODO once we have layout inference across transpose op, we can remove
    // this check and layout annotation; or we might remove this `transpose`
    // flag from the packing shape function altogether.
    // TODO in order to recover the transpose in a `reshape(transpose(weight))`,
    // we'll need to sink transpose through reshape, but we'll punt on that
    // since (1). this sequence of computations will be hoisted in MGP anyway so
    // we don't gain anything, and (2). the transpose checking here might be
    // replaced by more general view fusion into matmul.
    if (transpose) {
      if (!weight.hasOneUse())
        return rewriter.notifyMatchFailure(
            op, "pre-transposed weights have multiple uses");
      auto preTranspWeightsLayout = rewriter.getAttr<MO::LayoutAttr>("NK");
      weight.setType(weightsType.cloneWith(preTranspWeightsLayout));
    }

    Type inputDtype = op.getInputA().getType().getElementType();
    Type resultDtype = op.getType().getElementType();
    Value packedWeights =
        lowerMatmulPackingFunc(op->getLoc(), rewriter, cache, weight,
                               inputDtype, resultDtype, transpose);

    op.setOperand(1, packedWeights);
    return mlir::success();
  }

private:
  PackParamCache &cache;
};

struct MatmulTranspBPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern, MO::MatmulOp> {
  MatmulTranspBPattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::MatmulOp op,
                             MO::DeviceAwareRewriter &rewriter) const override {
    Value input, weight, perm;

    auto weightTransP = m_Op<MO::TransposeOp>(rewriter.pushOp(), m_Any(&weight),
                                              m_MOVector(m_Any(&perm)));
    if (!m_Op<MO::MatmulOp>(
             m_Any(&input),
             m_Optional(m_Op<MO::RebindOp>(rewriter.pushOp(), weightTransP),
                        weightTransP))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !weight || !perm)
      return rewriter.notifyMatchFailure(op, "null input value");
    if (validate2DTranspose(perm).failed())
      return rewriter.notifyMatchFailure(op, "invalid transpose");

    MO::LayoutAttr weightLayout = op.getInputB().getType().getLayoutAttr();
    if (weightLayout.getFormat() == "NK")
      return rewriter.notifyMatchFailure(op, "Matmul is already transposed");

    if (weightLayout.getFormat() == "KNkni")
      return rewriter.notifyMatchFailure(
          op, "The packed optimization has been run already");

    MO::TensorType weightType = cast<MO::TensorType>(weight.getType());
    auto preTranspWeightLayout = rewriter.getAttr<MO::LayoutAttr>("NK");
    // TODO This is to prevent breaking type system, e.g., if `weight` came from
    // a `mo.mul`. Once we have layout inference across transpose op, we can
    // remove the check.
    if (auto weightDefOp = weight.getDefiningOp();
        weightDefOp && !isa<MO::ConstantOp>(weightDefOp)) {
      return rewriter.notifyMatchFailure(
          op, "can't update layout for non-constant");
    }
    // Let matmul take in the non-transposed weight, and update weight layout

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    // TODO once we have layout inference across transpose op, we can remove
    // this layout annotation.
    auto rebWeight = rewriter.create<MO::RebindOp>(
        op.getLoc(), weightType.cloneWith(preTranspWeightLayout), weight);
    op.setOperand(1, rebWeight);

    return mlir::success();
  }
};

struct BatchMatmulAdjBPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern,
                                           MO::BatchMatmulOp> {
  BatchMatmulAdjBPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::BatchMatmulOp op,
                             MO::DeviceAwareRewriter &rewriter) const override {
    Value input, weight, perm;

    if (isBMMSmall(op))
      return rewriter.notifyMatchFailure(op, "BatchedMatmul is too small");

    auto weightTransP = m_Op<MO::TransposeOp>(rewriter.pushOp(), m_Any(&weight),
                                              m_MOVector(m_Any(&perm)));
    if (!m_Op<MO::BatchMatmulOp>(
             m_Any(&input),
             m_Optional(m_Op<MO::RebindOp>(rewriter.pushOp(), weightTransP),
                        weightTransP))
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

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    // Take the non-transposed weight.
    op.setOperand(1, weight);

    // Mark this matmul as being a tranposed kernel.
    op.setTransposeBAttr(rewriter.getBoolAttr(true));

    return mlir::success();
  }
};

struct ConcatZeroAxisPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern, MO::ConcatOp> {
  ConcatZeroAxisPattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::ConcatOp concat,
                             MO::DeviceAwareRewriter &rewriter) const override {
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

    if (!rewriter.deviceAnnots.isConsistentWith(MO::getHostDeviceAny(concat))) {
      // TODO(#25626): Support outside of host.
      return rewriter.notifyMatchFailure(
          concat, "concat optimizations only supported on host");
    }

    rewriter.replaceOpWithNewOp<MOGG::ConcatOp>(
        concat,
        /*outputTypes=*/concat->getResultTypes(),
        /*inputOperands=*/concat.getInputs());

    return mlir::success();
  }
};

struct ConvConstFilterPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern, MO::ConvOp> {
  ConvConstFilterPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 3) {}

  /// Return an integer attribute that represents the output width of `op`, or
  /// -1 if output width is dynamic.
  static IntegerAttr tryGetOutputWidth(MO::ConvOp op) {
    MO::TensorType resTy = op.getType();
    // NOTE we assume output has NHWC layout here
    if (resTy.hasRank()) {
      if (auto attr =
              dyn_cast<IntegerAttr>(resTy.getTypedShapeAttr().getValues()[2])) {
        return attr;
      }
    }

    return IntegerAttr::get(IndexType::get(op.getContext()), -1);
  }

  LogicalResult
  deviceAwareMatchAndRewrite(MO::ConvOp op,
                             MO::DeviceAwareRewriter &rewriter) const override {
    // Define some meta variables.
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    MO::TensorValue filter = op.getFilter();
    MO::TensorType filterType = filter.getType();
    MO::LayoutAttr filterLayout = filterType.getLayoutAttr();
    if (filterLayout.getFormat() == "FRSCf")
      return rewriter.notifyMatchFailure(op, "filter is already packed");

    if (!dependsOnConstantsOnly<MO::ConstantOp>(filter.getDefiningOp()))
      return rewriter.notifyMatchFailure(op, "filter isn't constant");

    auto maybeNumGroups = MO::tryGetConstantInt(op.getNumGroups());
    if (!maybeNumGroups) {
      return rewriter.notifyMatchFailure(
          op, "currently only pack constant num_groups");
    }

    if (!rewriter.deviceAnnots.isConsistentWith(MO::getHostDeviceAny(op))) {
      // TODO(#25114): Support outside of host.
      return rewriter.notifyMatchFailure(
          op, "packing is only supported on the host device");
    }

    int64_t numGroups = *maybeNumGroups;

    // Create shape parameters for the new type.
    SmallVector<KGEN::ParamDeclAttr> paramDecls;
    auto graphOp = op->getParentOfType<MO::GraphOp>();
    for (StringRef name : {"F", "R", "S", "C", "f"})
      paramDecls.push_back(graphOp.getNewDimParam(name));
    auto paramDeclsAttr = KGEN::ParamDeclArrayAttr::get(ctx, paramDecls);

    // Pass `num_groups` as a parameter to the `pack_conv_filter_shape` and
    // `pack_conv_filter` kernels, and optionally pass static output width to
    // pack_conv_filter_shape.
    //
    // Do so through the `kgenParams` attribute dictionary, which `mogg.call`
    // passes to the KGEN function call as parameters.
    auto numGroupsAttr =
        rewriter.getNamedAttr("num_groups", rewriter.getIndexAttr(numGroups));
    SmallVector<NamedAttribute> namedParams{numGroupsAttr};
    DictionaryAttr kernelKgenParams = rewriter.getDictionaryAttr(namedParams);

    IntegerAttr outWidthAttr = tryGetOutputWidth(op);
    namedParams.push_back(rewriter.getNamedAttr("WO", outWidthAttr));
    DictionaryAttr shapeFuncKgenParams =
        rewriter.getDictionaryAttr(namedParams);

    // Invoke the pack filter fallback shape function.
    // TODO(#18770) Find a better registration mechanism.
    SmallVector<mlir::NamedAttribute> attrs;
    rewriter.create<MO::InvokeShapeFuncOp>(
        loc, SmallVector<Value>{filter}, "pack_conv_filter_shape",
        paramDeclsAttr, shapeFuncKgenParams, ArrayAttr{});

    // Construct type (new shape and layout) for the packed filter.
    SmallVector<TypedAttr> outputParams;
    for (KGEN::ParamDeclAttr outputParamDecl : paramDecls)
      outputParams.push_back(KGEN::ParamDeclRefAttr::get(outputParamDecl));
    auto moshShapeTy = MOSH::ShapeType::get(ctx);
    auto shapeAttr = MOSH::ShapeAttr::get(outputParams, moshShapeTy);
    auto packedFilterLayout = rewriter.getAttr<MO::LayoutAttr>("FRSCf");
    auto packedFilterType =
        filterType.cloneWith(shapeAttr).cloneWith(packedFilterLayout);

    // Pack the filter.
    auto packedFilter = rewriter.create<MO::LayoutTransformOp>(
        loc, /*outputTypes=*/packedFilterType, filter, kernelKgenParams);
    op.setOperand(1, packedFilter);
    return mlir::success();
  }
};

struct SplitZeroAxisPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern, MO::SplitOp> {
  SplitZeroAxisPattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::SplitOp op,
                             MO::DeviceAwareRewriter &rewriter) const override {
    std::optional<int64_t> axis =
        MO::tryGetConstantInt(cast<MO::TensorValue>(op.getAxis()));
    if (!axis.has_value() || axis.value() != 0)
      return rewriter.notifyMatchFailure(op, "axis not constant 0");

    // Ensure input isn't constant
    if (op.getInput().getDefiningOp<MO::ConstantOp>())
      return rewriter.notifyMatchFailure(op, "input is a constant");

    if (!rewriter.deviceAnnots.isConsistentWith(MO::getHostDeviceAny(op))) {
      // TODO(#25626): Support outside of host.
      return rewriter.notifyMatchFailure(
          op, "split optimizations only supported on host");
    }

    rewriter.replaceOpWithNewOp<MOGG::SplitOp>(
        op,
        /*outputTypes=*/op->getResultTypes(),
        /*input=*/op.getInput(),
        /*splitSizes=*/op.getSplitSizes());

    return mlir::success();
  }
};

struct LayerNormPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern, MO::AddOp> {
  LayerNormPattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 6) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::AddOp op,
                             MO::DeviceAwareRewriter &rewriter) const override {
    // Abbreviate the helper to try to keep things readable.
    auto po = [&rewriter]() { return rewriter.pushOp(); };

    // x = input
    // layernorm = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
    Value gamma, beta, eps;
    auto gam = m_Op<MO::Static::BroadcastToOp>(po(), m_Any(&gamma));
    auto bet = m_Op<MO::Static::BroadcastToOp>(po(), m_Any(&beta));
    auto ep = m_Op<MO::Static::BroadcastToOp>(po(), m_Any(&eps));

    auto last_axis = m_Optional(m_ScalarConstant(2), m_ScalarConstant(-1));
    // TODO: Current capture system can't distinguish multiple uses of the same
    // value in the pattern, need to fix, this might be slightly risky
    Value input;
    auto x = m_Ranked(3, m_Any(&input));
    auto mean = m_Op<MO::Static::BroadcastToOp>(
        po(), m_Op<MO::MeanOp>(po(), x, last_axis));
    auto diff = m_Op<MO::SubOp>(po(), x, mean);
    auto var =
        m_Op<MO::MeanOp>(po(), m_Op<MO::MulOp>(po(), diff, diff), last_axis);
    // pytorch specific pattern
    auto p_ratio = m_Op<MO::DivOp>(
        po(), diff,
        m_Op<MO::Static::BroadcastToOp>(
            po(), m_Op<MO::SqrtOp>(
                      po(), m_BinaryCommutativeOp<MO::AddOp>(po(), var, ep))));
    auto p_layernorm = m_BinaryCommutativeOp<MO::AddOp>(
        po(), m_BinaryCommutativeOp<MO::MulOp>(po(), gam, p_ratio), bet);

    // TF specific Pattern
    auto t_rDenom = m_BinaryCommutativeOp<MO::MulOp>(
        po(), gam,
        m_Op<MO::Static::BroadcastToOp>(
            po(), m_Op<MO::RsqrtOp>(
                      po(), m_BinaryCommutativeOp<MO::AddOp>(po(), var, ep))));
    auto t_term1 = m_BinaryCommutativeOp<MO::MulOp>(po(), x, t_rDenom);
    auto t_term2 = m_BinaryCommutativeOp<MO::MulOp>(po(), mean, t_rDenom);
    auto t_term3 = m_Op<MO::SubOp>(po(), bet, t_term2);
    auto t_layernorm = m_BinaryCommutativeOp<MO::AddOp>(po(), t_term1, t_term3);

    if (!t_layernorm.match(op.getResult()) &&
        !p_layernorm.match(op.getResult()))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !beta || !gamma || !eps)
      return rewriter.notifyMatchFailure(op, "null input value");

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

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

    rewriter.replaceOpWithNewOp<MO::LayerNormOp>(
        op,
        /*outputTypes=*/op->getResultTypes(),
        /*inputOperands=*/
        SmallVector<Value>{input, gamma, beta, eps});
    return mlir::success();
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
                 ConcatZeroAxisPattern, ConvConstFilterPattern,
                 SplitZeroAxisPattern, LayerNormPattern>(ctx);

    MO::GraphOp graphOp = getOperation();
    if (failed(applyPatternsAndFoldGreedily(graphOp, std::move(patterns))))
      return signalPassFailure();
    MO::synchronizeMetadataUpdates(graphOp);
  }
};

} // namespace M::MOGG
