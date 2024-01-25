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
using namespace M::MO;
using namespace graph_compiler::matchers;

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
                             MO::TensorType inputType,
                             MO::TensorType resultType, bool transpose) {

  MO::TensorType weightsType = cast<MO::TensorType>(weight.getType());
  MO::TensorType packedWeightsType;

  Type inputEleType = inputType.getElementType();
  Type weightsEleType = weightsType.getElementType();
  Type resultEleType = resultType.getElementType();

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

  // Propagate shape information for the "pack_b" mojo function (#27398).
  SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("a_type", TypeAttr::get(inputEleType)));
  attrs.push_back(rewriter.getNamedAttr(
      "a_shape", rewriter.getI64VectorAttr(inputType.getIntDims())));
  attrs.push_back(
      rewriter.getNamedAttr("b_type", TypeAttr::get(weightsEleType)));
  attrs.push_back(rewriter.getNamedAttr(
      "b_shape", rewriter.getI64VectorAttr(weightsType.getIntDims())));
  attrs.push_back(
      rewriter.getNamedAttr("c_type", TypeAttr::get(resultEleType)));
  attrs.push_back(rewriter.getNamedAttr(
      "c_shape", rewriter.getI64VectorAttr(resultType.getIntDims())));

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
    DeviceRefAttr device = op->getParentOfType<MO::GraphOp>().getDeviceRef();
    // TODO: CUDA does not support pre-packing yet
    if (device && device.getLabel() == M::kCUDALabel)
      return failure();

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

    MO::TensorType inputType = op.getInputA().getType();
    MO::TensorType resultType = op.getType();
    Value packedWeights =
        lowerMatmulPackingFunc(op->getLoc(), rewriter, cache, weight, inputType,
                               resultType, transpose);

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
    int64_t numGroups = *maybeNumGroups;
    auto numGroupsAttr =
        rewriter.getNamedAttr("num_groups", rewriter.getIndexAttr(numGroups));
    SmallVector<NamedAttribute> namedParams{numGroupsAttr};
    DictionaryAttr kernelKgenParams = rewriter.getDictionaryAttr(namedParams);

    // Pass input, filter, and output shape to `pack_conv_filter_shape`.
    MO::TensorValue input = op.getFilter();
    MO::TensorType inputType = input.getType();
    MO::TensorType outputType = op.getType();
    namedParams.push_back(rewriter.getNamedAttr(
        "input_shape", rewriter.getI64VectorAttr(inputType.getIntDims())));
    namedParams.push_back(rewriter.getNamedAttr(
        "filter_shape", rewriter.getI64VectorAttr(filterType.getIntDims())));
    namedParams.push_back(rewriter.getNamedAttr(
        "output_shape", rewriter.getI64VectorAttr(outputType.getIntDims())));

    // Get static or default dynamic values.
    auto getMaybeStaticI64VectorAttr = [&](MO::TensorValue val) {
      if (auto maybeVal = MO::tryGetConstantIntVector(val)) {
        return rewriter.getI64VectorAttr(maybeVal.value());
      } else {
        SmallVector<int64_t> dynamicDims;
        MO::TensorType valType = val.getType();

        if (valType.hasStaticShape())
          dynamicDims.insert(dynamicDims.end(), *valType.getNumElements(),
                             mlir::ShapedType::kDynamic);
        else
          dynamicDims.push_back(mlir::ShapedType::kDynamic);

        return rewriter.getI64VectorAttr(dynamicDims);
      }
    };

    // Pass convolution attributes to `pack_conv_filter_shape`.
    namedParams.push_back(rewriter.getNamedAttr(
        "strides", getMaybeStaticI64VectorAttr(op.getStrides())));
    namedParams.push_back(rewriter.getNamedAttr(
        "dilations", getMaybeStaticI64VectorAttr(op.getDilations())));
    namedParams.push_back(rewriter.getNamedAttr(
        "paddings", getMaybeStaticI64VectorAttr(op.getPaddings())));

    DictionaryAttr shapeFuncKgenParams =
        rewriter.getDictionaryAttr(namedParams);

    // Invoke the pack filter fallback shape function.
    // TODO(#18770) Find a better registration mechanism.
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
        /*input=*/op.getInput());

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

    // TODO(#27337): remove when gpu kernel exists
    if (!rewriter.deviceAnnots.isConsistentWith(MO::getHostDeviceAny(op))) {
      return rewriter.notifyMatchFailure(op,
                                         "Layer norm only implemented on cpu");
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

std::string
getPatternMatchingError(graph_compiler::Logger &patternMatcherLogger) {
  std::string errMsg;
  llvm::raw_string_ostream st(errMsg);
  patternMatcherLogger.print(st);
  return errMsg;
}

ErrorOrSuccess checkRank(TensorValue value, const std::string &valueName,
                         const SmallVector<int64_t> &allowedRanks) {
  if (!value.getType().hasRank()) {
    return Error(valueName + " is unranked");
  }

  int64_t rank = value.getType().getRank();
  for (int64_t allowedRank : allowedRanks) {
    if (allowedRank == rank)
      return success();
  }

  std::string errMsg;
  llvm::raw_string_ostream st(errMsg);
  st << valueName << " has rank " << rank << " but expected one of: ";
  for (int64_t allowedRank : allowedRanks)
    st << allowedRank << " ";
  return Error(errMsg);
}

/// A helper for matching against softmax(input) @ V for fused attention
class MultiHeadAttentionEpilogueMatcher {
private:
  // Bottom of pattern
  TensorValue value; // V   , layout B,H,S,D
  Operation *reprojectProducer = nullptr;

  // Top of the pattern
  TensorValue softmaxInput;
  Operation *softmaxProducer = nullptr;

  graph_compiler::Logger patternMatcherLogger;
  MultiHeadAttentionEpilogueMatcher() = default;

public:
  static ErrorOr<MultiHeadAttentionEpilogueMatcher>
  match(BatchMatmulOp reprojection) {
    MultiHeadAttentionEpilogueMatcher matcher;
    auto pV = m_Any(&matcher.value);

    // softmax
    auto pSoftmaxAttnMatrix =
        m_Op<SoftmaxOp>(&matcher.softmaxProducer, m_Any(&matcher.softmaxInput));

    // re-project using V
    auto pAnswer =
        m_Op<BatchMatmulOp>(&matcher.reprojectProducer, pSoftmaxAttnMatrix, pV);

    bool matches = pAnswer.match(reprojection, &matcher.patternMatcherLogger);
    if (!matches)
      return Error("[MultiHeadAttentionEpilogueMatcher]: " +
                   getPatternMatchingError(matcher.patternMatcherLogger));

    return matcher;
  }
  TensorValue getValue() { return value; }
  TensorValue getSoftMaxInput() { return softmaxInput; }
  TensorValue getTopOfPattern() { return getSoftMaxInput(); }
  ErrorOrSuccess hasExpectedRanks() {
    // Allowed to elide the batch dimension
    ErrorOrSuccess valueOk =
        checkRank(value, "[MultiHeadAttentionEpilogueMatcher] value", {3, 4});
    if (valueOk.isError())
      return valueOk.takeError();

    return success();
  }
};

/// A helper for matching against the initial Q @ KT step of fused attention
class MultiHeadAttentionPrologueMatcher {
private:
  // Should always be filled by match()
  TensorValue query, // Q   , layout B,H,s,D
      keyT;          // K^T , layout B,H,D,S

  graph_compiler::Logger patternMatcherLogger;
  MultiHeadAttentionPrologueMatcher() = default;

public:
  static ErrorOr<MultiHeadAttentionPrologueMatcher>
  match(BatchMatmulOp producingOp) {
    MultiHeadAttentionPrologueMatcher matcher;
    auto pQ = m_Any(&matcher.query);
    auto pKT = m_Any(&matcher.keyT);

    // match Q @ K_T
    Operation *attentionMatrixProducer = nullptr;
    auto pAttentionMatrix =
        m_Op<BatchMatmulOp>(&attentionMatrixProducer, pQ, pKT);

    bool matches = pAttentionMatrix.match(producingOp.getOperation(),
                                          &matcher.patternMatcherLogger);
    if (!matches) {
      return Error("[MultiHeadAttentionPrologueMatcher]: " +
                   getPatternMatchingError(matcher.patternMatcherLogger));
    }

    return matcher;
  }

  TensorValue getQuery() { return query; }
  TensorValue getKeyT() { return keyT; }
  ErrorOrSuccess hasExpectedRanks() {
    // Allowed to elide the batch dimension
    ErrorOrSuccess queryOk =
        checkRank(query, "[MultiHeadAttentionPrologueMatcher] query", {3, 4});
    if (queryOk.isError())
      return queryOk.takeError();

    ErrorOrSuccess keyTOk =
        checkRank(keyT, "[MultiHeadAttentionPrologueMatcher] keyT", {3, 4});
    if (keyTOk.isError())
      return keyTOk.takeError();

    return success();
  }
};

/// A helper to match multi-headed attention and it's variations
class MultiHeadAttentionPatternMatcher {
  /*
  Input values to the pattern.

  Dimension names:
    - (B)atch
    - Attention (H)ead
    - (S)equence Length
    - Embedding (D)imension

  We delineate the current tokens attended to (s) with those in the cache (S)

  Pseudocode:
    # Calculate Attention matrix
    **Step 1:
    attentionMatrix = query @ keyT

    **Step 2:
    norm = broadcast_to(normScalar, shape_of(attentionMatrix))

    # Normalize and apply masking
    # This is optional, else `attentionMatrixNorm = attentionMatrix`
    attentionMatrixNorm = attentionMatrix * norm

    # Masking optional, else `attentionMatrixNormMasked = attentionMatrixNorm`
    attnMaskBCast = broadcast(attnMask, attentionMatrixNorm)
    attentionMatrixNormMasked = attentionMatrixNorm + attnMaskBCast

    **Step 3:
    # Apply softmax and reproject result
    attentionMatrixSoftMax = softmax(attentionMatrixNormMasked)
    answer = attentionMatrixSoftMax @ value
  */
private:
  ErrorOr<MultiHeadAttentionPrologueMatcher> prologueMatcher =
      Error("Note set");
  ErrorOr<MultiHeadAttentionEpilogueMatcher> epilogueMatcher = Error("Not set");

  // Optional Values
  TensorValue attnMask; //     , layout     s,S

  TensorValue normScalar; // the factor for normalizing Q^K_t to be var. 1

  // Intermediate values for checkpointing across separate patterns
  TensorValue attentionMatrix;

  // Intermediate values and their producing operations
  Operation *normProducer = nullptr, *attentionMatrixNormProducer = nullptr;

  // Attention masking-related operations (optional, may remain nullptr)
  Operation *attentionMaskBCastProducer = nullptr,
            *attentionMatrixNormMaskedProducer = nullptr;

  graph_compiler::Logger patternMatcherLogger;

  ErrorOrSuccess intermediateValuesHaveOneUse() {
    // Checks all intermediate values are used once. The exception are
    // `norm`, `attnMask`, `attentionMaskBCastProducer` since their values are
    // typically reused per attention layer.

    if (attentionMatrixNormProducer &&
        !attentionMatrixNormProducer->hasOneUse())
      return Error("attentionMatrixNorm has multiple uses.");

    // Remember this is also optional
    if (attentionMatrixNormMaskedProducer &&
        !attentionMatrixNormMaskedProducer->hasOneUse())
      return Error("attentionMatrixNormMasked has multiple uses.");

    return success();
  }

  ErrorOrSuccess hasExpectedRanks() {
    ErrorOrSuccess prologueOK = prologueMatcher->hasExpectedRanks();
    if (prologueOK.isError())
      return prologueOK.takeError();

    ErrorOrSuccess epilogueOk = epilogueMatcher->hasExpectedRanks();
    if (epilogueOk.isError())
      return epilogueOk.takeError();

    if (attnMask) {
      // May have a pre-broadcasted batch-dimension
      ErrorOrSuccess attnMaskOk = checkRank(attnMask, "attnMask", {2, 3});
      if (attnMaskOk.isError())
        return attnMaskOk.takeError();
    }

    if (normScalar) {
      // May have a pre-broadcasted batch-dimension
      ErrorOrSuccess normScalarOk = checkRank(normScalar, "normScalar", {0, 1});
      if (normScalarOk.isError())
        return normScalarOk.takeError();
    }

    return success();
  }
  MultiHeadAttentionPatternMatcher() = default;

public:
  static ErrorOr<MultiHeadAttentionPatternMatcher>
  match(BatchMatmulOp finalOp) {
    MultiHeadAttentionPatternMatcher matcher;

    // Note we match things in a reverse-ward direction as we are given the
    // very final operation. We then bubble our way up.
    // **Step 3:
    matcher.epilogueMatcher = MultiHeadAttentionEpilogueMatcher::match(finalOp);
    if (matcher.epilogueMatcher.isError())
      return matcher.epilogueMatcher.takeError();
    TensorValue epilogueTopOfPattern =
        matcher.epilogueMatcher->getTopOfPattern();
    if (!epilogueTopOfPattern.hasOneUse())
      return Error("Softmax input producer has more than one use.");

    // **Step 2:
    auto pAttentionMatrix = m_Any(&matcher.attentionMatrix);

    // TODO: normScalar is derived in some scenarios, it is usually 1 / sqrt(D)
    auto pNorm = m_Op<Static::BroadcastToOp>(&matcher.normProducer,
                                             m_Any(&matcher.normScalar));

    // normalize by scale
    auto pNormalizedAttnMatrix = m_BinaryCommutativeOp<MulOp>(
        &matcher.attentionMatrixNormProducer, pAttentionMatrix, pNorm);

    // Normalization is optional
    auto pMaybeNormalizedAttnMatrix =
        m_Optional(pNormalizedAttnMatrix, pAttentionMatrix);

    // attention mask (optional)
    auto pAttentionMaskBCast = m_Op<Static::BroadcastToOp>(
        &matcher.attentionMaskBCastProducer, m_Any(&matcher.attnMask));
    auto pMaskedNormalizedAttnMatrix = m_BinaryCommutativeOp<AddOp>(
        &matcher.attentionMatrixNormMaskedProducer, pMaybeNormalizedAttnMatrix,
        pAttentionMaskBCast);

    // handle with and without attention mask via m_Optional
    auto pCenterPattern =
        m_Optional(pMaskedNormalizedAttnMatrix, pMaybeNormalizedAttnMatrix);

    // attempt to match the pattern
    bool matchesPattern = pCenterPattern.match(epilogueTopOfPattern,
                                               &matcher.patternMatcherLogger);
    if (!matchesPattern)
      return Error(getPatternMatchingError(matcher.patternMatcherLogger));

    // **Step 1: Match the prologue
    BatchMatmulOp attentionMatrixProducer =
        matcher.attentionMatrix.getDefiningOp<BatchMatmulOp>();
    if (!attentionMatrixProducer)
      return Error("AttentionMatrix has no BatchMatmulOp Producer!");
    if (!attentionMatrixProducer->hasOneUse())
      return Error("AttentionMatrix has more than one use!");

    matcher.prologueMatcher =
        MultiHeadAttentionPrologueMatcher::match(attentionMatrixProducer);
    if (matcher.prologueMatcher.isError())
      return matcher.prologueMatcher.takeError();

    ErrorOrSuccess hasOneUse = matcher.intermediateValuesHaveOneUse();
    if (hasOneUse.isError())
      return hasOneUse.takeError();

    ErrorOrSuccess hasExpectedRanks = matcher.hasExpectedRanks();
    if (hasExpectedRanks.isError())
      return hasExpectedRanks.takeError();

    return matcher;
  }

  bool hasAttentionMask() { return attnMask != nullptr; }

  ErrorOr<TensorValue>
  extractCompatibleAttentionMask(PatternRewriter &rewriter) {
    // Our kernels expect a mask of shape [s, S]. So try to extract this
    if (!hasAttentionMask()) {
      return Error("No attention mask in this pattern");
    }

    TensorValue fixedUpMask = attnMask;
    if (!fixedUpMask.getType().hasRank())
      return Error("Attention mask needs a known rank.");

    // Target shape for the attention mask:
    TensorValue query = getQuery();
    TensorValue keyT = getKeyT();
    SmallVector<TypedAttr> bcastAttnMaskDims = {
        query.getType().getDim(query.getType().getRank() - 2),
        keyT.getType().getDim(query.getType().getRank() - 1)};
    ShapeAttr attnMaskTargetShapeAttr = ShapeAttr::get(
        bcastAttnMaskDims, MOSH::ShapeType::get(rewriter.getContext()));

    M::MO::TensorType attnMaskTargetShapeType = M::MO::TensorType::get(
        attnMaskTargetShapeAttr, fixedUpMask.getType().getElementType());

    if (fixedUpMask.getType().getRank() > 2) {
      // Handles case where mask is of shape like [1, s, S].
      if (attnMaskTargetShapeAttr.getSymbolicNumElements() !=
          fixedUpMask.getType().getSymbolicNumElements())
        return Error("Given mask cannot be reshaped to [s, S] shaped mask!");
      fixedUpMask = rewriter.create<Static::ReshapeOp>(
          fixedUpMask.getLoc(), attnMaskTargetShapeType, fixedUpMask);
    }

    // Manually broadcast out the attention mask to guarantee layout of s,S
    // Handles case where mask is just a scalar which is equally applied.
    fixedUpMask = rewriter.create<Static::BroadcastToOp>(
        fixedUpMask.getLoc(), attnMaskTargetShapeType, fixedUpMask);

    return fixedUpMask;
  }

  // layout B,H,s,D
  TensorValue getQuery() { return prologueMatcher->getQuery(); }

  // layout B,H,D,S
  TensorValue getKeyT() { return prologueMatcher->getKeyT(); }

  // layout B,H,s,D
  TensorValue getValue() { return epilogueMatcher->getValue(); }
  TensorValue getNormScalar(OpBuilder &builder) {
    if (normScalar)
      return normScalar;

    // Kernel still expects normalization, just set it to 1 to make it a no-op
    TensorValue one = builder.create<ConstantOp>(getQuery().getLoc(), 1.0);
    return builder.create<CastOp>(
        getQuery().getLoc(),
        one.getType().cloneWith(getQuery().getType().getDtype()), one);
  }
};

// For MIN Llama 2 demo
struct MultiHeadAttentionPatternCPU : public OpRewritePattern<BatchMatmulOp> {
  MultiHeadAttentionPatternCPU(MLIRContext *ctx) : OpRewritePattern(ctx, 20) {}

  LogicalResult matchAndRewrite(BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
    DeviceRefAttr device = op->getParentOfType<MO::GraphOp>().getDeviceRef();

    // Just assume we want to use the CPU kernel only if the default device
    // is cpu (e.g. if the default was CUDA, run on CUDA).
    //
    // That way avoid device logic for now.
    if (device && device.getLabel() != M::kCPULabel)
      return rewriter.notifyMatchFailure(op->getLoc(), "Devices are wrong.");

    ErrorOr<MultiHeadAttentionPatternMatcher> maybeMatcher =
        MultiHeadAttentionPatternMatcher::match(op);
    if (maybeMatcher.isError())
      return rewriter.notifyMatchFailure(op.getLoc(), maybeMatcher.getError());

    MultiHeadAttentionPatternMatcher matcher = maybeMatcher.takeValue();

    // Attention masking case
    if (matcher.hasAttentionMask()) {
      ErrorOr<TensorValue> maybeAttentionMask =
          matcher.extractCompatibleAttentionMask(rewriter);

      if (maybeAttentionMask.isError())
        return rewriter.notifyMatchFailure(op->getLoc(),
                                           maybeAttentionMask.getError());

      SmallVector<Value> inputValues = {
          matcher.getQuery(), matcher.getKeyT(), matcher.getValue(),
          maybeAttentionMask.takeValue(), matcher.getNormScalar(rewriter)};
      SmallVector<Type> outputTypes = {op.getType()};
      ValueRange inputs(inputValues);
      TypeRange outputTy(outputTypes);
      rewriter.setInsertionPointAfterValue(op);
      auto result = rewriter.create<CustomOp>(op->getLoc(), outputTy, inputs,
                                              "with_mask_fused_attention_cpu");
      auto fused_kernel_result = cast<TensorValue>(result->getResult(0));
      fused_kernel_result.setType(op.getType());
      rewriter.replaceAllUsesWith(op, fused_kernel_result);
    } else {
      // No attention masking case
      SmallVector<Value> inputValues = {matcher.getQuery(), matcher.getKeyT(),
                                        matcher.getValue(),
                                        matcher.getNormScalar(rewriter)};
      SmallVector<Type> outputTypes = {op.getType()};
      ValueRange inputs(inputValues);
      TypeRange outputTy(outputTypes);
      rewriter.setInsertionPointAfterValue(op);
      auto result = rewriter.create<CustomOp>(op->getLoc(), outputTy, inputs,
                                              "no_mask_fused_attention_cpu");

      auto fused_kernel_result = cast<TensorValue>(result->getResult(0));
      fused_kernel_result.setType(op.getType());
      rewriter.replaceAllUsesWith(op, fused_kernel_result);
    }

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
                 SplitZeroAxisPattern, LayerNormPattern,
                 MultiHeadAttentionPatternCPU>(ctx);

    MO::GraphOp graphOp = getOperation();
    if (failed(applyPatternsAndFoldGreedily(graphOp, std::move(patterns))))
      return signalPassFailure();
    MO::synchronizeMetadataUpdates(graphOp);
  }
};

} // namespace M::MOGG
