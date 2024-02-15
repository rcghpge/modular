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

    if (llvm::is_contained({"FQRSCf", "FRSCf"}, filterLayout.getFormat()))
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

    size_t filterRank = filterType.getRank();

    if (filterRank != 4 && filterRank != 5)
      return rewriter.notifyMatchFailure(
          op, "Conv filter packing is currently only supported "
              "for 2D and 3D filters");

    StringRef packedLayoutStr = [filterRank]() {
      switch (filterRank) {
      case 4:
        return "FRSCf";
      case 5:
        return "FQRSCf";
      default:
        llvm_unreachable(
            "Only 2D and 3D convolutions are supported for packing");
      }
    }();

    const char *layoutStrBegin = packedLayoutStr.begin();
    for (size_t i = 0, e = packedLayoutStr.size(); i < e; i++) {
      paramDecls.push_back(
          graphOp.getNewDimParam(StringRef(layoutStrBegin + i, 1)));
    }

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
    auto packedFilterLayout = rewriter.getAttr<MO::LayoutAttr>(packedLayoutStr);
    auto packedFilterType =
        filterType.cloneWith(shapeAttr).cloneWith(packedFilterLayout);

    // Pack the filter.
    auto packedFilter = rewriter.create<MO::LayoutTransformOp>(
        loc, /*outputTypes=*/packedFilterType, filter, kernelKgenParams);
    op.setOperand(1, packedFilter);
    return mlir::success();
  }
};

struct ConvTransposeConstFilterPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern,
                                           MO::ConvTransposeOp> {
  ConvTransposeConstFilterPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 3) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::ConvTransposeOp op,
                             MO::DeviceAwareRewriter &rewriter) const override {
    // Define some meta variables.
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    MO::TensorValue filter = op.getFilter();
    MO::TensorType filterType = filter.getType();
    MO::LayoutAttr filterLayout = filterType.getLayoutAttr();

    if (llvm::is_contained({"FQRSCf", "FRSCf"}, filterLayout.getFormat()))
      return rewriter.notifyMatchFailure(op, "filter is already packed");

    if (!dependsOnConstantsOnly<MO::ConstantOp>(filter.getDefiningOp()))
      return rewriter.notifyMatchFailure(op, "filter isn't constant");

    if (!rewriter.deviceAnnots.isConsistentWith(MO::getHostDeviceAny(op))) {
      // TODO(#25114): Support outside of host.
      return rewriter.notifyMatchFailure(
          op, "packing is only supported on the host device");
    }

    // Create shape parameters for the new type.
    SmallVector<KGEN::ParamDeclAttr> paramDecls;
    auto graphOp = op->getParentOfType<MO::GraphOp>();

    size_t filterRank = filterType.getRank();

    if (filterRank != 4 && filterRank != 5)
      return rewriter.notifyMatchFailure(
          op, "ConvTranspose filter packing is currently only supported "
              "for 2D and 3D filters");

    StringRef packedLayoutStr = [filterRank]() {
      switch (filterRank) {
      case 4:
        return "FRSCf";
      case 5:
        return "FQRSCf";
      default:
        llvm_unreachable(
            "Only 2D and 3D conv transposes are supported for packing");
      }
    }();

    const char *layoutStrBegin = packedLayoutStr.begin();
    for (size_t i = 0, e = packedLayoutStr.size(); i < e; i++) {
      paramDecls.push_back(
          graphOp.getNewDimParam(StringRef(layoutStrBegin + i, 1)));
    }

    auto paramDeclsAttr = KGEN::ParamDeclArrayAttr::get(ctx, paramDecls);

    // Invoke the pack filter fallback shape function.
    // TODO(#18770) Find a better registration mechanism.
    rewriter.create<MO::InvokeShapeFuncOp>(
        loc, SmallVector<Value>{filter}, "pack_conv_transpose_filter_shape",
        paramDeclsAttr, mlir::DictionaryAttr{}, mlir::ArrayAttr{});

    // Construct type (new shape and layout) for the packed filter.
    SmallVector<TypedAttr> outputParams;
    for (KGEN::ParamDeclAttr outputParamDecl : paramDecls)
      outputParams.push_back(KGEN::ParamDeclRefAttr::get(outputParamDecl));
    auto moshShapeTy = MOSH::ShapeType::get(ctx);
    auto shapeAttr = MOSH::ShapeAttr::get(outputParams, moshShapeTy);
    auto packedFilterLayout = rewriter.getAttr<MO::LayoutAttr>(packedLayoutStr);
    auto packedFilterType =
        filterType.cloneWith(shapeAttr).cloneWith(packedFilterLayout);

    // Pack the filter.
    auto packedFilter = rewriter.create<MO::LayoutTransformOp>(
        loc, /*outputTypes=*/packedFilterType, filter);
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

template <typename pInT>
auto inline pOptionalBroadcast(Operation **bindDest, pInT patternIn) {
  // Allows optional insertion of reshapes and bcasts between this pattern
  // Specifically patternIn --> (optional) mo.static.reshape
  //  --> (optional) mo.static.broadcast
  auto bcast = m_Op<Static::BroadcastToOp>(bindDest, patternIn);
  return m_Optional(bcast, patternIn);
}

template <typename pInT>
auto inline pOptionalReshape(Operation **bindDest, pInT patternIn) {
  // Allows optional insertion of reshapes and bcasts between this pattern
  // Specifically patternIn --> (optional) mo.static.reshape
  //  --> (optional) mo.static.broadcast
  auto reshape = m_Op<Static::ReshapeOp>(bindDest, patternIn);
  return m_Optional(reshape, patternIn);
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

  // For unsafe broadcasts
  Operation *optBCast1 = nullptr;
  Operation *optBCast2 = nullptr;

  // For internal reshapes
  Operation *optReshape1 = nullptr;
  Operation *optReshape2 = nullptr;

  graph_compiler::Logger patternMatcherLogger;
  MultiHeadAttentionEpilogueMatcher() = default;

public:
  static ErrorOr<MultiHeadAttentionEpilogueMatcher>
  match(BatchMatmulOp reprojection, bool allowUnsafeBCast,
        bool allowInternalReshape, bool debugMHAPattern) {
    MultiHeadAttentionEpilogueMatcher matcher;
    auto pV = m_Any(&matcher.value);

    // optional broadcast + softmax
    auto maybeReshape1 =
        pOptionalReshape(&matcher.optReshape1, m_Any(&matcher.softmaxInput));
    auto maybeBCast1 = pOptionalBroadcast(&matcher.optBCast1, maybeReshape1);
    auto pSoftmaxAttnMatrix =
        m_Op<SoftmaxOp>(&matcher.softmaxProducer, maybeBCast1);

    // optional broadcast + re-project using V
    auto maybeReshape2 =
        pOptionalReshape(&matcher.optReshape2, pSoftmaxAttnMatrix);
    auto maybeBCast2 = pOptionalBroadcast(&matcher.optBCast2, maybeReshape2);
    auto pAnswer =
        m_Op<BatchMatmulOp>(&matcher.reprojectProducer, maybeBCast2, pV);

    bool matches = pAnswer.match(reprojection,
                                 debugMHAPattern ? &matcher.patternMatcherLogger
                                                 : nullptr);
    if (!matches)
      return Error("[MultiHeadAttentionEpilogueMatcher]: " +
                   getPatternMatchingError(matcher.patternMatcherLogger));

    if (!allowUnsafeBCast && (matcher.optBCast1 || matcher.optBCast2))
      return Error("[MultiHeadAttentionEpilogueMatcher]: pattern matches but "
                   "has unsafe broadcasts.");

    if (!allowInternalReshape && (matcher.optReshape1 || matcher.optReshape2))
      return Error("[MultiHeadAttentionEpilogueMatcher]: pattern matches but "
                   "has internal reshapes.");

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
  void fillUnsafeBCasts(SmallVector<Static::BroadcastToOp> &unsafeBcasts) {
    if (optBCast1)
      unsafeBcasts.push_back(cast<Static::BroadcastToOp>(optBCast1));
    if (optBCast2)
      unsafeBcasts.push_back(cast<Static::BroadcastToOp>(optBCast2));
  }
};

/// A helper for matching against attention masking for fused attention
/// Supports masking through select statements. This is usually done for
/// implementing causal masking of the attentionMatrix.
///
/// %cond : ...
/// %true : ...
/// %false : ... all same shape
/// %result = mo.select(%cond, %true, %false)
///
/// Where "cond" is usually a triangular matrix whose lower part and main
/// diagonal are "false" and the upper part is "true". The %true branch
/// is usually a scalar -inf broadcasted and %false is the attentionMatrix.
///
/// Essentially this sets the upper triangle of attentionMatrix to "-inf".
/// Note, %cond and true/false can be switched because why not?
///
/// Notes on masking:
/// https://www.notion.so/modularai/Multi-Head-Attention-bbc2d300937a495da40c18b6a2028d22
class MultiHeadAttentionSelectMaskingMatcher {
  friend class MultiHeadAttentionMaskingMuxMatcher;

  // Thresholds for what is considered "-inf" for different datatypes
  // ~ -3.40282347E+38 seems popular, -1E+38 is chosen since its close to
  // the smallest normal floating point number (as opposed to subnormal).
  const double FLOAT32_NEGINF_THRESHOLD = -1E+38;

private:
  TensorValue patternInput;
  TensorValue scalarMask;
  TensorValue conditionalStatement;

  // We assume scalarMasked is immediately broadcasted
  Operation *scalarMaskBCast = nullptr;

  // Whether the attention matrix masked is in the "true" or "false" branch
  // of the config.
  bool attentionMatrixTrueBranch;

  // For unsafe broadcasts of the attentionMatrix branch
  Operation *optBCast1 = nullptr;
  Operation *optBCast2 = nullptr;

  // For internal reshapes
  Operation *optReshape1 = nullptr;

  graph_compiler::Logger patternMatcherLogger;

  MultiHeadAttentionSelectMaskingMatcher() = default;

  ErrorOrSuccess verifyScalarNegInf() {
    // Returns whether the detected scalar is -inf or something close
    // NOTE: in the future consider different thresholds for different DTypes
    if (!scalarMask)
      return Error("scalar mask is nullptr!");

    auto maybeFloat = tryGetConstantOfRank<double>(scalarMask, 0);
    if (!maybeFloat)
      return Error("could not extract scalar float!");

    double value = (*maybeFloat)[0];
    if (value > FLOAT32_NEGINF_THRESHOLD)
      return Error("scalar detected but not small enough");

    return success();
  }

  static ErrorOr<MultiHeadAttentionSelectMaskingMatcher>
  match(Operation *producingOp, bool allowUnsafeBCast,
        bool allowInternalReshape, bool debugMHAPattern) {
    MultiHeadAttentionSelectMaskingMatcher matcher;

    // Input to the pattern -- the "attentionMask"
    auto pPatternInputBase = m_Any(&matcher.patternInput);
    auto pPatternInputRank3 = m_Ranked(3, pPatternInputBase);
    auto pPatternInputRank4 = m_Ranked(4, pPatternInputBase);
    auto pPatternInput = m_Optional(pPatternInputRank3, pPatternInputRank4);

    // One more bcast for starcoder 2
    auto pInitialMaybeBCast =
        pOptionalBroadcast(&matcher.optBCast1, pPatternInput);
    auto pPatternMaybeReshape =
        pOptionalReshape(&matcher.optReshape1, pInitialMaybeBCast);
    auto pPatternMaybeBCast =
        pOptionalBroadcast(&matcher.optBCast2, pPatternMaybeReshape);

    // Scalar and conditional for select
    auto pBCastScalar = m_Op<Static::BroadcastToOp>(
        &matcher.scalarMaskBCast, m_Ranked(0, m_Any(&matcher.scalarMask)));
    auto pConditional = m_Any(&matcher.conditionalStatement);

    // consider the case where attentionMask is in the "false branch"
    auto pAttentionMaskAsFalse =
        m_Op<SelectOp>(pConditional, pBCastScalar, pPatternMaybeBCast);
    bool matchesNegBranch = pAttentionMaskAsFalse.match(
        producingOp, debugMHAPattern ? &matcher.patternMatcherLogger : nullptr);
    if (matchesNegBranch && !matcher.verifyScalarNegInf().isError() &&
        (allowUnsafeBCast || (!matcher.optBCast1 && !matcher.optBCast2)) &&
        (allowInternalReshape || !matcher.optReshape1)) {
      matcher.attentionMatrixTrueBranch = false;
      return matcher;
    }

    // case where attentionMask is in the "true branch"
    auto pAttentionMaskAsTrue =
        m_Op<SelectOp>(pConditional, pPatternMaybeBCast, pBCastScalar);
    bool matchesPosBranch = pAttentionMaskAsTrue.match(
        producingOp, debugMHAPattern ? &matcher.patternMatcherLogger : nullptr);
    if (matchesPosBranch && !matcher.verifyScalarNegInf().isError() &&
        (allowUnsafeBCast || (!matcher.optBCast1 && !matcher.optBCast2)) &&
        (allowInternalReshape || !matcher.optReshape1)) {
      matcher.attentionMatrixTrueBranch = true;
      return matcher;
    }

    return Error(
        "[MultiHeadAttentionSelectMaskingMatcher] No matches detected!");
  }

public:
  TensorValue getTopOfPattern() { return patternInput; }

  TypedAttr getAttentionMaskShape() {
    return conditionalStatement.getType().getShapeAttr();
  }

  TensorValue getAttentionMask(mlir::OpBuilder &builder) {
    // NOTE: in general `select(%cond, %true, %false)` is:
    //   - %cond * %true + not(%cond) * %false
    //
    // But we've checked the scalar is -inf so will assume dominate any value in
    // the addition mask. Note it is expected this masked result will be
    // soft-maxed. For example, assume `attentionMatrixTrueBranch` so the mask
    // will occur in %false. Then the following is equivalent:
    //
    // select(%cond, %true, %false) ~= [not(%cond) * %false] + %true
    //
    // This is nice because it is essentially explicit masking with the mask
    // in brackets.
    TensorValue bcastedMask = cast<TensorValue>(scalarMaskBCast->getResult(0));
    Location loc = bcastedMask.getLoc();

    if (attentionMatrixTrueBranch) {
      TensorValue notCond = builder.create<NotOp>(loc, conditionalStatement);
      TensorValue notCondCasted = builder.create<CastOp>(
          loc, notCond, bcastedMask.getType().getDtype());
      return builder.create<MulOp>(loc, bcastedMask, notCondCasted);
    } else {
      TensorValue condCasted = builder.create<CastOp>(
          loc, conditionalStatement, bcastedMask.getType().getDtype());
      return builder.create<MulOp>(loc, bcastedMask, condCasted);
    }
  }

  ErrorOrSuccess hasExpectedRanks() {
    // NOTE: select requires all inputs to be the same shape
    // Allowed to elide the batch dimension
    ErrorOrSuccess rankOk = checkRank(
        conditionalStatement,
        "[MultiHeadAttentionSelectMaskingMatcher] select inputs", {3, 4});
    if (rankOk.isError())
      return rankOk.takeError();
    return success();
  }

  void fillUnsafeBCasts(SmallVector<Static::BroadcastToOp> &unsafeBcasts) {
    if (optBCast1)
      unsafeBcasts.push_back(cast<Static::BroadcastToOp>(optBCast1));
    if (optBCast2)
      unsafeBcasts.push_back(cast<Static::BroadcastToOp>(optBCast2));
  }
};

/// A helper for matching against attention masking for fused attention
/// Supports explicit masking:
///
/// Explicit masking occurs when we add a tensor directly to our attention
/// matrix
///
/// %attentionMatrix
/// %attentionMask
/// %result = mo.add(%attentionMatrix, %attentionMask)
///
/// Where usually some elements of %attentionMask will be -inf
///
/// Notes on masking:
/// https://www.notion.so/modularai/Multi-Head-Attention-bbc2d300937a495da40c18b6a2028d22
class MultiHeadAttentionExplicitMaskingMatcher {
  friend class MultiHeadAttentionMaskingMuxMatcher;

private:
  Operation *attentionMatrixNormMaskedProducer = nullptr;

  TensorValue patternInput;
  TensorValue attentionMask;

  // For unsafe broadcasts
  Operation *optBCast1 = nullptr;

  // For internal reshapes
  Operation *optReshape1 = nullptr;

  graph_compiler::Logger patternMatcherLogger;

  MultiHeadAttentionExplicitMaskingMatcher() = default;

  static ErrorOr<MultiHeadAttentionExplicitMaskingMatcher>
  match(Operation *producingOp, bool allowUnsafeBCast,
        bool allowInternalReshape, bool debugMHAPattern) {
    MultiHeadAttentionExplicitMaskingMatcher matcher;

    auto pPatternInput = m_Any(&matcher.patternInput);

    // Optional Broadcast into the add
    auto maybeReshape1 = pOptionalReshape(&matcher.optReshape1, pPatternInput);
    auto maybeBCast1 = pOptionalBroadcast(&matcher.optBCast1, maybeReshape1);
    auto pMaskedNormalizedAttnMatrix = m_BinaryCommutativeOp<AddOp>(
        &matcher.attentionMatrixNormMaskedProducer, maybeBCast1,
        m_Any(&matcher.attentionMask));

    if (!producingOp)
      return Error("[MultiHeadAttentionExplicitMaskingMatcher] producing "
                   "operation is nullptr.");

    if (producingOp->getNumResults() != 1)
      return Error(
          "[MultiHeadAttentionExplicitMaskingMatcher] producing operation "
          "does not have one result.");

    TensorValue producingOpResult =
        cast<TensorValue>(producingOp->getResult(0));

    // attempt to match the pattern
    bool matchesPattern = pMaskedNormalizedAttnMatrix.match(
        producingOpResult,
        debugMHAPattern ? &matcher.patternMatcherLogger : nullptr);
    if (!matchesPattern)
      return Error("Attention Masking Component Does Not Match: " +
                   getPatternMatchingError(matcher.patternMatcherLogger));

    if (!allowUnsafeBCast && matcher.optBCast1)
      return Error(
          "[MultiHeadAttentionExplicitMaskingMatcher]: pattern matches but "
          "has unsafe broadcasts.");
    if (!allowInternalReshape && matcher.optReshape1)
      return Error(
          "[MultiHeadAttentionExplicitMaskingMatcher]: pattern matches but "
          "has internal reshapes.");

    return matcher;
  }

public:
  TypedAttr getAttentionMaskShape() {
    return attentionMask.getType().getShapeAttr();
  }
  TensorValue getTopOfPattern() { return patternInput; }
  TensorValue getAttentionMask() { return attentionMask; }

  ErrorOrSuccess hasExpectedRanks() {
    // Allowed to elide the batch dimension
    ErrorOrSuccess attnMaskOk = checkRank(
        attentionMask, "[MultiHeadAttentionExplicitMaskingMatcher] attnMask",
        {3, 4});
    if (attnMaskOk.isError())
      return attnMaskOk.takeError();

    ErrorOrSuccess patternInputOk = checkRank(
        attentionMask,
        "[MultiHeadAttentionExplicitMaskingMatcher] patternInput", {3, 4});
    if (patternInputOk.isError())
      return patternInputOk.takeError();

    return success();
  }

  void fillUnsafeBCasts(SmallVector<Static::BroadcastToOp> &unsafeBcasts) {
    if (optBCast1)
      unsafeBcasts.push_back(cast<Static::BroadcastToOp>(optBCast1));
  }
};

/// MHA Masking multiplexer. In practice models are allowed to have both
/// explicit and causal masking and some models have no masking at all!
///
/// What more, models can vary the order of explicit and causal masking freely.
///
/// So in total, there are 5 variations of masking in transformer models which
/// seem common:
///
/// - No masking at all
/// - Only causal masking
/// - Only explicit masking
/// - Causal and explicit masking
///    - Order we apply them maybe different
/// This is designed to handle all of these cases and return a single compatible
/// "attention mask" which combines the details of every masking pattern.
class MultiHeadAttentionMaskingMuxMatcher {
private:
  TensorValue patternInput;

  // Each determine whether the particular pattern was detected
  ErrorOr<MultiHeadAttentionSelectMaskingMatcher> causalMask =
      Error("Not initialized!");
  ErrorOr<MultiHeadAttentionExplicitMaskingMatcher> explicitMask =
      Error("Not initialized!");
  MultiHeadAttentionMaskingMuxMatcher() = default;

public:
  static ErrorOr<MultiHeadAttentionMaskingMuxMatcher>
  match(Operation *producingOp, bool allowUnsafeBCast,
        bool allowInternalReshape, bool debugMHAPattern) {
    MultiHeadAttentionMaskingMuxMatcher matcher;

    // See if any masking patterns match
    Operation *curProducingOp = producingOp;
    matcher.causalMask = MultiHeadAttentionSelectMaskingMatcher::match(
        curProducingOp, allowUnsafeBCast, allowInternalReshape,
        debugMHAPattern);
    matcher.explicitMask = MultiHeadAttentionExplicitMaskingMatcher::match(
        curProducingOp, allowUnsafeBCast, allowInternalReshape,
        debugMHAPattern);

    GENERICML_ASSERT(matcher.causalMask.isError() ||
                     matcher.explicitMask.isError())
        << "[MultiHeadAttentionMaskingMuxMatcher] expect only a single mask "
           "pattern to initial match.";

    // no masking at all!
    if (matcher.causalMask.isError() && matcher.explicitMask.isError()) {
      if (producingOp->getNumResults() != 1)
        return Error("[MultiHeadAttentionMaskingMuxMatcher] tried to match "
                     "against something with more than one result.");

      matcher.patternInput = cast<TensorValue>(producingOp->getResult(0));
      return matcher;
    }

    // There still could be the other type of masking left!
    if (!matcher.causalMask.isError()) {
      matcher.patternInput = matcher.causalMask->getTopOfPattern();
      matcher.explicitMask = MultiHeadAttentionExplicitMaskingMatcher::match(
          matcher.patternInput.getDefiningOp(), allowUnsafeBCast,
          allowInternalReshape, debugMHAPattern);
      if (!matcher.explicitMask.isError())
        matcher.patternInput = matcher.explicitMask->getTopOfPattern();
    } else { // explicit mask is filled
      matcher.patternInput = matcher.explicitMask->getTopOfPattern();
      matcher.causalMask = MultiHeadAttentionSelectMaskingMatcher::match(
          matcher.patternInput.getDefiningOp(), allowUnsafeBCast,
          allowInternalReshape, debugMHAPattern);
      if (!matcher.causalMask.isError())
        matcher.patternInput = matcher.causalMask->getTopOfPattern();
    }

    // the two masks however may be different shapes (they most def. are not)
    // but ignoring this is only allowed if `allowUnsafeBCast`
    if (!matcher.causalMask.isError() && !matcher.explicitMask.isError()) {
      bool shapesMatch = matcher.causalMask->getAttentionMaskShape() ==
                         matcher.explicitMask->getAttentionMaskShape();
      if (!shapesMatch && !allowUnsafeBCast)
        return Error("[MultiHeadAttentionMaskingMuxMatcher] Both explicit and "
                     "causal masking found but their shapes are not the same!");
    }

    return matcher;
  }
  bool hasAttentionMask() {
    return !causalMask.isError() || !explicitMask.isError();
  }

  TensorValue getTopOfPattern() { return patternInput; }
  TensorValue getAttentionMask(OpBuilder &builder) {
    SmallVector<TensorValue> masks;
    if (!explicitMask.isError())
      masks.push_back(explicitMask->getAttentionMask());
    if (!causalMask.isError())
      masks.push_back(causalMask->getAttentionMask(builder));

    if (masks.empty())
      return {};

    if (masks.size() == 1)
      return masks[0];

    TensorValue mask = masks[0];
    for (size_t i = 1; i < masks.size(); i++) {
      // statically cast incoming mask to the current shape
      // note this is always assumed to be safe because of `allowUnsafeBCast`
      TensorValue otherMask = masks[i];
      otherMask = builder.create<BroadcastToOp>(mask.getLoc(), otherMask, mask);

      mask = builder.create<AddOp>(mask.getLoc(), mask, otherMask);
    }
    return mask;
  }

  ErrorOrSuccess hasExpectedRanks() {
    if (!causalMask.isError()) {
      ErrorOrSuccess causalMaskOk = causalMask->hasExpectedRanks();
      if (causalMaskOk.isError())
        return causalMaskOk.takeError();
    }

    if (!explicitMask.isError()) {
      ErrorOrSuccess explicitMaskOk = explicitMask->hasExpectedRanks();
      if (explicitMaskOk.isError())
        return explicitMaskOk.takeError();
    }

    if (!causalMask.isError() && !explicitMask.isError()) {
      MOSH::ShapeAttr causalMaskShape =
          cast<MOSH::ShapeAttr>(causalMask->getAttentionMaskShape());
      MOSH::ShapeAttr explicitMaskShape =
          cast<MOSH::ShapeAttr>(explicitMask->getAttentionMaskShape());

      if (causalMaskShape.getRank() != explicitMaskShape.getRank())
        return Error("Attention masks are not the same rank!");
    }
    return success();
  }

  TypedAttr getAttentionMaskShape() {
    if (!causalMask.isError())
      return causalMask->getAttentionMaskShape();
    if (!explicitMask.isError())
      return explicitMask->getAttentionMaskShape();
    return {};
  }

  void fillUnsafeBCasts(SmallVector<Static::BroadcastToOp> &unsafeBcasts) {
    if (!causalMask.isError())
      causalMask->fillUnsafeBCasts(unsafeBcasts);
    if (!explicitMask.isError())
      explicitMask->fillUnsafeBCasts(unsafeBcasts);
  }
};

/// A helper for matching against normalization of the attentionMatrix
///
/// attentionMatrix = Q @ KT
/// attentionMatrixNorm = attentionMatrix .* scalarConstant
///   OR
/// attentionMatrixNorm = attentionMatrix ./ scalarConstant
///   OR
/// attentionMatrixNorm = attentionMatrix (no normalization)
///
/// Note normalization is optional!
class MultiHeadAttentionNormalizationMatcher {
private:
  // Will be nullptr if no normalization
  TensorValue scalarConstant;
  bool isDiv;

  // Must bcast scalar in MO due to lack of implicit broadcasting
  Operation *scalarBcastProducer = nullptr;

  // The actual operation which produces the normalized value, either
  // DIV, MUL or nullptr (if normalization was not done).
  Operation *normalizationProducer = nullptr;

  // Top of the pattern, the matrix
  TensorValue attentionMatrix;

  // For unsafe bcasts
  Operation *optBCast1 = nullptr;

  // For internal reshapes
  Operation *optReshape1 = nullptr;

  MultiHeadAttentionNormalizationMatcher() = default;

public:
  static ErrorOr<MultiHeadAttentionNormalizationMatcher>
  match(Operation *producingOp, bool allowUnsafeBCast,
        bool allowInternalReshape) {
    graph_compiler::Logger patternMatcherLogger;

    MultiHeadAttentionNormalizationMatcher matcher;
    auto scalarConstant = m_Any(&matcher.scalarConstant);

    // NOTE: normScalar is derived in some scenarios, it is usually 1 / sqrt(D)
    // So this may be able to made into an implicit match of some sort in the
    // kernel.
    auto pNorm = m_Op<Static::BroadcastToOp>(&matcher.scalarBcastProducer,
                                             scalarConstant);
    auto pAttentionMatrix = m_Any(&matcher.attentionMatrix);

    // normalize by mul with optional bcast
    auto maybeReshape1 =
        pOptionalReshape(&matcher.optReshape1, pAttentionMatrix);
    auto maybeBCast1 = pOptionalBroadcast(&matcher.optBCast1, maybeReshape1);
    auto pNormalizedByMul =
        m_Op<MulOp>(&matcher.normalizationProducer, maybeBCast1, pNorm);
    bool matchesMul =
        pNormalizedByMul.match(producingOp, &patternMatcherLogger);
    bool bcastAllowedMul = allowUnsafeBCast || !matcher.optBCast1;
    bool internalReshapeAllowedMul =
        allowInternalReshape || !matcher.optReshape1;
    if (matchesMul && bcastAllowedMul && internalReshapeAllowedMul) {
      matcher.isDiv = false;
      return matcher;
    }

    // normalize by div with optional bcast
    auto pNormalizedByDiv =
        m_Op<DivOp>(&matcher.normalizationProducer, maybeBCast1, pNorm);
    bool matchesDiv = pNormalizedByDiv.match(producingOp);
    bool bcastAllowedDiv = allowUnsafeBCast || !matcher.optBCast1;
    bool internalReshapeAllowedDiv =
        allowInternalReshape && !matcher.optReshape1;
    if (matchesDiv && bcastAllowedDiv && internalReshapeAllowedDiv) {
      matcher.isDiv = true;
      return matcher;
    }

    // no normalization, producingOp is also the "top of the pattern"
    // aka attentionMask. No optional bcast -- covered by next pattern
    if (producingOp->getNumResults() == 1) {
      matcher.isDiv = false;
      matcher.attentionMatrix = cast<TensorValue>(producingOp->getResult(0));
      return matcher;
    }

    return Error("[MultiHeadAttentionNormalizationMatcher] matching failed.");
  }

  TensorValue getScalarValueForMul(OpBuilder &builder) {
    // Get the scalar value which is appropriate to use for normalizing via mul
    Location loc = attentionMatrix.getLoc();
    if (!scalarConstant) {
      // No normalization, return one as an equivalent
      TensorValue one = builder.create<ConstantOp>(loc, 1.0);
      TensorType targetType =
          one.getType().cloneWith(attentionMatrix.getType().getDtype());
      TensorValue oneCasted = builder.create<CastOp>(loc, targetType, one);
      return oneCasted;
    }

    if (isDiv) {
      TensorValue one = builder.create<ConstantOp>(loc, 1.0);
      TensorType targetType =
          one.getType().cloneWith(scalarConstant.getType().getDtype());
      TensorValue oneCasted = builder.create<CastOp>(loc, targetType, one);

      // Must cast from shape `[]` to `[1]` possibly.
      TensorValue oneBCast = builder.create<Static::BroadcastToOp>(
          loc, scalarConstant.getType(), oneCasted);
      return builder.create<DivOp>(loc, oneBCast, scalarConstant);
    }

    // already read for mul
    return scalarConstant;
  }

  TensorValue getAttentionMatrix() { return attentionMatrix; }
  TensorValue getTopOfPattern() { return getAttentionMatrix(); }
  ErrorOrSuccess hasExpectedRanks() {
    // Allowed to elide the batch dimension
    if (scalarConstant) {
      // May have a pre-broadcasted batch-dimension
      ErrorOrSuccess normScalarOk = checkRank(
          scalarConstant, "[MultiHeadAttentionNormalizationMatcher] normScalar",
          {0, 1});
      if (normScalarOk.isError())
        return normScalarOk.takeError();
    }

    return success();
  }

  void fillUnsafeBCasts(SmallVector<Static::BroadcastToOp> &unsafeBcasts) {
    if (optBCast1)
      unsafeBcasts.push_back(cast<Static::BroadcastToOp>(optBCast1));
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
  match(Operation *producingOp, bool debugMHAPattern) {
    MultiHeadAttentionPrologueMatcher matcher;
    auto pQ = m_Any(&matcher.query);
    auto pKT = m_Any(&matcher.keyT);

    // match Q @ K_T
    Operation *attentionMatrixProducer = nullptr;
    auto pAttentionMatrix =
        m_Op<BatchMatmulOp>(&attentionMatrixProducer, pQ, pKT);

    bool matches = pAttentionMatrix.match(
        producingOp, debugMHAPattern ? &matcher.patternMatcherLogger : nullptr);
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
class MultiHeadAttentionPatternHelper {
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

    **Step 3:
    # Normalize and apply masking
    # This is optional, else `attentionMatrixNorm = attentionMatrix`
    attentionMatrixNorm = attentionMatrix * norm

    # Masking optional, else `attentionMatrixNormMasked = attentionMatrixNorm`
    attnMaskBCast = broadcast(attnMask, attentionMatrixNorm)
    attentionMatrixNormMasked = attentionMatrixNorm + attnMaskBCast

    **Step 4:
    # Apply softmax and reproject result
    attentionMatrixSoftMax = softmax(attentionMatrixNormMasked)
    answer = attentionMatrixSoftMax @ value
  */
private:
  ErrorOr<MultiHeadAttentionPrologueMatcher> prologueMatcher =
      Error("Note set");
  ErrorOr<MultiHeadAttentionEpilogueMatcher> epilogueMatcher = Error("Not set");
  ErrorOr<MultiHeadAttentionNormalizationMatcher> normalizationMatcher =
      Error("Not set");
  ErrorOr<MultiHeadAttentionMaskingMuxMatcher> maskingMatcher =
      Error("Not set.");
  OpBuilder *builder = nullptr;

  ErrorOrSuccess hasExpectedRanks() {
    ErrorOrSuccess prologueOK = prologueMatcher->hasExpectedRanks();
    if (prologueOK.isError())
      return prologueOK.takeError();

    ErrorOrSuccess epilogueOk = epilogueMatcher->hasExpectedRanks();
    if (epilogueOk.isError())
      return epilogueOk.takeError();

    ErrorOrSuccess normalizationOk = normalizationMatcher->hasExpectedRanks();
    if (normalizationOk.isError())
      return normalizationOk.takeError();

    ErrorOrSuccess maskingOk = maskingMatcher->hasExpectedRanks();
    if (maskingOk.isError())
      return maskingOk.takeError();

    // Ensure Q, KT, V and the same rank
    size_t expectedQKVRank = prologueMatcher->getQuery().getType().getRank();
    if (prologueMatcher->getKeyT().getType().getRank() != expectedQKVRank ||
        epilogueMatcher->getValue().getType().getRank() != expectedQKVRank) {
      return Error("[MultiHeadAttentionPrologueMatcher] QKV tensors are not "
                   "all the same rank!");
    }

    // We either allow the rank and attention mask to be the same rank
    // OR the attention mask to be rank 4 and QKV to be rank 3.
    //
    // In the special case they are not the same we expect the BMM to be
    // done at rank 3, and the tensor is reshaped for masking.
    if (maskingMatcher->hasAttentionMask()) {
      MOSH::ShapeAttr attentionMaskShape =
          cast<MOSH::ShapeAttr>(maskingMatcher->getAttentionMaskShape());
      bool maskEqualRank = expectedQKVRank == attentionMaskShape.getRank();
      bool maskDifferentRank =
          (expectedQKVRank == 3) && (attentionMaskShape.getRank() == 4);
      if (!maskEqualRank && !maskDifferentRank) {
        return Error("[MultiHeadAttentionPrologueMatcher] unsupported rank "
                     "difference between QKV and attention mask");
      }
    }
    return success();
  }

  MultiHeadAttentionPatternHelper() = default;

public:
  static ErrorOr<MultiHeadAttentionPatternHelper>
  match(BatchMatmulOp finalOp, OpBuilder &builder, bool allowUnsafeBCast,
        bool allowInternalReshape, bool debugMHAPattern) {
    MultiHeadAttentionPatternHelper matcher;
    matcher.builder = &builder;

    // Note we match things in a reverse-ward direction as we are given the
    // very final operation.
    //
    // This is done by using a set of composable patterns:
    //
    //   D                     C                      B                   A
    // <----[BOTTOM PATTERN] <---- [MIDDLE PATTERN] <---- [TOP PATTERN] <----
    //
    // In general each pattern matches against the **output** and we derive
    // the **input**. This input is the output of the next pattern to match
    // against. For example, we match BOTTOM against `D` and derive `C`
    // we use against MIDDLE.
    //
    // The initial finalOp corresponds to `D` in this diagram.

    // *** NOTE ON `allowUnsafeBCast`
    // This special flag allows the pattern to ignore/match through single
    // optional instances of `mo.static.reshape`.
    //
    // This is done to overcome limitations in shape inference where we are
    // unable to prove two dimension are compatible for an operation like
    // monnx.add (which supports implicit broadcasting), so in lowering we
    // have a broadcast operation remain. In practice, this is extraneous in
    // 100% of the cases currently examined and could be removed given
    // improvements to shape inference and constraint solving.
    //
    // Where should these be inserted? We should generally put these optional
    // bcasts next to ops which might have analogues in other frameworks which
    // are commonly broadcast. For example, "addition" and "multiplication" and
    // even "matmul" may have elements of this. Generally we place the
    // optional broadcast patterns before the operation of interest e.g.
    // `mo.static.broadcast_to --> mo.add`
    //
    // Note, an adversarial example could potentially lead to this pattern
    // replacing things with a non-equivalent kernel, but we determine it to
    // be very unlikely in the wild.

    // This helper functions ensures the input to each pattern has a defining
    // op and has one use and returns the next Operation* to match against.
    auto checkInputToPattern =
        [](TensorValue topOfPattern,
           const std::string &name) -> ErrorOr<Operation *> {
      Twine errMsgRoot =
          Twine("[MultiHeadAttentionPatternHelper] Intermediate value ") +
          Twine(name) + Twine(": ");
      if (!topOfPattern)
        return Error(errMsgRoot + Twine("is nullptr!"));

      if (!topOfPattern.hasOneUse())
        return Error(errMsgRoot + Twine("has more than one use!"));

      Operation *definingOp = topOfPattern.getDefiningOp();
      if (!definingOp)
        return Error(errMsgRoot + Twine("has no defining op!"));

      return definingOp;
    };

    // **Step 4: Match epilogue
    matcher.epilogueMatcher = MultiHeadAttentionEpilogueMatcher::match(
        finalOp, allowUnsafeBCast, allowInternalReshape, debugMHAPattern);
    if (matcher.epilogueMatcher.isError())
      return matcher.epilogueMatcher.takeError();

    ErrorOr<Operation *> maskPatternBottom = checkInputToPattern(
        matcher.epilogueMatcher->getTopOfPattern(), "softmax");
    if (maskPatternBottom.isError())
      return maskPatternBottom.takeError();

    // **Step 3: Match masking
    matcher.maskingMatcher = MultiHeadAttentionMaskingMuxMatcher::match(
        *maskPatternBottom, allowUnsafeBCast, allowInternalReshape,
        debugMHAPattern);
    if (matcher.maskingMatcher.isError())
      return matcher.maskingMatcher.takeError();

    ErrorOr<Operation *> normPatternBottom = checkInputToPattern(
        matcher.maskingMatcher->getTopOfPattern(), "mask_input");
    if (normPatternBottom.isError())
      return normPatternBottom.takeError();

    // **Step 2: Match normalization
    matcher.normalizationMatcher =
        MultiHeadAttentionNormalizationMatcher::match(
            *normPatternBottom, allowUnsafeBCast, allowInternalReshape);
    if (matcher.normalizationMatcher.isError())
      return matcher.normalizationMatcher.takeError();

    ErrorOr<Operation *> prologuePatternBottom = checkInputToPattern(
        matcher.normalizationMatcher->getTopOfPattern(), "attention_matrix");
    if (prologuePatternBottom.isError())
      return prologuePatternBottom.takeError();

    // **Step 1: Match the prologue
    matcher.prologueMatcher = MultiHeadAttentionPrologueMatcher::match(
        *prologuePatternBottom, debugMHAPattern);
    if (matcher.prologueMatcher.isError())
      return matcher.prologueMatcher.takeError();

    // Do additional verification
    ErrorOrSuccess hasExpectedRanks = matcher.hasExpectedRanks();
    if (hasExpectedRanks.isError())
      return hasExpectedRanks.takeError();

    return matcher;
  }

  bool hasAttentionMask() { return maskingMatcher->hasAttentionMask(); }

  // layout B,H,s,D
  TensorValue getAttentionMask() {
    return maskingMatcher->getAttentionMask(*builder);
  }

  // layout B,H,s,D
  TensorValue getQuery() { return prologueMatcher->getQuery(); }

  // layout B,H,D,S
  TensorValue getKeyT() { return prologueMatcher->getKeyT(); }

  // layout B,H,s,D
  TensorValue getValue() { return epilogueMatcher->getValue(); }

  // the factor for mul. Q^K_t to be variance 1
  TensorValue getNormScalar() {
    return normalizationMatcher->getScalarValueForMul(*builder);
  }

  // Returns any unsafe broadcasts that were required to be matched.
  SmallVector<Static::BroadcastToOp> getUnsafeBroadcasts() {
    SmallVector<Static::BroadcastToOp> unsafeBCasts;

    epilogueMatcher->fillUnsafeBCasts(unsafeBCasts);
    normalizationMatcher->fillUnsafeBCasts(unsafeBCasts);
    maskingMatcher->fillUnsafeBCasts(unsafeBCasts);

    return unsafeBCasts;
  }
};

// For MIN Llama 2 demo
struct MultiHeadAttentionPatternCPU : public OpRewritePattern<BatchMatmulOp> {
  bool allowUnsafeBCast, allowInternalReshape, debugMHAPattern;
  MultiHeadAttentionPatternCPU(MLIRContext *ctx, bool allowUnsafeBCast,
                               bool allowInternalReshape, bool debugMHAPattern)
      : OpRewritePattern(ctx, 20), allowUnsafeBCast(allowUnsafeBCast),
        allowInternalReshape(allowInternalReshape),
        debugMHAPattern(debugMHAPattern) {}

  void insertShapeGuard(PatternRewriter &rewriter, GraphOp graphOp,
                        TypedAttr beforeShape, TypedAttr afterShape,
                        Location loc, const std::string &checkName,
                        const std::string &errorMsg) const {
    // Inserts a runtime guard checking (beforeShape == afterShape)
    IntegerType boolTy = rewriter.getI1Type();
    TypedAttr cmpAttr =
        KGEN::emitMLIROperationCall(MOSH::EqualOp::getOperationName(), {},
                                    {beforeShape, afterShape}, boolTy);
    KGEN::ParamDeclAttr condAttr = graphOp.getNewParamOfType(checkName, boolTy);
    rewriter.create<KGEN::ParamDeclareOp>(loc, condAttr, cmpAttr);

    // Add runtime assertion into graph
    TypedAttr assertAttr = MOSH::AssertOp::emitStaticCall(
        KGEN::ParamDeclRefAttr::get(condAttr), errorMsg);
    auto chainTy = rewriter.getType<ChainType>();
    KGEN::ParamDeclAttr chainAttr =
        graphOp.getNewParamOfType(checkName, chainTy);
    rewriter.create<KGEN::ParamDeclareOp>(loc, chainAttr, assertAttr);
  }

  void guardUnsafeBCast(Static::BroadcastToOp unsafeBCast,
                        PatternRewriter &rewriter) const {
    // We expect all bcasts matched in the pattern with `allowUnsafeBCast`
    // to be no-ops (e.g. they don't change shape). This guards against this
    // condition by inserting runtime checks against a potentially unsafe bcast.
    //
    // Note, at this point in the program it is assumed shape functions have
    // been inlined so no internal ops of the pattern produce result parameters.
    //
    // E.g. the following is assumed to be impossible
    // %bmm1 = mo.batch_matmul<() -> N1, N2, N3>(%0, %1) ...
    //            -> !mo.tensor<[N1, N2, N3]>
    // mo.static.broadcast(%something): ... -> !mo.tensor<[N1, N2, N3]>
    //
    // Where the issue is rewriting the pattern would rewrite %bmm1 and
    // remove the definition of `N1` `N2` and `N3`.

    TypedAttr beforeShape =
        unsafeBCast.getInput().getType().getTypedShapeAttr();
    TypedAttr afterShape = unsafeBCast.getResult().getType().getShapeAttr();

    GraphOp graphOp = unsafeBCast->getParentOfType<GraphOp>();
    Location loc = unsafeBCast->getLoc();

    // Make comparison of shapes operator
    insertShapeGuard(rewriter, graphOp, beforeShape, afterShape, loc,
                     "MHA_BCAST_CHECK",
                     "MHA Pattern Detected unsafe broadcast. This is of great "
                     "interest so please report!");
  }

  void guardBHDimensionFused(GraphOp graphOp, ArrayRef<TypedAttr> rank3Dims,
                             ArrayRef<TypedAttr> rank4Dims,
                             PatternRewriter &rewriter, Location loc) const {
    // insert guards that the rank 3 dimensions [A, B, C], and rank4 target
    // dimensions [E, F, G, H], that A = E * F, B == G, C == H.
    //
    // This usually corresponds to the fusion of the B and H dimensions in LLMs

    // mosh.eq expects shapes so wrap results with shapes
    SmallVector<TypedAttr> mulOperands = {rank4Dims[0], rank4Dims[1]};
    SmallVector<TypedAttr> lhsShape = {
        KGEN::ParamOperatorAttr::get(KGEN::POC::MulNuw, mulOperands),
        rank4Dims[2], rank4Dims[3]};
    MOSH::ShapeAttr comparisonLHS = MOSH::ShapeAttr::get(
        lhsShape, MOSH::ShapeType::get(rewriter.getContext()));
    MOSH::ShapeAttr comparisonRHS = MOSH::ShapeAttr::get(
        rank3Dims, MOSH::ShapeType::get(rewriter.getContext()));

    // Add run time checks for changed dimensions
    insertShapeGuard(rewriter, graphOp, comparisonLHS, comparisonRHS, loc,
                     "MHA_INTERNAL_RESHAPE_CHECK",
                     "MHA Pattern Detected unsafe internal reshape. This is of "
                     "great interest so please report!");
  }

  TensorValue upcastRankWithGuard(GraphOp graphOp, TensorValue rank3Input,
                                  ArrayRef<TypedAttr> rank4Target,
                                  PatternRewriter &rewriter) const {
    // It could be the case that the QKV tensors and the mask are not the
    // same rank. Sometimes, the BMM are done in rank 3, and later reshaped
    // to a rank 4 version to do masking. The rank 3 version is the same
    // as the rank 4 except the batch and head dimensions are combined.
    //
    // We allow this pattern here by reshaping Q to attentionMask. We
    // assume this occurs because the rank 4 tensor is shaped: [B, H, S, S]
    //
    // while the rank 3 tensors are shaped [B * H, S, D] or [B * H, S, D].
    //
    // This method handles this by, given a rank3Input with assumed dimensions
    // [B * H, S, D] or [B * H, D, S] and the dimensions of the rank4 input
    // returns the newly reshaped rank4 input. [B, H, S, D] or [B, H, D, S].
    ArrayRef<TypedAttr> oldDims = rank3Input.getType().getDims();
    SmallVector<TypedAttr> newDims(4);

    Location loc = rank3Input.getLoc();

    newDims[0] = rank4Target[0];
    newDims[1] = rank4Target[1];
    newDims[2] = oldDims[1];
    newDims[3] = oldDims[2];

    // Create new tensor with given shape
    MOSH::ShapeType ty = rank3Input.getType().getTypedShapeAttr().getType();
    MOSH::ShapeAttr newShape = MOSH::ShapeAttr::get(newDims, ty);
    TensorType newType = rank3Input.getType().cloneWith(newShape);
    TensorValue newValue =
        rewriter.create<Static::ReshapeOp>(loc, newType, rank3Input);

    // Need MOSH::ShapeAttr to match
    guardBHDimensionFused(graphOp,
                          rank3Input.getType().getTypedShapeAttr().getValues(),
                          newDims, rewriter, loc);
    return newValue;
  }

  LogicalResult matchAndRewrite(BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
    DeviceRefAttr device = op->getParentOfType<MO::GraphOp>().getDeviceRef();

    // Just assume we want to use the CPU kernel only if the default device
    // is cpu (e.g. if the default was CUDA, run on CUDA).
    //
    // That way avoid device logic for now.
    if (device && device.getLabel() != M::kCPULabel)
      return rewriter.notifyMatchFailure(op->getLoc(), "Devices are wrong.");

    ErrorOr<MultiHeadAttentionPatternHelper> matcher =
        MultiHeadAttentionPatternHelper::match(op, rewriter, allowUnsafeBCast,
                                               allowInternalReshape,
                                               debugMHAPattern);
    if (matcher.isError())
      return rewriter.notifyMatchFailure(op.getLoc(), matcher.getError());

    // If we have unsafe broadcasts, we want to add runtime guards against them.
    for (Static::BroadcastToOp unsafeBCast : matcher->getUnsafeBroadcasts()) {
      guardUnsafeBCast(unsafeBCast, rewriter);
    }

    // NOTE: the creation of `matcher` must imply future success()
    // as it has a handle to the rewriter which can be used to create new ops.
    // Creating new ops and then returning failure may result in the pass
    // not converging (as the new ops will tend to be DCEd.).
    // Attention masking case
    if (matcher->hasAttentionMask()) {

      // QKV are guaranteed to have the same ranks.
      TensorValue query = matcher->getQuery();
      TensorValue keyT = matcher->getKeyT();
      TensorValue value = matcher->getValue();

      // Attention mask may or may not have the same rank
      TensorValue attentionMask = matcher->getAttentionMask();

      // Sometimes the BMM are done in rank 3 where the batch and head dims
      // are combined. Masking is still done in rank4 however.
      TensorType fusedKernelOutType = op.getResult().getType();
      GraphOp graphOp = op->getParentOfType<GraphOp>();

      if (query.getType().getRank() == 3 &&
          attentionMask.getType().getRank() == 4) {
        ArrayRef<TypedAttr> rank4Target = attentionMask.getType().getDims();
        query = upcastRankWithGuard(graphOp, query, rank4Target, rewriter);
        keyT = upcastRankWithGuard(graphOp, keyT, rank4Target, rewriter);
        value = upcastRankWithGuard(graphOp, value, rank4Target, rewriter);
        fusedKernelOutType =
            upcastRankWithGuard(graphOp, op.getResult(), rank4Target, rewriter)
                .getType();
      }

      SmallVector<Value> inputValues = {query, keyT, value, attentionMask,
                                        matcher->getNormScalar()};

      SmallVector<Type> outputTypes = {fusedKernelOutType};
      ValueRange inputs(inputValues);
      TypeRange outputTy(outputTypes);
      rewriter.setInsertionPointAfterValue(op);
      auto result = rewriter.create<CustomOp>(op->getLoc(), outputTy, inputs,
                                              "with_mask_fused_attention_cpu");
      auto fusedKernelResult = cast<TensorValue>(result->getResult(0));
      fusedKernelResult.setType(fusedKernelOutType);

      // NOTE: if this is true, then checks in matcher should guarantee we are
      // moving from rank4 back to rank3.
      if (fusedKernelResult.getType() != op.getType()) {
        MOSH::ShapeAttr fusedKernelShape =
            cast<MOSH::ShapeAttr>(fusedKernelResult.getType().getShapeAttr());
        MOSH::ShapeAttr targetShape =
            cast<MOSH::ShapeAttr>(op.getType().getShapeAttr());
        guardBHDimensionFused(graphOp, targetShape.getValues(),
                              fusedKernelShape.getValues(), rewriter,
                              op->getLoc());
        fusedKernelResult = rewriter.create<Static::ReshapeOp>(
            op->getLoc(), op.getType(), fusedKernelResult);
      }

      rewriter.replaceAllUsesWith(op, fusedKernelResult);
    } else {
      // No attention masking case
      SmallVector<Value> inputValues = {matcher->getQuery(), matcher->getKeyT(),
                                        matcher->getValue(),
                                        matcher->getNormScalar()};
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
                 ConvTransposeConstFilterPattern, SplitZeroAxisPattern,
                 LayerNormPattern>(ctx);

    if (enableMHAPattern.getValue()) {
      patterns.add<MultiHeadAttentionPatternCPU>(
          ctx, mhaAllowUnsafeBCast.getValue(),
          mhaAllowInternalReshape.getValue(), debugMHAPattern.getValue());
    }

    MO::GraphOp graphOp = getOperation();
    if (failed(applyPatternsAndFoldGreedily(graphOp, std::move(patterns))))
      return signalPassFailure();
    MO::synchronizeMetadataUpdates(graphOp);
  }
};

} // namespace M::MOGG
