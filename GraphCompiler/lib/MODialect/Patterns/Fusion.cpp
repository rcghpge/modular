//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/Patterns/Fusion.h"

#include "GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOGGDialect/Support/SplitSlicePatterns.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GraphCompiler/Matchers.h"
#include "GraphCompiler/Support/DataflowUtils.h"
#include "GraphCompiler/Support/Dims.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/DeviceSpecs.h"
#include "Support/ML/Permutation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

#include <limits>
#include <numeric>
#if defined(__APPLE__)
#include <pthread.h>
#endif

using namespace M;
using namespace M::MO;
using namespace M::MOGG;
using namespace GraphCompiler::Matchers;

namespace M::MO {

void PackParamCache::put(MO::TensorType inputTy,
                         mlir::DictionaryAttr kgenParams,
                         MO::TensorType outputTy) {
  cache[inputTy][kgenParams] = outputTy;
}

MO::TensorType PackParamCache::get(MO::TensorType inputTy,
                                   mlir::DictionaryAttr kgenParams) {
  auto cacheIt = cache.find(inputTy);
  if (cacheIt == cache.end())
    return {};
  auto &kgenParamsToOutTy = cacheIt->second;
  auto it = kgenParamsToOutTy.find(kgenParams);
  if (it == kgenParamsToOutTy.end())
    return {};
  return it->second;
}

namespace {

// The size which we regard as too small to run the special matmuls.
constexpr int64_t kTinyMatmulFlops = 5000;
constexpr int64_t kTinyBMMatmulFlops = 512;

static bool isMatmulSmall(MO::MatmulOp matmul) {
  MO::TensorType aType = matmul.getInputA().getType();
  MO::TensorType bType = matmul.getInputB().getType();

  if (!aType.hasStaticShape() || !bType.hasStaticShape())
    return false;

  SmallVector<int64_t> aDims = aType.getIntDims();
  SmallVector<int64_t> bDims = bType.getIntDims();

  // M * K * N
  int64_t approxFlops = aDims[0] * aDims[1] * bDims[1];
  return approxFlops < kTinyMatmulFlops;
}

static bool isBMMSmall(MO::BatchMatmulOp matmul) {
  MO::TensorType aType = matmul.getInputA().getType();
  MO::TensorType bType = matmul.getInputB().getType();

  if (!aType.hasStaticShape() || !bType.hasStaticShape())
    return false;

  SmallVector<int64_t> aDims = aType.getIntDims();
  SmallVector<int64_t> bDims = bType.getIntDims();

  size_t rank = aType.getRank();
  int64_t batchSize = 1;
  for (size_t i = 0; i < aDims.size() - 2; ++i)
    batchSize *= aDims[i];

  // M * K * N
  int64_t approxFlops =
      batchSize * (aDims[rank - 2] * aDims[rank - 1] * bDims[rank - 1]);
  return approxFlops < kTinyBMMatmulFlops;
}

// Return the device attached to the op or the given `defaultTargetDevice` if
// it does not exist.
static DeviceRefAttr getTargetDevice(MO::MOHasDeviceInterface op,
                                     DeviceRefAttr defaultTargetDevice) {
  DeviceRefAttr graphDevice = op.getExecutionDevice();
  if (!graphDevice && defaultTargetDevice)
    graphDevice = defaultTargetDevice;

  return graphDevice;
}

Value lowerMatmulPackingFunc(Location loc, MO::DeviceAwareRewriter &rewriter,
                             MO::GraphOp graphOp, PackParamCache &cache,
                             Value weight, MO::TensorType inputType,
                             MO::TensorType resultType, bool transpose) {

  MO::TensorType weightsType = cast<MO::TensorType>(weight.getType());

  Type inputEleType = inputType.getElementType();
  Type weightsEleType = weightsType.getElementType();
  Type resultEleType = resultType.getElementType();

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

  // Create a pair of parameters for the new type.
  auto newParams = KGEN::ParamDeclArrayAttr::get(
      graphOp.getContext(),
      {graphOp.getNewDimParam("MMPackSh"), graphOp.getNewDimParam("MMPackSh")});

  // Create the packed B shape function.
  // TODO(#18770) Find a better registration mechanism.
  SmallVector<mlir::NamedAttribute> shapeFuncAttrs = attrs;
  shapeFuncAttrs.push_back(
      rewriter.getNamedAttr("transpose_in_0", rewriter.getBoolAttr(transpose)));
  mlir::DictionaryAttr shapeFuncKgenParams =
      rewriter.getDictionaryAttr(shapeFuncAttrs);

  MO::TensorType packedWeightsType = cache.get(inputType, shapeFuncKgenParams);
  // If we haven't lowered the shape function, lower it now so we have
  // parameterized shapes.
  if (!packedWeightsType) {
    rewriter.create<MO::InvokeShapeFuncOp>(
        loc, SmallVector<Value>{weight}, "pack_matmul_b_shape_func", newParams,
        shapeFuncKgenParams, ArrayAttr{});

    SmallVector<TypedAttr> outputShapeDims;
    for (KGEN::ParamDeclAttr outputParamDecl : newParams)
      outputShapeDims.push_back(KGEN::ParamDeclRefAttr::get(outputParamDecl));

    auto moshShapeTy = MOSH::ShapeType::get(graphOp.getContext());
    auto shapeAttr = MOSH::ShapeAttr::get(outputShapeDims, moshShapeTy);
    auto packedWeightsLayout = rewriter.getAttr<MO::LayoutAttr>("KNkni");
    packedWeightsType =
        weightsType.cloneWith(shapeAttr).cloneWith(packedWeightsLayout);

    // Cache the result so the next type has a hit too.
    cache.put(inputType, shapeFuncKgenParams, packedWeightsType);
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
    Value input, indices, reduceAxis;

    // The gather `axis` is now a compile-time `index` attribute, so it is no
    // longer a matchable operand; we check it explicitly below.
    if (!m_Op<MO::ReduceAddOp>(m_Op<MO::GatherOp>(m_MOMatrix(m_Any(&input)),
                                                  m_MOMatrix(m_Any(&indices))),
                               m_ScalarConstant(&reduceAxis, 1))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !indices || !reduceAxis)
      return rewriter.notifyMatchFailure(op, "null input value");

    // The gather axis must be 0.
    auto gatherOp = op.getInput().getDefiningOp<MO::GatherOp>();
    if (!gatherOp || gatherOp.getAxisAttr().getInt() != 0)
      return rewriter.notifyMatchFailure(op, "gather axis must be 0");

    rewriter.replaceOpWithNewOp<MO::GatherSumOp>(
        op, op->getResultTypes(), SmallVector<Value>{input, indices});

    return mlir::success();
  }
};

// Decomposes `reduce.add(input, axis)` over a small, static-extent axis (the
// other dims may be dynamic) into a sum of
struct DecomposeSmallReduceAddPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern, MO::ReduceAddOp> {
  // Only decompose reductions whose extent is in (1, kMaxDecomposeExtent];
  // larger extents would bloat the IR and can overflow the elementwise fusion
  // set, so they stay as a dedicated reduce kernel.
  static constexpr int64_t kMaxDecomposeExtent = 8;

  // Lower benefit than GatherSumPattern so the more specific gather+reduce
  // fusion wins when both could match.
  DecomposeSmallReduceAddPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}

  static bool feedsRmsNorm(Value v) {
    while (v.hasOneUse()) {
      Operation *user = *v.getUsers().begin();
      if (isa<MO::ReduceRMSNormOp>(user))
        return true;
      if (isa<MO::CastOp, MO::RebindOp>(user)) {
        v = user->getResult(0);
        continue;
      }
      return false;
    }
    return false;
  }

  LogicalResult
  deviceAwareMatchAndRewrite(MO::ReduceAddOp op,
                             MO::DeviceAwareRewriter &rewriter) const override {
    Value input = op.getInput();

    // Leave gather + reduce.add to GatherSumPattern.
    if (input.getDefiningOp<MO::GatherOp>())
      return rewriter.notifyMatchFailure(
          op, "gather + reduce handled by GatherSumPattern");

    // Only decompose when the result feeds an rms_norm; otherwise the slices +
    // adds have nothing to fuse into.
    if (!feedsRmsNorm(op.getResult()))
      return rewriter.notifyMatchFailure(
          op, "reduce.add result does not feed an rms_norm");

    auto inputType = dyn_cast<MO::TensorType>(input.getType());
    if (!inputType || !inputType.hasRank())
      return rewriter.notifyMatchFailure(op,
                                         "reduce input not statically ranked");

    int64_t rank = inputType.getRank();
    int64_t axis = normalizeAxis(op.getAxisAttr().getInt(), rank);
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "reduce axis out of range");

    // Only the *reduced* axis needs a static extent
    int64_t extent = inputType.getIntDim(axis);
    if (mlir::ShapedType::isDynamic(extent))
      return rewriter.notifyMatchFailure(op, "reduce axis extent not static");
    if (extent <= 1 || extent > kMaxDecomposeExtent)
      return rewriter.notifyMatchFailure(
          op, "reduce extent not in (1, kMaxDecomposeExtent]");

    Location loc = op.getLoc();
    // Every per-index slice and every partial sum has the reduce.add result
    // type (the input shape with `axis` collapsed to 1).
    Type sliceType = op.getResult().getType();

    SmallVector<int64_t> step(rank, 1);
    Value stepVal = rewriter.createHostConstant(loc, step);

    Value acc;
    for (int64_t k = 0; k < extent; ++k) {
      SmallVector<int64_t> start(rank, 0);
      SmallVector<int64_t> stop;
      stop.reserve(rank);
      // Non-reduced dims span their full extent. A dynamic dim uses INT64_MAX,
      // which slice's numpy semantics clamp to the runtime size; static dims
      // keep their exact size so the all-static path stays byte-identical.
      for (int64_t i = 0; i < rank; ++i) {
        int64_t dim = inputType.getIntDim(i);
        stop.push_back(mlir::ShapedType::isDynamic(dim)
                           ? std::numeric_limits<int64_t>::max()
                           : dim);
      }
      start[axis] = k;
      stop[axis] = k + 1;
      Value startVal = rewriter.createHostConstant(loc, start);
      Value stopVal = rewriter.createHostConstant(loc, stop);
      Value slice = rewriter
                        .create<MO::SliceOp>(loc, sliceType, input, startVal,
                                             stopVal, stepVal)
                        .getResult();
      acc = k == 0 ? slice
                   : rewriter.create<MO::AddOp>(loc, sliceType, acc, slice)
                         .getResult();
    }

    rewriter.replaceOp(op, acc);
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
  DeviceRefAttr targetDevice;
  MatmulConstBPattern(MLIRContext *ctx, PackParamCache &cache,
                      DeviceRefAttr targetDevice)
      : DeviceAwareRewritePattern(ctx, 3), targetDevice(targetDevice),
        cache(cache) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::MatmulOp op,
                             MO::DeviceAwareRewriter &rewriter) const override {
    DeviceRefAttr device = getTargetDevice(op, targetDevice);
    // TODO: CUDA does not support pre-packing yet
    if (device && device.getLabel() == M::kGPULabel)
      return rewriter.notifyMatchFailure(op,
                                         "CUDA does not support prepacking.");

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
      return rewriter.notifyMatchFailure(op, "Pattern failed to match.");

    MO::TensorType weightsType = cast<MO::TensorType>(weight.getType());
    MO::LayoutAttr weightsLayout = op.getInputB().getType().getLayoutAttr();
    if (weightsLayout.getFormat() == "KNkni")
      return rewriter.notifyMatchFailure(op, "Matmul is already packed");

    if (weightsLayout.getFormat() == "NK")
      return rewriter.notifyMatchFailure(op,
                                         "Transposed op has already been run");

    // TODO(KERN-1160): This can change back to MO::ConstantLike after the
    // kernel crash described in this ticket has been resolved.
    if (!input || !dependsOnConstantsOnly<MO::ConstantOp, MO::ChainCreateOp>(
                      weight.getDefiningOp()))
      return rewriter.notifyMatchFailure(op, "Need to solve KERN-1160.");

    if (transpose) {
      if (!perm || validate2DTranspose(perm).failed())
        return rewriter.notifyMatchFailure(op, "invalid transpose");
    }

    if (!rewriter.addOps({transpose})) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    if (!rewriter.deviceAnnots.isConsistentWith(getHostDevice(op))) {
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
    auto graphOp = op->getParentOfType<MO::GraphOp>();
    Value packedWeights =
        lowerMatmulPackingFunc(op->getLoc(), rewriter, graphOp, cache, weight,
                               inputType, resultType, transpose);

    rewriter.modifyOpInPlace(op, [&] { op.setOperand(1, packedWeights); });

    return mlir::success();
  }

private:
  PackParamCache &cache;
};

// Detects a matmul with a `mo.matmul(A, transpose(B))` pattern and replaces it
// with a single operation `mo.matmul(A, B, transpose_B=true)`.
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
        weightDefOp &&
        !dependsOnConstantsOnly<MO::ConstantLike, MO::ChainCreateOp>(
            weightDefOp)) {
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
    rewriter.modifyOpInPlace(op, [&] {
      op.setOperand(1, rebWeight);
      op.setTransposeB(true);
    });

    return mlir::success();
  }
};

// A pattern which detects `matmul(A, B)` where B is a constant and makes things
// `matmul(A, transpose(B), transpose_b=true)`. We expect constant-folding to
// materialize transpose(B) at compile time. Only for GPU at the moment.
//
// It differs from `MatmulTranspBPattern` in that it will force even regular
// matmuls into a transposed form if it's weights are constants.
//
// In combination with `MatmulTranspBPattern` this pattern will make most
// matmuls operate on transposed pre-packed weights, which are faster on GPU
// type architectures.
struct ForceMatmulTransposeBForGPU
    : public MO::DeviceAwareRewritePattern<OpRewritePattern, MO::MatmulOp> {
  DeviceRefAttr targetDevice;
  ForceMatmulTransposeBForGPU(MLIRContext *ctx, DeviceRefAttr targetDevice)
      : DeviceAwareRewritePattern(ctx, 2), targetDevice(targetDevice) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::MatmulOp matmul,
                             MO::DeviceAwareRewriter &rewriter) const override {
    TensorValue weights = matmul.getInputB();
    Operation *weightProducer = weights.getDefiningOp();

    DeviceRefAttr defaultDevice = getTargetDevice(matmul, targetDevice);
    bool defaultDeviceIsCuda =
        defaultDevice && defaultDevice.getLabel() == M::kGPULabel;

    if (weights.getType().getLayoutAttr().getFormat() != "KN")
      return rewriter.notifyMatchFailure(
          matmul, "Matmul use of weights is not annotated with the "
                  "expected default layout.");

    if (!dependsOnNoAliasConstantLikesOnly(weightProducer))
      return rewriter.notifyMatchFailure(
          matmul, "Weights do not only depend on constants");

    // Check all uses are as weights in the matmuls
    for (Operation *user : weights.getUsers()) {
      auto matmulUser = dyn_cast<MatmulOp>(user);
      if (!matmulUser)
        return rewriter.notifyMatchFailure(
            matmul, "Not all users of the weight are matmuls!");

      if (matmulUser.getInputB() != weights ||
          matmulUser.getInputA() == weights)
        return rewriter.notifyMatchFailure(
            matmulUser,
            "Matmul user does not exclusively use weight as `B` tensor.");

      bool matmulIsUnset = !matmulUser.getExecutionDevice();
      bool matmulIsSetToCuda =
          matmulIsUnset
              ? false
              : matmulUser.getExecutionDevice().getLabel() == M::kGPULabel;
      if (!matmulIsSetToCuda && !(matmulIsUnset && defaultDeviceIsCuda))
        return rewriter.notifyMatchFailure(
            matmulUser, "Found matmul user with incorrect devices");
    }

    // Rewrite the weights
    rewriter.setInsertionPointAfter(weightProducer);
    LayoutAttr matmulWeightsLayout = LayoutAttr::get(getContext(), "NK");

    // Transpose weights (assume to be folded by constant-folding)
    auto oldShape = weights.getType().getTypedShapeAttr().getValues();
    SmallVector<TypedAttr> newShape = {oldShape[1], oldShape[0]};
    auto newTransposeTy =
        weights.getType().cloneWith(newShape).cloneWith(matmulWeightsLayout);
    auto perm = rewriter.createHostConstant(weights.getLoc(),
                                            SmallVector<int64_t>{1, 0});
    auto transposedWeights = rewriter.create<MO::TransposeOp>(
        weights.getLoc(), newTransposeTy, weights, perm);

    rewriter.replaceAllUsesExcept(weights, transposedWeights,
                                  transposedWeights);
    return success();
  }
};

struct BatchMatmulAdjBPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern,
                                           MO::BatchMatmulOp> {
  DeviceRefAttr targetDevice;
  BatchMatmulAdjBPattern(MLIRContext *ctx, DeviceRefAttr targetDevice)
      : DeviceAwareRewritePattern(ctx, 2), targetDevice(targetDevice) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::BatchMatmulOp op,
                             MO::DeviceAwareRewriter &rewriter) const override {
    Value input, weight, perm;

    DeviceRefAttr device = op.getInputA().getType().getDeviceRef();
    if (!device)
      device = targetDevice;
    bool runsOnGPU = device && device.getLabel() == M::kGPULabel;
    if (runsOnGPU)
      return rewriter.notifyMatchFailure(
          op,
          "Do not support GPU for batch matmul transpose_b=true at this time");

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

    rewriter.modifyOpInPlace(op, [&] {
      // Take the non-transposed weight.
      op.setOperand(1, weight);

      // Mark this matmul as being a transposed kernel.
      op.setTransposeBAttr(rewriter.getBoolAttr(true));
    });

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
    if (concat.getAxisAttr().getInt() != 0)
      return rewriter.notifyMatchFailure(concat, "axis not constant 0");

    Block *concatBlock = concat.getResult().getParentBlock();
    DenseSet<Value> uniqueInputs;

    auto graphOp = concat->getParentOfType<GraphOp>();
    DeviceRefAttr defaultDevice =
        getTargetDevice(concat, getHostDevice(graphOp));
    auto getDevice = [&](TensorType ty) {
      auto maybeDev = ty.getDeviceRef();
      return maybeDev ? maybeDev : defaultDevice;
    };

    // We need all devices to be the same.
    DeviceRefAttr targetDevice =
        getDevice(cast<TensorType>(concat.getInputs()[0].getType()));
    for (Value input : concat.getInputs()) {
      TensorType tensorTy = cast<TensorType>(input.getType());
      if (getDevice(tensorTy) != targetDevice) {
        return rewriter.notifyMatchFailure(
            concat, "Not all inputs are on same device.");
      }
      Location inputLoc = input.getLoc();
      // This ensures we don't generate a slow `MGP::BufferConcatOp` that won't
      // be optimized away by `MGP::ConcatInPlacePass`.
      auto [_, inserted] = uniqueInputs.insert(input);
      if (!inserted)
        return rewriter.notifyMatchFailure(inputLoc, "duplicated input value");

      if (input.getParentBlock() != concatBlock) {
        return rewriter.notifyMatchFailure(
            inputLoc, "concat in a different block than inputs");
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

struct ConvConstFilterPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern, MO::ConvOp> {
  DeviceRefAttr targetDevice;
  ConvConstFilterPattern(MLIRContext *ctx, DeviceRefAttr targetDevice)
      : DeviceAwareRewritePattern(ctx, 3), targetDevice(targetDevice) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::ConvOp op,
                             MO::DeviceAwareRewriter &rewriter) const override {
    DeviceRefAttr device = op.getInput().getType().getDeviceRef();
    if (!device)
      device = getTargetDevice(op, targetDevice);
    // TODO: CUDA does not support pre-packing yet
    if (device && device.getLabel() == M::kGPULabel)
      return rewriter.notifyMatchFailure(op,
                                         "CUDA does not support prepacking.");

    // Define some meta variables.
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    MO::TensorValue filter = op.getFilter();
    MO::TensorType filterType = filter.getType();
    MO::LayoutAttr filterLayout = filterType.getLayoutAttr();

    if (llvm::is_contained({"FQRSCf", "FRSCf"}, filterLayout.getFormat()))
      return rewriter.notifyMatchFailure(op, "filter is already packed");

    auto maybeNumGroups = MO::tryGetConstantInt(op.getNumGroups());
    if (!maybeNumGroups) {
      return rewriter.notifyMatchFailure(
          op, "currently only pack constant num_groups");
    }

    bool filterIsConstant =
        dependsOnNoAliasConstantLikesOnly(filter.getDefiningOp());
    if (!filterIsConstant && *maybeNumGroups <= 1)
      return rewriter.notifyMatchFailure(
          op, "could pack this non-constant filter, but for num_groups <= 1 "
              "it isn't worth the runtime repack cost");

    if (!rewriter.deviceAnnots.isConsistentWith(getHostDevice(op))) {
      // TODO(#25114): Support outside of host.
      return rewriter.notifyMatchFailure(
          op, "packing is only supported on the host device");
    }

    // For FCRS/FCQRS filters, insert a transpose to RSCF/QRSCF first.
    // The transpose on a constant will be folded by constant folding.
    // This avoids needing a separate FCRS packing kernel.
    bool isFCRS = (filterLayout.getFormat() == "FCRS" ||
                   filterLayout.getFormat() == "FCQRS");
    if (isFCRS) {
      size_t rank = filterType.getRank();
      // FCRS [F,C,R,S] -> RSCF [R,S,C,F]: perm = [2,3,1,0]
      // FCQRS [F,C,Q,R,S] -> QRSCF [Q,R,S,C,F]: perm = [2,3,4,1,0]
      SmallVector<int64_t> perm;
      for (size_t i = 2; i < rank; ++i)
        perm.push_back(i);
      perm.push_back(1);
      perm.push_back(0);

      // Create permutation constant.
      auto permConstant = rewriter.createHostConstant(loc, perm);

      // Build the RSCF shape from the FCRS shape.
      SmallVector<TypedAttr> rscfDims;
      for (auto idx : perm)
        rscfDims.push_back(filterType.getTypedShapeAttr().getValues()[idx]);
      auto rscfShapeAttr =
          MOSH::ShapeAttr::get(rscfDims, rewriter.getType<MOSH::ShapeType>());

      StringRef rscfLayoutStr = (rank == 4) ? "RSCF" : "QRSCF";
      auto rscfLayout = rewriter.getAttr<MO::LayoutAttr>(rscfLayoutStr);
      auto rscfType = filterType.cloneWith(rscfShapeAttr).cloneWith(rscfLayout);

      filter =
          rewriter.create<MO::TransposeOp>(loc, rscfType, filter, permConstant);
      filterType = filter.getType();
      filterLayout = filterType.getLayoutAttr();
    }

    // Create shape parameters for the new type.
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

    SmallVector<KGEN::ParamDeclAttr> paramDecls;
    for (char c : packedLayoutStr)
      paramDecls.push_back(graphOp.getNewDimParam(Twine(c)));

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
    rewriter.modifyOpInPlace(op, [&] { op.setOperand(1, packedFilter); });
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

    if (!dependsOnNoAliasConstantLikesOnly(filter.getDefiningOp()))
      return rewriter.notifyMatchFailure(op, "filter isn't constant");

    if (!rewriter.deviceAnnots.isConsistentWith(getHostDevice(op))) {
      // TODO(#25114): Support outside of host.
      return rewriter.notifyMatchFailure(
          op, "packing is only supported on the host device");
    }

    if (filterLayout.getFormat() == "CFRS")
      return rewriter.notifyMatchFailure(op,
                                         "Packing CFRS format isn't supported");

    // Create shape parameters for the new type.
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

    SmallVector<KGEN::ParamDeclAttr> paramDecls;
    for (char c : packedLayoutStr)
      paramDecls.push_back(graphOp.getNewDimParam(Twine(c)));

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
    rewriter.modifyOpInPlace(op, [&] { op.setOperand(1, packedFilter); });
    return mlir::success();
  }
};

struct SplitZeroAxisPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern, MO::SplitOp> {
  SplitZeroAxisPattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::SplitOp op,
                             MO::DeviceAwareRewriter &rewriter) const override {
    int64_t axis = op.getAxisAttr().getInt();
    if (axis != 0)
      return rewriter.notifyMatchFailure(op, "axis not 0");

    // Ensure input isn't constant
    if (op.getInput().getDefiningOp<MO::ConstantOp>())
      return rewriter.notifyMatchFailure(op, "input is a constant");

    if (!rewriter.deviceAnnots.isConsistentWith(getHostDevice(op))) {
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

    // `axis` is a compile-time `index` attribute on `mo.reduce.mean`, so it is
    // no longer matched as an operand; the mean ops are captured here and their
    // axis attributes are validated (must be the last axis) after the match.
    mlir::Operation *meanOp = nullptr;
    mlir::Operation *varOp = nullptr;
    // TODO: Current capture system can't distinguish multiple uses of the same
    // value in the pattern, need to fix, this might be slightly risky
    Value input;
    auto x = m_Ranked(3, m_Any(&input));
    auto mean =
        m_Op<MO::Static::BroadcastToOp>(po(), m_Op<MO::MeanOp>(&meanOp, x));
    auto diff = m_Op<MO::SubOp>(po(), x, mean);
    auto square = m_Optional(
        m_Op<MO::MulOp>(po(), diff, diff),
        m_Op<MO::PowOp>(
            po(), diff,
            m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                2)));
    auto var = m_Op<MO::MeanOp>(&varOp, square);
    // pytorch specific pattern
    auto pRatio = m_Op<MO::DivOp>(
        po(), diff,
        m_Op<MO::Static::BroadcastToOp>(
            po(), m_Op<MO::SqrtOp>(
                      po(), m_BinaryCommutativeOp<MO::AddOp>(po(), var, ep))));
    auto pLayernorm = m_BinaryCommutativeOp<MO::AddOp>(
        po(), m_BinaryCommutativeOp<MO::MulOp>(po(), gam, pRatio), bet);

    // TF specific Pattern
    auto tRDenom = m_BinaryCommutativeOp<MO::MulOp>(
        po(), gam,
        m_Op<MO::Static::BroadcastToOp>(
            po(), m_Op<MO::RsqrtOp>(
                      po(), m_BinaryCommutativeOp<MO::AddOp>(po(), var, ep))));
    auto tTerm1 = m_BinaryCommutativeOp<MO::MulOp>(po(), x, tRDenom);
    auto tTerm2 = m_BinaryCommutativeOp<MO::MulOp>(po(), mean, tRDenom);
    auto tTerm3 = m_Op<MO::SubOp>(po(), bet, tTerm2);
    auto tLayernorm = m_BinaryCommutativeOp<MO::AddOp>(po(), tTerm1, tTerm3);

    if (!tLayernorm.match(op.getResult()) && !pLayernorm.match(op.getResult()))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !beta || !gamma || !eps)
      return rewriter.notifyMatchFailure(op, "null input value");

    // Both reductions must be along the last axis (the layer-norm axis).
    auto isLastAxis = [&](mlir::Operation *meanLikeOp) {
      auto reduction = cast<MO::MeanOp>(meanLikeOp);
      int64_t rank = reduction.getInput().getType().getRank();
      int64_t axis = reduction.getAxisAttr().getInt();
      if (axis < 0)
        axis += rank;
      return axis == rank - 1;
    };
    if (!meanOp || !varOp || !isLastAxis(meanOp) || !isLastAxis(varOp))
      return rewriter.notifyMatchFailure(op, "reduction axis is not last axis");

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    // TODO(#27337): remove when gpu kernel exists
    if (!rewriter.deviceAnnots.isConsistentWith(getHostDevice(op))) {
      return rewriter.notifyMatchFailure(op,
                                         "Layer norm only implemented on cpu");
    }

    // Handle Scalar beta and gamma
    auto betaType = llvm::cast<MO::TensorType>(beta.getType());
    if (betaType && betaType.getRank() == 0) {
      beta = rewriter.create<MO::Static::BroadcastToOp>(
          op.getLoc(),
          MO::TensorType::get(SmallVector<TypedAttr>{op.getType().getDim(2)},
                              betaType.getElementType()),
          beta);
    }

    auto gammaType = llvm::cast<MO::TensorType>(gamma.getType());
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
getPatternMatchingError(GraphCompiler::Logger &patternMatcherLogger) {
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

template <typename pInT>
auto inline pOptionalCast(Operation **bindDest, pInT patternIn) {
  // Allows optional insertion of reshapes and bcasts between this pattern
  // Specifically patternIn --> (optional) mo.static.reshape
  //  --> (optional) mo.static.broadcast
  auto reshape = m_Op<CastOp>(bindDest, patternIn);
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

  // For optional casts
  Operation *optCast1 = nullptr;
  Operation *optCast2 = nullptr;

  GraphCompiler::Logger patternMatcherLogger;
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

    // support up to two optional casts (from bf16 --> fp32 --> bf16)
    auto maybeCast1 = pOptionalCast(&matcher.optCast1, pSoftmaxAttnMatrix);
    auto maybeCast2 = pOptionalCast(&matcher.optCast2, maybeCast1);

    // optional broadcast + re-project using V
    auto maybeReshape2 = pOptionalReshape(&matcher.optReshape2, maybeCast2);
    auto maybeBCast2 = pOptionalBroadcast(&matcher.optBCast2, maybeReshape2);
    auto pAnswer =
        m_Op<BatchMatmulOp>(&matcher.reprojectProducer, maybeBCast2, pV);

    bool matches = pAnswer.match(reprojection,
                                 debugMHAPattern ? &matcher.patternMatcherLogger
                                                 : nullptr);
    if (!matches)
      return Error("[MultiHeadAttentionEpilogueMatcher]: " +
                   getPatternMatchingError(matcher.patternMatcherLogger));

    // `axis` is a compile-time `index` attribute on the softmax op.
    int64_t axis =
        cast<SoftmaxOp>(matcher.softmaxProducer).getAxisAttr().getInt();
    if (size_t(axis) != matcher.softmaxInput.getType().getRank() - 1)
      return Error("[MultiHeadAttentionEpilogueMatcher]: softmax axis is not "
                   "the last axis");

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

  GraphCompiler::Logger patternMatcherLogger;

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

  // Return the conditional such that masking should be applied in the "false"
  // branch.
  TensorValue getConditionalStatement(mlir::OpBuilder &builder) {
    if (attentionMatrixTrueBranch)
      return conditionalStatement;
    else
      return NotOp::create(builder, conditionalStatement.getLoc(),
                           conditionalStatement.getType(),
                           conditionalStatement);
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
      TensorValue notCond = NotOp::create(builder, loc, conditionalStatement);
      TensorValue notCondCasted = CastOp::create(
          builder, loc, notCond, bcastedMask.getType().getDtype());
      return MulOp::create(builder, loc, bcastedMask, notCondCasted);
    } else {
      TensorValue condCasted = CastOp::create(
          builder, loc, conditionalStatement, bcastedMask.getType().getDtype());
      return MulOp::create(builder, loc, bcastedMask, condCasted);
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

  GraphCompiler::Logger patternMatcherLogger;

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

    ASSERT_STREAM(
        matcher.causalMask.isError() || matcher.explicitMask.isError(),
        << "[MultiHeadAttentionMaskingMuxMatcher] expect only a single mask "
           "pattern to initial match.");

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

  // Has a mask in the form of an explicit addition
  bool hasExplicitMask() { return !explicitMask.isError(); }

  // Returns the explicit mask. Does not construct any new operators.
  // Returns Null if no explicit mask exists.
  TensorValue getExplicitMask() {
    if (!hasExplicitMask())
      return {};
    return explicitMask->getAttentionMask();
  }

  // Mask in the form of using mo.select to apply a boolean mask
  bool hasCausalMask() { return !causalMask.isError(); }

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
    for (size_t i = 1; i < masks.size(); ++i) {
      // statically cast incoming mask to the current shape
      // note this is always assumed to be safe because of `allowUnsafeBCast`
      TensorValue otherMask = masks[i];
      otherMask =
          BroadcastToOp::create(builder, mask.getLoc(), otherMask, mask);

      mask = AddOp::create(builder, mask.getLoc(), mask, otherMask);
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

  // For internal casts
  Operation *optCast1 = nullptr;

  MultiHeadAttentionNormalizationMatcher() = default;

public:
  static ErrorOr<MultiHeadAttentionNormalizationMatcher>
  match(Operation *producingOp, bool allowUnsafeBCast,
        bool allowInternalReshape) {
    GraphCompiler::Logger patternMatcherLogger;

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

    // Allow additional cast
    auto maybeCast = pOptionalCast(&matcher.optCast1, pNormalizedByMul);
    bool matchesMul =
        maybeCast.match(producingOp->getResult(0), &patternMatcherLogger);
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
    DeviceRefAttr device = attentionMatrix.getType().getDeviceRef();
    if (!scalarConstant) {
      // No normalization, return one as an equivalent
      TensorValue one = ConstantOp::create(builder, loc, 1.0, device);
      TensorType targetType =
          one.getType().cloneWith(attentionMatrix.getType().getDtype());
      TensorValue oneCasted = CastOp::create(builder, loc, targetType, one);
      return oneCasted;
    }

    if (isDiv) {
      TensorValue one = ConstantOp::create(builder, loc, 1.0, device);
      TensorType targetType =
          one.getType().cloneWith(scalarConstant.getType().getDtype());
      TensorValue oneCasted = CastOp::create(builder, loc, targetType, one);

      // Must cast from shape `[]` to `[1]` possibly.
      TensorValue oneBCast = Static::BroadcastToOp::create(
          builder, loc, scalarConstant.getType(), oneCasted);
      return DivOp::create(builder, loc, oneBCast, scalarConstant);
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
      ErrorOrSuccess normScalarOk =
          checkRank(scalarConstant,
                    "[MultiHeadAttentionNormalizationMatcher] normScalar", {0});
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

  GraphCompiler::Logger patternMatcherLogger;
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

/// A helper to match multi-headed attention and its variations
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

    // We need the attention mask to match the attention matrix shape
    if (maskingMatcher->hasAttentionMask()) {
      MOSH::ShapeAttr attentionMaskShape =
          cast<MOSH::ShapeAttr>(maskingMatcher->getAttentionMaskShape());
      MOSH::ShapeAttr attentionMatrixShape =
          normalizationMatcher->getAttentionMatrix()
              .getType()
              .getTypedShapeAttr();

      bool sameRank =
          attentionMaskShape.getRank() == attentionMatrixShape.getRank();
      bool differentRank = attentionMaskShape.getRank() == 4 &&
                           attentionMatrixShape.getRank() == 3;
      if (!sameRank && !differentRank)
        return Error("[MultiHeadAttentionPatternHelper] Attention mask only "
                     "supports being the same rank as attention matrix or "
                     "attention mask being rank 4 and the matrix rank 3.");
    }
    return success();
  }

  MultiHeadAttentionPatternHelper() = default;

public:
  static ErrorOr<MultiHeadAttentionPatternHelper>
  match(BatchMatmulOp finalOp, DeviceAwareRewriter &rewriter,
        bool allowUnsafeBCast, bool allowInternalReshape,
        bool debugMHAPattern) {
    MultiHeadAttentionPatternHelper matcher;
    matcher.builder = &rewriter.rewriter;

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
      std::string errMsgRoot(
          "[MultiHeadAttentionPatternHelper] Intermediate value " + name +
          ": ");

      if (!topOfPattern)
        return Error(errMsgRoot + Twine("is nullptr!"));

      if (!topOfPattern.hasOneUse()) {
        return Error(errMsgRoot + Twine("has more than one use!"));
      }

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

  bool hasExplicitMask() { return maskingMatcher->hasExplicitMask(); }
  TensorValue getExplicitMask() { return maskingMatcher->getExplicitMask(); }
  bool hasAttentionMask() { return maskingMatcher->hasAttentionMask(); }

  // layout B,H,s,D
  TensorValue getAttentionMask() {
    return maskingMatcher->getAttentionMask(*builder);
  }

  TensorValue getAttentionMatrix() {
    return normalizationMatcher->getAttentionMatrix();
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

struct MultiHeadAttentionPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, BatchMatmulOp> {
  bool allowUnsafeBCast, allowInternalReshape, debugMHAPattern;
  DeviceRefAttr targetDevice;
  MultiHeadAttentionPattern(MLIRContext *ctx, bool allowUnsafeBCast,
                            bool allowInternalReshape, bool debugMHAPattern,
                            DeviceRefAttr targetDevice)
      : DeviceAwareRewritePattern<OpRewritePattern, BatchMatmulOp>(ctx, 20),
        allowUnsafeBCast(allowUnsafeBCast),
        allowInternalReshape(allowInternalReshape),
        debugMHAPattern(debugMHAPattern), targetDevice(targetDevice) {}

  static void insertShapeGuard(DeviceAwareRewriter &rewriter, GraphOp graphOp,
                               TypedAttr beforeShape, TypedAttr afterShape,
                               Location loc, const Twine &checkName,
                               const Twine &errorMsg) {
    // Skip the guard if the shapes are equal at compile time
    if (beforeShape == afterShape)
      return;

    // Inserts a cpuDevice guard checking (beforeShape == afterShape)
    Type boolTy =
        KGEN::SIMDType::get(graphOp.getContext(), 1, KGEN::KGENDType::kBool);
    TypedAttr cmpAttr =
        KGEN::emitMLIROperationCall(MOSH::EqualOp::getOperationName(), {},
                                    {beforeShape, afterShape}, boolTy);
    KGEN::ParamDeclAttr condAttr = graphOp.getNewParamOfType(checkName, boolTy);
    rewriter.create<KGEN::ParamDeclareOp>(loc, condAttr, cmpAttr);

    // Add cpuDevice assertion into graph
    TypedAttr assertAttr = MO::AssertOp::emitStaticCall(
        KGEN::ParamDeclRefAttr::get(condAttr), errorMsg);
    auto chainTy = rewriter.getType<ChainType>();
    KGEN::ParamDeclAttr chainAttr =
        graphOp.getNewParamOfType(checkName, chainTy);
    rewriter.create<KGEN::ParamDeclareOp>(loc, chainAttr, assertAttr);
  }

  void guardUnsafeBCast(Static::BroadcastToOp unsafeBCast,
                        DeviceAwareRewriter &rewriter) const {
    // We expect all bcasts matched in the pattern with `allowUnsafeBCast`
    // to be no-ops (e.g. they don't change shape). This guards against this
    // condition by inserting cpuDevice checks against a potentially unsafe
    // bcast.
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

  // The matcher's for QKV returns values with the following layouts:
  // Q: BHSD
  // K: BHDS
  // V: BHSD
  //
  // However, some kernels expect all of these layouts to be `BSHD`.
  // Within most models with attention, tensors with the layout `BSHD` will be
  // transposed to the above. Therefore capturing this transformation in the
  // model is of great importance.
  //
  // Unfortunately, because we are importing from ONNX which has super loose
  // semantics around ops, the transpose may have random `static_reshapes`
  // and `static_broadcasts` sprinkled throughout. The pattern we support is
  //
  // BSHD_tensor --> transpose --> *static_reshape --> *static_broadcast
  //             --> Q/K/V tensor
  //
  // Where all the * are optional.
  //
  // We ultimately want to extract the BSHD tensor for these special kernels.
  // We also must make sure the intermediate `static_reshapes/broadcasts` are
  // no-ops and do not modify shape (since that is the assumption).
  //
  // In addition, we must account for the case that sometimes Q/K/V is rank 3
  // with the batch dimension elided.
  //
  // Finally, we are using this in a rewrite step, which by MLIR contracts means
  // we cannot fail a rewrite. If the user requests a BSHD tensor when there
  // is no parent transpose, we must still give it to them by inserting a
  // transpose.
  struct TransposeTensorHelper {
    static TensorValue backwardGetBSHDTensor(
        // indices for a transpose operator to move the layout of `start` to
        // BSHD e.g. {0, 2, 1, 3} implies `start` is BHSD
        ArrayRef<int64_t> transposeToBSHD, TensorValue start,
        DeviceAwareRewriter &rewriter, GraphOp graphOp) {
      // Stores all tensor types from intermediate reshapes/broadcasts.
      // These all need to be equal
      SmallVector<TensorType> equivalentShapes;
      TensorValue current = start;

      // Walk past any reshapes/bcasts we assume are no-ops
      while (current.getDefiningOp<Static::BroadcastToOp>() ||
             current.getDefiningOp<Static::ReshapeOp>()) {
        if (auto bcast = current.getDefiningOp<Static::BroadcastToOp>()) {
          current = bcast.getInput();
          equivalentShapes.push_back(bcast.getType());
        }
        if (auto reshape = current.getDefiningOp<Static::ReshapeOp>()) {
          current = reshape.getInput();
          equivalentShapes.push_back(reshape.getType());
        }
      }

      // If no transpose, still must return something valid for the kernel.
      auto transposeOp = current.getDefiningOp<TransposeOp>();
      if (!transposeOp)
        return backwardFallbackGetBSHDTensor(transposeToBSHD, start, rewriter);

      TensorValue tensorBSHD = transposeOp.getInput();

      // We must make validate our assumption that initial tensor given
      // permutation gives us BSHD layout, otherwise just re-transpose.
      SmallVector<TypedAttr> tensorStartDimsRank4Permuted =
          permute(getRank4Dims(start, rewriter), transposeToBSHD);
      SmallVector<TypedAttr> tensorBSHDDimsRank4 =
          getRank4Dims(tensorBSHD, rewriter);
      if (tensorStartDimsRank4Permuted != tensorBSHDDimsRank4)
        return backwardFallbackGetBSHDTensor(transposeToBSHD, start, rewriter);

      // Runtime Check: all reshapes/bcasts really do nothing to shape.
      for (TensorType equivalentShapeTy : equivalentShapes) {
        insertShapeGuard(
            rewriter, graphOp, equivalentShapeTy.getShapeAttr(),
            start.getType().getShapeAttr(), start.getLoc(),
            "MHA_QKV_TRANSPOSE_BCAST_OR_RESHAPE_CHECK",
            "MHA Pattern Detected unsafe broadcast/reshape. This is of great "
            "interest so please report!");
      }

      return tensorBSHD;
    }

    // Returns dimensions of value. If value is rank-3, assume batch dimension
    // was elided and pretends to add it back in as a dim of size `1`.
    static SmallVector<TypedAttr> getRank4Dims(TensorValue value,
                                               DeviceAwareRewriter &rewriter) {
      [[maybe_unused]] size_t rank = value.getType().getRank();
      ASSERT_STREAM(rank == 3 || rank == 4, << "Support only rank3/4 inputs.");

      SmallVector<TypedAttr> answer;
      auto curDims = value.getType().getTypedShapeAttr().getValues();
      if (curDims.size() == 3)
        answer.push_back(ShapeAttr::getDim(rewriter.getContext(), 1));

      answer.append(curDims.begin(), curDims.end());
      return answer;
    }

    // If we cannot find a parent, transpose, move from the current layout back
    // to BSHD.
    static TensorValue
    backwardFallbackGetBSHDTensor(ArrayRef<int64_t> transposeToBSHD,
                                  TensorValue start,
                                  DeviceAwareRewriter &rewriter) {
      // Pretend everything is rank 4 for simplicity
      SmallVector<TypedAttr> rank4Dim = getRank4Dims(start, rewriter);
      TensorValue startRank4 = rewriter.create<Static::ReshapeOp>(
          start.getLoc(), start.getType().cloneWith(rank4Dim), start);

      SmallVector<TypedAttr> expectedBSHDShape =
          permute(rank4Dim, transposeToBSHD);
      ArrayRef<TypedAttr> finalShape = expectedBSHDShape;
      auto permConst =
          rewriter.createHostConstant(start.getLoc(), transposeToBSHD);
      TensorValue transposedTensor = rewriter.create<TransposeOp>(
          start.getLoc(), startRank4.getType().cloneWith(finalShape),
          startRank4, permConst);

      // If we were rank-3 to begin with, chop off elided batch dimension
      if (start.getType().getRank() == 3)
        transposedTensor = rewriter.create<Static::ReshapeOp>(
            start.getLoc(), start.getType().cloneWith(finalShape.slice(1)),
            transposedTensor);

      return transposedTensor;
    }

    // Given a tensor given `backwardGetBSHDTensor`, find the grouped attention
    // weight tensor.
    //
    // Grouped attention is a parameter-compression technique which reuses the
    // same K and V weights across attention heads.
    //
    // Consider a Q @ K multiplication:
    //
    // mo.batch_matmul(%Q, %K) : (!mo.tensor<[B, H, S, D], f32>, !mo.tensor<[B,
    // H, D, S], f32>) -> !mo.tensor<[B, H, S, S], f32>
    //
    // Under grouped attention, we have g groups, and H = g * h. We match the
    // following pattern:
    //
    // %K_compressed: !mo.tensor<[B, h, D, S], f32>
    // %stg1 = mo.static.reshape(%K_compressed): !mo.tensor<[B, h, D, S], f32>
    // -> !mo.tensor<[B, h, 1, D, S], f32> %stg2 =
    // mo.static.broadcast_to(%stg1): !mo.tensor<[B, h, 1, D, S], f32> ->
    // !mo.tensor<[B, h, g, D, S], f32> %K_orig = mo.static.reshape(%stg2):
    // !mo.tensor<[B, h, g, D, S], f32> -> !mo.tensor<[B, H, D, S], f32> %K =
    // mo.transpose(%K_orig, %permutes): (!mo.tensor<[B, H, S, D], f32>, ...) ->
    // !mo.tensor<[B, H, D, S], f32>
    //
    // We want to extract `K_compressed`.
    //
    // Returns std::nullopt if we cannot grab the compressed tensor.
    static ErrorOr<TensorValue>
    extractGroupedAttentionTensorBSHD(TensorValue tensorBSHD,
                                      DeviceAwareRewriter &rewriter,
                                      GraphOp graphOp) {
      // Go through optional transposes and reshapes which do not matter
      TensorValue startTensor = tensorBSHD;

      // Dump identity permutation.
      SmallVector<int64_t> identityPermutation = {0, 1, 2, 3};

      // Keep track the layout of `startTensor` as we push through transposes.
      // We map the identity permutation to BSHD, e.g. H = 2.
      SmallVector<int64_t> curLayout = identityPermutation;
      while (true) {
        if (auto transposeOp = startTensor.getDefiningOp<TransposeOp>()) {
          startTensor = transposeOp.getInput();
          auto maybePerm = tryGetConstantIntVector(transposeOp.getPerm());
          if (!maybePerm.has_value())
            return Error("[extractGroupedAttentionTensor] transpose does not "
                         "have constant permutation.");

          // Get the input layout given the permutation
          curLayout = permuteReverse(curLayout, *maybePerm);
          continue;
        }

        auto reshapeOp = startTensor.getDefiningOp<Static::ReshapeOp>();
        if (reshapeOp &&
            reshapeOp.getInput().getType() == startTensor.getType()) {
          startTensor = reshapeOp.getInput();
          continue;
        }

        break;
      }

      auto reshape1 = startTensor.getDefiningOp<Static::ReshapeOp>();
      if (!reshape1)
        return Error(
            "[extractGroupedAttentionTensor] unable to get reshape for `orig`");

      TensorValue stg2 = reshape1.getInput();
      auto staticBcast = stg2.getDefiningOp<Static::BroadcastToOp>();
      if (!staticBcast)
        return Error("[extractGroupedAttentionTensor] unable to get broadcast "
                     "for `stg2`");

      TensorValue stg1 = staticBcast.getInput();
      auto reshape2 = stg1.getDefiningOp<Static::ReshapeOp>();
      if (!reshape2)
        return Error("[extractGroupedAttentionTensor] unable to get broadcast "
                     "for `stg1`");

      TensorValue compressedValue = reshape2.getInput();

      // We want to go back to curLayout --> identityPermutation.
      // So figure this out.
      SmallVector<int64_t> permutationBackToBSHD =
          solvePermutation(identityPermutation, curLayout);
      return backwardGetBSHDTensor(permutationBackToBSHD, compressedValue,
                                   rewriter, graphOp);
    }
  };

  void rewriteRootOp(BatchMatmulOp rootOp, DeviceAwareRewriter &rewriter,
                     MultiHeadAttentionPatternHelper &matcher,
                     bool isCPUDevice) const {
    // The matcher returns QKV with the following layouts:
    // Q: BHSD
    // K: BHDS
    // V: BHSD
    //
    // With possible elided batch dimension.
    TensorValue initQuery = matcher.getQuery();
    TensorValue initKey = matcher.getKeyT();
    TensorValue initValue = matcher.getValue();

    SmallVector<int64_t> permuteBHSD2BSHD = {0, 2, 1, 3};
    ArrayRef<int64_t> permuteBSHD2BHSD = permuteBHSD2BSHD; // cyclical

    SmallVector<int64_t> permuteBHDS2BSHD = {0, 3, 1, 2};
    auto permuteBSHD2BHSDValue =
        rewriter.createHostConstant(rootOp.getLoc(), permuteBSHD2BHSD);

    // However the kernel wants the layout `BSHD` which we get below
    // (this also handles elided batch dimensions gracefully).
    auto graphOp = rootOp->getParentOfType<MO::GraphOp>();
    TensorValue query = TransposeTensorHelper::backwardGetBSHDTensor(
        permuteBHSD2BSHD, initQuery, rewriter, graphOp);
    TensorValue key = TransposeTensorHelper::backwardGetBSHDTensor(
        permuteBHDS2BSHD, initKey, rewriter, graphOp);
    TensorValue value = TransposeTensorHelper::backwardGetBSHDTensor(
        permuteBHSD2BSHD, initValue, rewriter, graphOp);

    // For grouped query attention, try to extract the non-broadcasted K and V.
    // We expect K and V to be layout BSHD.
    auto maybeGroupedKey =
        TransposeTensorHelper::extractGroupedAttentionTensorBSHD(key, rewriter,
                                                                 graphOp);
    auto maybeGroupedValue =
        TransposeTensorHelper::extractGroupedAttentionTensorBSHD(
            value, rewriter, graphOp);
    if (!maybeGroupedKey.isError() && !maybeGroupedValue.isError()) {
      // Both must extracted both grouped weights to work properly
      key = maybeGroupedKey.takeValue();
      value = maybeGroupedValue.takeValue();
    }

    // Flash attention pattern includes the transposes from `mhaResult`
    // which is BHSD to BSHD.
    SmallVector<TypedAttr> initialOutputShape = {
        rootOp.getType().getTypedShapeAttr().getValues().begin(),
        rootOp.getType().getTypedShapeAttr().getValues().end()};
    if (initialOutputShape.size() == 3) {
      // add elided batch dimension if needed
      initialOutputShape.insert(initialOutputShape.begin(),
                                ShapeAttr::getDim(rewriter.getContext(), 1));
    }

    // Because output shape is rank4, to fit kernel spec, QKV must be rank 4
    // too. If rank 3 -- add back in elided batch dimension
    auto getRank4Tensor = [&](TensorValue val) {
      if (val.getType().getRank() == 4)
        return val;

      ArrayRef<TypedAttr> dims = val.getType().getTypedShapeAttr().getValues();
      ASSERT_STREAM(dims.size() == 3, << "Only expect rank 3!");
      SmallVector<TypedAttr> valShape{
          ShapeAttr::getDim(rewriter.getContext(), 1), dims[0], dims[1],
          dims[2]};
      TensorType newTy = val.getType().cloneWith(valShape);
      return rewriter.create<Static::ReshapeOp>(rootOp.getLoc(), newTy, val)
          .getResult();
    };
    query = getRank4Tensor(query);
    key = getRank4Tensor(key);
    value = getRank4Tensor(value);

    SmallVector<TypedAttr> mhaOutputShape =
        permute(initialOutputShape, permuteBHSD2BSHD);
    // Kernel expects FP32 scale
    auto normScalar = cast<TensorValue>(matcher.getNormScalar());
    TensorValue normScalarAsFP32 = rewriter.create<CastOp>(
        rootOp.getLoc(), normScalar.getType().cloneWith(DType::f32),
        normScalar);

    // The scale operand to all variants of flash attention are scalar
    // values, which must always be on the host. If they are not present on
    // the host, transfer the value.
    if (!isOnHost(normScalarAsFP32.getDefiningOp())) {
      ChainValue inChain =
          rewriter.create<ChainCreateOp>(rootOp.getLoc(), ValueRange{});

      normScalarAsFP32 = cast<TensorValue>(
          rewriter
              .create<TransferOp>(rootOp.getLoc(), normScalarAsFP32,
                                  getHostDevice(rootOp), inChain)
              .getResult());
    }

    // Replace the matched subgraph with the device/mask-specific composite op.
    // Each is a 1:1 wrapper around an existing flash-attention kernel, so the
    // op name alone selects the kernel (no dispatch in the graph compiler). The
    // GPU attention fusion always carries a mask (see runAdditionalGPUChecks).
    TensorType resultTy = rootOp.getType().cloneWith(mhaOutputShape);
    TensorValue mhaResult;
    if (isCPUDevice) {
      if (matcher.hasAttentionMask())
        mhaResult = cast<TensorValue>(
            rewriter
                .create<MO::Composite::MaskedFlashAttentionCPUOp>(
                    rootOp.getLoc(), resultTy, query, key, value,
                    matcher.getAttentionMask(), normScalarAsFP32)
                .getResult());
      else
        mhaResult = cast<TensorValue>(
            rewriter
                .create<MO::Composite::NoMaskFlashAttentionCPUOp>(
                    rootOp.getLoc(), resultTy, query, key, value,
                    normScalarAsFP32)
                .getResult());
    } else {
      mhaResult = cast<TensorValue>(
          rewriter
              .create<MO::Composite::MaskedFlashAttentionGPUOp>(
                  rootOp.getLoc(), resultTy, query, key, value,
                  matcher.getAttentionMask(), normScalarAsFP32)
              .getResult());
    }

    // The mhaResult should be shape BSHD, however the rooted op is layout BHSD.
    //
    // Try to go forward from the rooting op until we find a transpose.
    // Much like `backwardGetBSHDTensor`, except we go forward in uses.
    bool oneUseBMM = rootOp->hasOneUse();
    Operation *curOp = oneUseBMM ? *rootOp->getUsers().begin() : nullptr;

    // Single user ops between the `rootOp` and a future transpose. If we
    // find a transpose, we will replace the transpose with the kernel call.
    // Then, we must delete the chain of ops leading from `rootOp` to follow
    // the pattern rewriter contract (which states the root op must be
    // modified somehow).
    //
    // Includes the `rootOp`, does not include the transpose.
    SmallVector<Operation *> trailingOpsToTranspose = {rootOp};
    while (curOp && curOp->hasOneUse() &&
           isa<Static::ReshapeOp, Static::BroadcastToOp>(curOp)) {
      TensorValue result = cast<TensorValue>(curOp->getResult(0));
      if (result.hasOneUse()) {
        trailingOpsToTranspose.push_back(curOp);
        curOp = *result.getUsers().begin();
      }
    }

    // We only want to take the final transpose as part of our pattern if we
    // know the final shape is potentially as we expect. Specifically we want
    //
    // The transpose to turn the layout from BHSD --> BSHD.
    //
    // In practice this is impossible to always prove so for many models
    // we assume the transpose is to BSHD and insert a dynamic cpuDevice check
    // later in case this is violated.
    //
    // However if we can prove this is false at compile time, e.g. if the shapes
    // are all integral, we can just check directly and avoid this optimizing
    // pass instead.
    //
    // This is a bit odd and may be a good target of spring cleaning to see
    // if we need this anymore.
    auto finalTranspose = dyn_cast<TransposeOp>(curOp);
    auto transposeIsStaticallyWrong = [&]() {
      ArrayRef<TypedAttr> transposeShapeAttr =
          finalTranspose.getType().getTypedShapeAttr().getValues();
      ArrayRef<TypedAttr> finalResultShapeAttr =
          mhaResult.getType().getTypedShapeAttr().getValues();
      return llvm::any_of(llvm::zip(transposeShapeAttr, finalResultShapeAttr),
                          [&](std::tuple<TypedAttr, TypedAttr> values) {
                            auto &[a, b] = values;
                            return MO::TensorType::isIntegerDimParam(a) &&
                                   MO::TensorType::isIntegerDimParam(b) &&
                                   a != b;
                          });
    };

    if (finalTranspose && !transposeIsStaticallyWrong() && oneUseBMM) {
      // The idea is we now have a chain of ops like
      // *rootOp (BHSD) --> *bcast/reshape --> *bcast/reshape
      //                --> transpose (BSHD)
      //
      // Where the * appear in `oneUseOps`. We now want to replace the final
      // transpose with the result of the pattern and delete all the previous
      // operations in `oneUseOps`.

      // Runtime Check 1: make sure all the ops in `oneUseOps` have the same
      // shape. This guarantees the bcasts/reshapes we collected do nothing.
      // Note we allow the elision of batch dimension still.
      TypedAttr targetShape = ShapeAttr::get(
          TransposeTensorHelper::getRank4Dims(rootOp.getResult(), rewriter),
          MOSH::ShapeType::get(rewriter.getContext()));
      for (Operation *op : trailingOpsToTranspose) {
        auto curValue = cast<TensorValue>(op->getResult(0));
        TypedAttr actualShape = ShapeAttr::get(
            TransposeTensorHelper::getRank4Dims(curValue, rewriter),
            MOSH::ShapeType::get(rewriter.getContext()));
        insertShapeGuard(
            rewriter, graphOp, targetShape, actualShape, rootOp.getLoc(),
            "MHA_ROOT_TRANSPOSE_BCAST_OR_RESHAPE_CHECK",
            "MHA Pattern Detected unsafe reshapes/bcasts. This is of great "
            "interest so please report!");
      }

      // Runtime Check 2: make sure the transpose does go to BSHD.
      insertShapeGuard(rewriter, graphOp,
                       finalTranspose.getType().getShapeAttr(),
                       mhaResult.getType().getShapeAttr(), rootOp->getLoc(),
                       "MHA_ROOT_TRANSPOSE_BSHD_CHECK",
                       "MHA Pattern Detected final transpose does not go to "
                       "BSHD layout. This is of great "
                       "interest so please report!");

      // Replace ops, and delete the rest.
      rewriter.replaceAllOpUsesWith(finalTranspose, mhaResult);
      rewriter.eraseOp(finalTranspose);
      for (Operation *op : llvm::reverse(trailingOpsToTranspose)) {
        rewriter.eraseOp(op);
      }
    } else {
      // There is not a transpose or we can not otherwise replace the BMM
      // operation.
      //
      // Instead we must manually add a transpose from BSHD --> BHSD.
      //
      // Insert a final transpose to move from BSHD --> BHSD.
      // This will be canonicalized away in case of adjacent transposes.
      SmallVector<TypedAttr> finalOutputShape =
          permute(mhaResult.getType().getTypedShapeAttr().getValues(),
                  permuteBHSD2BSHD);
      TensorType finalTy = mhaResult.getType().cloneWith(finalOutputShape);
      auto finalOp = rewriter.create<TransposeOp>(
          mhaResult.getLoc(), finalTy, mhaResult, permuteBSHD2BHSDValue);

      rewriter.replaceAllOpUsesWith(rootOp, finalOp);
    }
  }

  // Check additional constraints on rewriting MHA operations on GPU, and
  // extract information as needed.
  ErrorOrSuccess runAdditionalGPUChecks(
      ErrorOr<MultiHeadAttentionPatternHelper> &matcher) const {

    // CHECK 1: Must be rank 3/4 q/k/v only. Note rank(q) == rank(k) ==
    // rank(v)
    if (matcher->getQuery().getType().getRank() != 4 &&
        matcher->getQuery().getType().getRank() != 3)
      return Error("GPU Kernel expects rank 3 or 4 QKV.");

    // CHECK 2: D == 128 and static H due to kernel limitation.
    // TODO: lift to static D once kernel is generalized.
    // Query is BHSD, Key is BHDS and Value is BHSD.
    // If rank 3, B is elided.
    // Note due to bcasts, only some of QKV might have static int dimensions.
    // The rest may be parameters;
    bool isRank3 = matcher->getQuery().getType().getRank() == 3;
    int64_t dimD = -1;
    dimD = std::max(dimD, matcher->getQuery().getType().getIntDim(3 - isRank3));
    dimD = std::max(dimD, matcher->getKeyT().getType().getIntDim(2 - isRank3));
    dimD = std::max(dimD, matcher->getValue().getType().getIntDim(3 - isRank3));

    SmallVector<int64_t> dimHs{
        matcher->getQuery().getType().getIntDim(1 - isRank3),
        matcher->getKeyT().getType().getIntDim(1 - isRank3),
        matcher->getValue().getType().getIntDim(1 - isRank3),
    };
    bool isDimHStatic =
        std::any_of(dimHs.begin(), dimHs.end(), [](int64_t value) {
          return value != mlir::ShapedType::kDynamic;
        });

    // Original guard was `dimD != 128 || !isDimHStatic` to restrict the GPU
    // MHA fusion to Llama2's head_dim. This was blocking GPT-2 124M fused
    // attention. Relax the guard to add dimD != 64.
    if ((dimD != 64 && dimD != 128) || !isDimHStatic)
      return Error("Right now must prove (D == 128 || D == 64) && static H.");

    // CHECK 3: The GPU kernel requires an attention mask tensor.
    if (!matcher->hasAttentionMask())
      return Error("GPU Kernel expects attention mask tensor.");

    return success();
  }

  LogicalResult
  deviceAwareMatchAndRewrite(BatchMatmulOp op,
                             DeviceAwareRewriter &rewriter) const override {
    DeviceRefAttr device = op.getInputA().getType().getDeviceRef();
    if (!device)
      device = getTargetDevice(op, targetDevice);

    // Just assume we want to use the CPU kernel only if the default device
    // is cpu (e.g. if the default was CUDA, run on CUDA).
    //
    // That way avoid device logic for now.
    if (device && device.getLabel() != M::kCPULabel &&
        device.getLabel() != M::kGPULabel)
      return rewriter.notifyMatchFailure(
          op->getLoc(), "Only support CPU and CUDA for this pattern.");

    // if not CPU, assume CUDA
    bool isCPUDevice = !device || device.getLabel() == M::kCPULabel;

    // Matching portion of pattern.
    ErrorOr<MultiHeadAttentionPatternHelper> matcher =
        MultiHeadAttentionPatternHelper::match(op, rewriter, allowUnsafeBCast,
                                               allowInternalReshape,
                                               debugMHAPattern);
    if (matcher.isError())
      return rewriter.notifyMatchFailure(op.getLoc(), matcher.getError());

    // TODO: The tail logic of rewriteRootOp appends a rank 4 transpose op if no
    // transpose op already exists, but this does not work if QKV is rank 3.
    // Disable support for rank 3 tensors until the flow can be fixed.
    if (matcher->getQuery().getType().getRank() != 4)
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "Supports rank 4 QKV only");

    // The GPU kernel has some additional checks and needs an extracted rank3 op
    if (!isCPUDevice) {
      ErrorOrSuccess gpuCheck = runAdditionalGPUChecks(matcher);
      if (gpuCheck.isError())
        return rewriter.notifyMatchFailure(op.getLoc(), gpuCheck.getError());
    }

    // Rewrite portion of pattern. Note: it is unsafe to now abort rewrites in
    // the middle of the pattern so we must be 100% sure the rewrite will
    // succeed now.

    // If we have unsafe broadcasts, we want to add cpuDevice guards against
    // them.
    for (Static::BroadcastToOp unsafeBCast : matcher->getUnsafeBroadcasts())
      guardUnsafeBCast(unsafeBCast, rewriter);

    rewriteRootOp(op, rewriter, *matcher, isCPUDevice);

    return mlir::success();
  }
};

struct RMSNormFusedResidualAddPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern,
                                           MO::ReduceRMSNormOp> {
  RMSNormFusedResidualAddPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 3) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::ReduceRMSNormOp op,
                             MO::DeviceAwareRewriter &rewriter) const override {
    // Match the pattern: rms_norm(rms_norm(input) + residual_input)
    // This means we're looking for a RMSNormOp whose input is the result of
    // (rms_norm(input) + residual_input)

    // Look for an add operation feeding into this rms_norm's input
    auto addOp = op.getInput().getDefiningOp<MO::AddOp>();
    if (!addOp)
      return rewriter.notifyMatchFailure(
          op, "first operand not from add operation");

    // One of the add operands should be the result of an inner rms_norm
    if (addOp.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "add operation expects 2 operands");

    Value innerRmsNormResult = addOp.getOperand(0);
    Value residualInput = addOp.getOperand(1);
    auto innerRmsNorm = innerRmsNormResult.getDefiningOp<MO::ReduceRMSNormOp>();
    if (!innerRmsNorm) {
      innerRmsNormResult = addOp.getOperand(1);
      residualInput = addOp.getOperand(0);
      innerRmsNorm = innerRmsNormResult.getDefiningOp<MO::ReduceRMSNormOp>();
      if (!innerRmsNorm)
        return rewriter.notifyMatchFailure(op, "inner rms_norm not found");
    }

    // Extract the original input fed into the inner rms_norm
    Value originalInput = innerRmsNorm.getInput();

    // Extract parameters from outer rms_norm
    Value gamma2 = op.getWeight();
    Value eps2 = op.getEpsilon();
    Value weightOffset2 = op.getWeightOffset();
    // Use outer op's multiply_before_cast (inner assumed to match outer)
    BoolAttr multiplyBeforeCast = op.getMultiplyBeforeCastAttr();

    // Extract parameters from inner rms_norm
    Value gamma1 = innerRmsNorm.getWeight();
    Value eps1 = innerRmsNorm.getEpsilon();
    Value weightOffset1 = innerRmsNorm.getWeightOffset();

    // Check device consistency
    if (!rewriter.addOps({addOp, innerRmsNorm})) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    // The fused operation returns two outputs: [output, intermediate]
    // where output = rms_norm(rms_norm(input) + residual)
    // and intermediate = rms_norm(input) + residual
    auto fusedOp =
        rewriter.create<MO::Composite::ReduceRMSNormFusedResidualAddOp>(
            op.getLoc(), op.getResult().getType(), addOp.getResult().getType(),
            originalInput, residualInput, gamma1, gamma2, eps1, eps2,
            weightOffset1, weightOffset2, multiplyBeforeCast.getValue(),
            op.getOutputParamDecls());

    // Replace the outer rms_norm with the first output of the fused operation
    rewriter.replaceOp(op, fusedOp.getOutput());

    // Replace all consumers of the add operation with the second output of the
    // fused operation
    rewriter.replaceAllUsesWith(addOp.getResult(), fusedOp.getIntermediate());

    return mlir::success();
  }
};

// Fuses the single-norm residual-add + RMSNorm boundary
// `rms_norm(add(a, b))` into one `mo.composite.rms_norm_residual_add` launch,
// emitting both the normalized output and the `a + b` intermediate (so the
// pre-add residual carried to the next block reuses the fused op's second
// output instead of a separate add). This is the canonical transformer/mamba
// pre-norm boundary. It is intentionally distinct from
// `RMSNormFusedResidualAddPattern` (which handles the double-norm topology
// `rms_norm(rms_norm(x) + r)`); this pattern deliberately declines that case
// and runs at lower benefit so the double-norm pattern wins where both apply.
struct RMSNormResidualAddPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern,
                                           MO::ReduceRMSNormOp> {
  RMSNormResidualAddPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::ReduceRMSNormOp op,
                             MO::DeviceAwareRewriter &rewriter) const override {
    // Match `rms_norm(add(a, b))`.
    auto addOp = op.getInput().getDefiningOp<MO::AddOp>();
    if (!addOp)
      return rewriter.notifyMatchFailure(
          op, "rms_norm input not from an add operation");
    if (addOp.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "add operation expects 2 operands");

    // Leave the double-norm topology `rms_norm(rms_norm(x) + r)` to
    // RMSNormFusedResidualAddPattern (higher benefit): if either add operand is
    // itself an inner rms_norm, decline here.
    Value lhs = addOp.getOperand(0);
    Value rhs = addOp.getOperand(1);
    if (lhs.getDefiningOp<MO::ReduceRMSNormOp>() ||
        rhs.getDefiningOp<MO::ReduceRMSNormOp>())
      return rewriter.notifyMatchFailure(
          op, "double-norm topology handled by RMSNormFusedResidualAddPattern");

    // Check device consistency across the matched ops.
    if (!rewriter.addOps({addOp}))
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");

    Value gamma = op.getWeight();
    Value eps = op.getEpsilon();
    Value weightOffset = op.getWeightOffset();
    BoolAttr multiplyBeforeCast = op.getMultiplyBeforeCastAttr();

    // The fused op returns [output, intermediate] where
    //   intermediate = a + b
    //   output       = rms_norm(intermediate, gamma, eps, weight_offset)
    auto fusedOp = rewriter.create<MO::Composite::RMSNormResidualAddOp>(
        op.getLoc(), op.getResult().getType(), addOp.getResult().getType(), lhs,
        rhs, gamma, eps, weightOffset, multiplyBeforeCast.getValue(),
        op.getOutputParamDecls());

    // Replace the rms_norm with the normalized output, and rewire the add's
    // other consumers (the carried residual) to the intermediate output.
    rewriter.replaceOp(op, fusedOp.getOutput());
    rewriter.replaceAllUsesWith(addOp.getResult(), fusedOp.getIntermediate());

    return mlir::success();
  }
};

struct RMSNormFusedQuantizeDynamicScaledFP8Pattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern, MO::CustomOp> {
  DeviceRefAttr targetDevice;
  RMSNormFusedQuantizeDynamicScaledFP8Pattern(MLIRContext *ctx,
                                              DeviceRefAttr targetDevice)
      : DeviceAwareRewritePattern(ctx, 3), targetDevice(targetDevice) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::CustomOp op,
                             MO::DeviceAwareRewriter &rewriter) const override {
    if (op.getSymbol() != "mo.quantize_dynamic_scaled_float8")
      return rewriter.notifyMatchFailure(op, "not a quantize operation");
    if (op.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected quantize operand count");

    DeviceRefAttr device = getTargetDevice(op, targetDevice);
    if (!device || device.getLabel() != M::kGPULabel)
      return rewriter.notifyMatchFailure(op, "fused op requires GPU");

    TensorValue input = dyn_cast<TensorValue>(op.getOperand(0));
    if (!input.hasOneUse())
      return rewriter.notifyMatchFailure(op, "input has multiple uses");

    auto rmsNormOp = input.getDefiningOp<MO::ReduceRMSNormOp>();
    if (!rmsNormOp)
      return rewriter.notifyMatchFailure(op, "input not a rms_norm operation");

    TensorType inputTy = input.getType();
    if (!inputTy.hasRank())
      return rewriter.notifyMatchFailure(op, "missing input rank");
    size_t rank = inputTy.getRank();
    if (rank == 0)
      return rewriter.notifyMatchFailure(op, "unexpected input rank");

    DictionaryAttr opParams = op.getParameters();
    if (opParams.size() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "missing or unexpected parameters");

    Attribute groupSizeAttr = opParams.get("group_size_or_per_token");
    if (!groupSizeAttr)
      return rewriter.notifyMatchFailure(op, "missing group_size_or_per_token");
    if (auto groupSizeIntAttr = dyn_cast<IntegerAttr>(groupSizeAttr)) {
      // Verify that the token size matches the inner dimension of the input
      // operand.
      int64_t groupSize = groupSizeIntAttr.getInt();
      if (groupSize != -1 && groupSize != inputTy.getIntDim(rank - 1))
        return rewriter.notifyMatchFailure(op,
                                           "tokenwise quantization required");
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "wrong group_size_or_per_token type");
    }

    if (!rmsNormOp.getMultiplyBeforeCast())
      return rewriter.notifyMatchFailure(op, "multiply_before_cast required");

    // Otherwise only fuse the RMSNorm + Quant into a dedicated composite op.
    SmallVector<Value> fusedOperands = {
        rmsNormOp.getInput(),        // rms_norm.input
        rmsNormOp.getWeight(),       // rms_norm.gamma
        rmsNormOp.getEpsilon(),      // rms_norm.epsilon
        rmsNormOp.getWeightOffset(), // rms_norm.weight_offset
        op.getOperand(1),            // quantize.scale_ub
    };

    rewriter.replaceOpWithNewOp<
        MO::Composite::RMSNormFusedQuantizeDynamicScaledFP8Op>(
        op, op.getResultTypes(), fusedOperands);

    return mlir::success();
  }
};

// Fuses `mo.reduce.rms_norm -> mo.quantize.dynamic.block.scaled` (MXFP8) into a
// single `mo.composite.rms_norm_fused_quantize_dynamic_block_scaled`. The
// block-scaled analog of `RMSNormFusedQuantizeDynamicScaledFP8Pattern`: the
// fused kernel computes the RMS norm and quantizes the bf16-rounded normed
// value to MXFP8 (E4M3 values + rank-5 E8M0 block scales) in one pass, dropping
// the HBM round-trip of the bf16 normed activation.
struct RMSNormFusedQuantizeDynamicBlockScaledPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern, MO::CustomOp> {
  DeviceRefAttr targetDevice;
  RMSNormFusedQuantizeDynamicBlockScaledPattern(MLIRContext *ctx,
                                                DeviceRefAttr targetDevice)
      : DeviceAwareRewritePattern(ctx, 3), targetDevice(targetDevice) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::CustomOp op,
                             MO::DeviceAwareRewriter &rewriter) const override {
    if (op.getSymbol() != "mo.quantize.dynamic.block.scaled")
      return rewriter.notifyMatchFailure(op, "not a block-scaled quantize op");
    // The block-scaled quantize takes (input, tensor_sf) and returns
    // (values, scales).
    if (op.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected quantize operand count");
    if (op.getNumResults() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "unexpected quantize result count");

    DeviceRefAttr device = getTargetDevice(op, targetDevice);
    if (!device || device.getLabel() != M::kGPULabel)
      return rewriter.notifyMatchFailure(op, "fused op requires GPU");

    // Fire only for the MXFP8 configuration the fused kernel implements: E4M3
    // packed values + E8M0 block scales. This precisely excludes MXFP4 (which
    // shares the E8M0 scale but packs values into uint8) and NVFP4 (E4M3
    // scales).
    auto valuesTy = dyn_cast<TensorType>(op.getResult(0).getType());
    auto scaleTy = dyn_cast<TensorType>(op.getResult(1).getType());
    if (!valuesTy || !scaleTy)
      return rewriter.notifyMatchFailure(op, "non-tensor quantize results");
    if (valuesTy.getDtype() != M::DType::f8e4m3fn)
      return rewriter.notifyMatchFailure(op, "values not E4M3 (not MXFP8)");
    if (scaleTy.getDtype() != M::DType::f8e8m0fnu)
      return rewriter.notifyMatchFailure(op, "scales not E8M0 (not MXFP8)");

    // Confirm the E8M0 block size is 32 (MXFP8). The fused kernel forwards this
    // to the composite op as a compile-time parameter.
    DictionaryAttr opParams = op.getParameters();
    auto sfAttr =
        opParams ? dyn_cast_or_null<IntegerAttr>(opParams.get("SF_VECTOR_SIZE"))
                 : nullptr;
    if (!sfAttr)
      return rewriter.notifyMatchFailure(op, "missing SF_VECTOR_SIZE");
    if (sfAttr.getInt() != 32)
      return rewriter.notifyMatchFailure(op, "SF_VECTOR_SIZE must be 32");

    // Single-consumer guard: the fused op does NOT emit the bf16 normed
    // tensor, so the norm result must feed only this quantize. If the norm
    // also feeds other ops, fusing would drop a needed value.
    TensorValue input = dyn_cast<TensorValue>(op.getOperand(0));
    if (!input || !input.hasOneUse())
      return rewriter.notifyMatchFailure(op, "quant input has multiple uses");

    auto rmsNormOp = input.getDefiningOp<MO::ReduceRMSNormOp>();
    if (!rmsNormOp)
      return rewriter.notifyMatchFailure(op, "input not a rms_norm operation");

    // The fused kernel hardcodes multiply_before_cast = true.
    if (!rmsNormOp.getMultiplyBeforeCast())
      return rewriter.notifyMatchFailure(op, "multiply_before_cast required");

    // Build the fused composite op. The `tensor_sf` operand (identity for
    // MXFP8) is dropped: the fused kernel does pure block scaling. `epsilon`
    // and `weight_offset` are threaded through unchanged, preserving semantics
    // for any weight_offset (e.g. the Gemma-style 1.0 the model uses). The
    // quantize result types (E4M3 values + rank-5 E8M0 scales) are reused
    // verbatim, so the downstream block-scaled GEMM is a drop-in consumer.
    SmallVector<Value> fusedOperands = {
        rmsNormOp.getInput(),        // rms_norm.input
        rmsNormOp.getWeight(),       // rms_norm.gamma
        rmsNormOp.getEpsilon(),      // rms_norm.epsilon
        rmsNormOp.getWeightOffset(), // rms_norm.weight_offset
    };

    // Read what we need from `op` before replacing it: replaceOpWithNewOp
    // erases `op`, so touching it afterwards (e.g. `op.getContext()`) is a
    // use-after-free. The attribute `sfAttr` is context-owned and outlives the
    // op, so caching its value is unnecessary.
    MLIRContext *context = op.getContext();
    auto fused = rewriter.replaceOpWithNewOp<
        MO::Composite::RMSNormFusedQuantizeDynamicBlockScaledOp>(
        op, op.getResultTypes(), fusedOperands);
    // Forward the block size to the kernel's `SF_VECTOR_SIZE` compile-time
    // parameter. Carried as a discardable attribute (matching the kernel param
    // name) rather than an ODS operand/attribute so the composite op stays a
    // clean 1:1 wrapper like the FP8 variant; `MOToMOGGLowering` forwards every
    // integer op attribute to the kernel by name.
    mlir::Builder builder(context);
    fused->setAttr("SF_VECTOR_SIZE",
                   builder.getI64IntegerAttr(sfAttr.getInt()));

    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Shared helpers used by both the distributed and bundled allreduce-add-
// rms_norm-quant-fp8 pattern matchers below.
//===----------------------------------------------------------------------===//

/// Finds an `mo.add` consumer of `val` that uses `val` on the RHS.  On success,
/// populates `residualOp` with the add op and `residualValue` with its LHS
/// operand (the "residual" side).  The RHS-only match avoids picking the wrong
/// operand when both inputs come from the same allreduce.
static bool matchResidualOp(Value val, MO::AddOp &residualOp,
                            Value &residualValue) {
  for (Operation *user : val.getUsers()) {
    auto addOp = dyn_cast<MO::AddOp>(user);
    // The pattern must match exactly the rhs operand for the AllReduce.
    // Otherwise we match suboptimal ops when subgraphs are disabled.
    // (The Add has two AllReduce operands, and we match with the wrong one).
    // TODO: This is very brittle, though this should be fixed once we switch
    // to mo.parallel.
    if (!addOp || addOp.getRhsInput() != val)
      continue;
    residualValue = addOp.getLhsInput();
    residualOp = addOp;
    return true;
  }
  return false;
}

/// Finds an `mo.reduce.rms_norm` consumer of `val` and returns it via
/// `rmsNormOp`.
static bool matchRMSNormOp(Value val, MO::ReduceRMSNormOp &rmsNormOp) {
  for (Operation *user : val.getUsers()) {
    auto op = dyn_cast<MO::ReduceRMSNormOp>(user);
    if (!op)
      continue;
    rmsNormOp = op;
    return true;
  }
  return false;
}

/// Finds an `mo.custom` consumer of `val` whose symbol equals `customOpName`.
static bool matchCustomOp(Value val, MO::CustomOp &customOp,
                          StringRef customOpName) {
  for (Operation *user : val.getUsers()) {
    auto op = dyn_cast<MO::CustomOp>(user);
    if (!op || op.getSymbol() != customOpName)
      continue;
    customOp = op;
    return true;
  }
  return false;
}

/// Validates the fused `allreduce + add + rms_norm + quantize_fp8` chain
/// ending at `quantizeOp`.  `inputTy` is the tensor type entering the add op
/// (i.e. the allreduce output on the relevant device).  Returns false unless:
///   - the quantize op reads `rmsNormOp.getResult()` as operand 0,
///   - `group_size_or_per_token` is either -1 or matches the inner dim,
///   - the rms_norm op reads `residualOp.getResult()` and has
///     `multiply_before_cast`.
static bool verifyFusedAllReducePath(TensorType inputTy, MO::AddOp residualOp,
                                     MO::ReduceRMSNormOp rmsNormOp,
                                     MO::CustomOp quantizeOp) {
  if (quantizeOp.getOperand(0) != rmsNormOp.getResult())
    return false;
  if (!inputTy.hasRank())
    return false;
  size_t rank = inputTy.getRank();
  DictionaryAttr quantizeOpParams = quantizeOp.getParameters();
  Attribute groupSizeAttr = quantizeOpParams.get("group_size_or_per_token");
  if (!groupSizeAttr)
    return false;
  auto groupSizeIntAttr = dyn_cast<IntegerAttr>(groupSizeAttr);
  if (!groupSizeIntAttr)
    return false;
  int64_t groupSize = groupSizeIntAttr.getInt();
  if (groupSize != -1 && groupSize != inputTy.getIntDim(rank - 1))
    return false;
  if (rmsNormOp.getInput() != residualOp.getResult())
    return false;
  if (!rmsNormOp.getMultiplyBeforeCast())
    return false;
  return true;
}

struct FuseAllReduceAddRMSNormQuantFP8PatternMatcher {
public:
  FuseAllReduceAddRMSNormQuantFP8PatternMatcher(MO::AllReduceSumOp allReduceOp,
                                                DeviceRefAttr targetDevice)
      : allReduceOp(allReduceOp), targetDevice(targetDevice),
        builder(allReduceOp->getContext()) {}

  void apply() {
    unsigned devicesCount = allReduceOp.getOutputs().size();
    // Start from AllReduce outputs to match everything
    for (Value out : allReduceOp.getOutputs()) {
      if (!matchFusedAllReducePath(cast<TensorValue>(out)))
        return;
    }
    if (residualOps.size() != devicesCount)
      return;

    // Let's build the new op
    builder.setInsertionPointAfter(allReduceOp);

    // Prepare the operands
    llvm::SmallVector<Type> outputTypes;
    llvm::SmallVector<Type> outScaleTypes;
    llvm::SmallVector<Type> outResidualTypes;
    llvm::SmallVector<Value> residualOperands;
    llvm::SmallVector<Value> gammaOperands;
    llvm::SmallVector<Value> epsilonOperands;
    llvm::SmallVector<Value> weightOffsetOperands;
    llvm::SmallVector<Value> scaleUbOperands;
    for (Value out : allReduceOp.getOutputs()) {
      DeviceRefAttr device = cast<TensorValue>(out).getType().getDeviceRef();
      MO::ReduceRMSNormOp rmsNormOp = rmsNormOps.at(device);
      MO::CustomOp quantizeOp = quantizeOps.at(device);
      outputTypes.push_back(quantizeOp->getResultTypes()[0]);
      outScaleTypes.push_back(quantizeOp->getResultTypes()[1]);
      outResidualTypes.push_back(residualOps.at(device).getResult().getType());
      residualOperands.push_back(residualValues.at(device));
      gammaOperands.push_back(rmsNormOp.getWeight());
      epsilonOperands.push_back(rmsNormOp.getEpsilon());
      weightOffsetOperands.push_back(rmsNormOp.getWeightOffset());
      scaleUbOperands.push_back(quantizeOp.getOperand(1));
    }

    // Build the op

    auto allReduceFusedOp =
        MO::Composite::AllReduceAddRMSNormQuantFP8Op::create(
            builder, allReduceOp.getLoc(),
            /*output=*/outputTypes,
            /*outScale=*/outScaleTypes,
            /*outResidual=*/outResidualTypes,
            /*outChain=*/allReduceOp.getOutChain().getType(),
            /*inputs=*/allReduceOp.getInputs(),
            /*signalBuffers=*/allReduceOp.getSignalBuffers(),
            /*residual=*/residualOperands,
            /*gamma=*/gammaOperands,
            /*epsilon=*/epsilonOperands,
            /*weightOffset=*/weightOffsetOperands,
            /*scaleUb=*/scaleUbOperands,
            /*inChain=*/allReduceOp.getInChain());

    // Replace the original outputs with the new op.
    // Replace the output chain.
    allReduceOp.getOutChain().replaceAllUsesWith(
        allReduceFusedOp.getOutChain());

    for (auto [i, out] : llvm::enumerate(allReduceOp.getOutputs())) {
      DeviceRefAttr device = cast<TensorValue>(out).getType().getDeviceRef();
      MO::AddOp residualOp = residualOps.at(device);
      MO::ReduceRMSNormOp rmsNormOp = rmsNormOps.at(device);
      MO::CustomOp quantizeOp = quantizeOps.at(device);
      // Replace the intermediate result.
      residualOp.getResult().replaceAllUsesExcept(
          allReduceFusedOp.getOutResidual()[i], rmsNormOp);

      // Replace the quantized op outputs
      quantizeOp.getResult(0).replaceAllUsesWith(
          allReduceFusedOp.getOutput()[i]);
      quantizeOp.getResult(1).replaceAllUsesWith(
          allReduceFusedOp.getOutScale()[i]);
    }
  }

private:
  MO::AllReduceSumOp allReduceOp;
  DeviceRefAttr targetDevice;
  OpBuilder builder;
  llvm::DenseMap<DeviceRefAttr, MO::AddOp> residualOps;
  llvm::DenseMap<DeviceRefAttr, MO::ReduceRMSNormOp> rmsNormOps;
  llvm::DenseMap<DeviceRefAttr, MO::CustomOp> quantizeOps;
  llvm::DenseMap<DeviceRefAttr, Value> residualValues;

  bool matchFusedAllReducePath(TensorValue allReduceOut) {
    DeviceRefAttr device = allReduceOut.getType().getDeviceRef();
    ASSERT_STREAM(device,
                  "all reduce outputs must be assigned to a specific device");
    // Check if we haven't already match the ops of this device.
    if (residualOps.contains(device))
      return false;

    MO::AddOp residualOp;
    MO::ReduceRMSNormOp rmsNormOp;
    MO::CustomOp quantizeOp;
    Value residualValue;

    // Match add + rms_norm + custom["mo.quantize_dynamic_scaled_float8"]
    if (!matchResidualOp(allReduceOut, residualOp, residualValue))
      return false;

    if (!matchRMSNormOp(residualOp.getResult(), rmsNormOp))
      return false;

    if (!matchCustomOp(rmsNormOp.getResult(), quantizeOp,
                       "mo.quantize_dynamic_scaled_float8"))
      return false;

    if (!verifyFusedAllReducePath(allReduceOut.getType(), residualOp, rmsNormOp,
                                  quantizeOp))
      return false;

    residualOps[device] = residualOp;
    rmsNormOps[device] = rmsNormOp;
    quantizeOps[device] = quantizeOp;
    residualValues[device] = residualValue;
    return true;
  }
};

//===----------------------------------------------------------------------===//
// FuseParallelBundledAllReduceAddRMSNormQuantFP8PatternMatcher
//
// Cross-boundary parallel-region matcher.  Given
//
//   mo.parallel(%arg) in (%inBundle : ...) buffers(...) chain(%c) -> ... {
//     ... = mo.bundled.expand(%arg) ...
//     %out, %ch_out = mo.bundled.allreduce.sum(...)
//     mo.yield %out
//   }
//   %r0..N = mo.tensor.unbundle(%bundle_result)
//   // Per-device chain, OUTSIDE the parallel:
//   %add_i  = mo.add(%residual_i, %r_i)
//   %norm_i = mo.reduce.rms_norm(%add_i, %gamma_i, %eps, %wo)
//   %q_i:2  = mo.custom { "mo.quantize_dynamic_scaled_float8" }
//               (%norm_i, %scale_ub)
//
// rebuilds the parallel with two extra bundle inputs (residuals, gammas)
// and three bundle results (FP8 outputs, scales, post-add residuals),
// replacing the in-body `bundled.allreduce.sum` with the fused
// `bundled.allreduce_add_rms_norm_quant_fp8` op.
//
// Per-launch shared scalars (epsilon, weight_offset, scale_ub) are
// captured directly from the outer scope via the parallel's graph region.
//===----------------------------------------------------------------------===//

namespace {

/// Returns true if \p a and \p b have the same operation name, the same
/// number of operands and results, and identical attribute dictionaries
/// modulo the `device` attribute (which legitimately differs across
/// per-launch ops).
static bool opsMatchModuloDevice(Operation *a, Operation *b) {
  if (a->getName() != b->getName() ||
      a->getNumOperands() != b->getNumOperands() ||
      a->getNumResults() != b->getNumResults())
    return false;

  auto stripDevice = [](DictionaryAttr attrs) {
    SmallVector<NamedAttribute> filtered;
    filtered.reserve(attrs.size());
    for (NamedAttribute attr : attrs)
      if (attr.getName() != MO::kOperationDeviceRefAttrName)
        filtered.push_back(attr);
    return DictionaryAttr::get(attrs.getContext(), filtered);
  };
  return stripDevice(a->getAttrDictionary()) ==
         stripDevice(b->getAttrDictionary());
}

} // namespace

struct FuseParallelBundledAllReduceAddRMSNormQuantFP8PatternMatcher {
public:
  explicit FuseParallelBundledAllReduceAddRMSNormQuantFP8PatternMatcher(
      MO::BundledAllReduceSumOp allReduceOp)
      : allReduceOp(allReduceOp), builder(allReduceOp->getContext()) {}

  void apply() {
    if (!matchAll())
      return;
    rewrite();
  }

private:
  // Inputs.
  MO::BundledAllReduceSumOp allReduceOp;
  OpBuilder builder;

  // Captured during matching.
  MO::ParallelOp parallelOp;
  MO::YieldOp yieldOp;
  MO::TensorUnbundleOp unbundleOp;
  // Position of the seed's tensor result in the body's mo.yield, which
  // also indexes the parallel's corresponding bundle result.
  size_t yieldOperandIdx = 0;
  // Per-launch operand chains.
  SmallVector<MO::AddOp> addOps;
  SmallVector<MO::ReduceRMSNormOp> rmsNormOps;
  SmallVector<MO::CustomOp> quantizeOps;
  SmallVector<Value> residualValues;
  SmallVector<Value> gammaValues;
  // Captured (same SSA value across launches).
  Value epsilonValue;
  Value weightOffsetValue;
  Value scaleUbValue;

  bool matchAll() {
    // 1. The seed must sit inside an mo.parallel body.
    parallelOp = allReduceOp->getParentOfType<MO::ParallelOp>();
    if (!parallelOp)
      return false;
    Block &body = parallelOp.getBodyRegion().front();
    yieldOp = cast<MO::YieldOp>(body.getTerminator());

    // 2. The seed's tensor result has exactly one use, and that use is
    //    the body's mo.yield.  Record which yield operand it occupies.
    Value seedOut = allReduceOp.getOutput();
    if (!seedOut.hasOneUse())
      return false;
    OpOperand &use = *seedOut.getUses().begin();
    if (use.getOwner() != yieldOp.getOperation())
      return false;
    yieldOperandIdx = use.getOperandNumber();

    // 3. The corresponding parallel result (yield operand index ==
    //    parallel bundle result index) is consumed by exactly one
    //    mo.tensor.unbundle.
    size_t numBundles =
        parallelOp.getNumResults() - (parallelOp.hasChain() ? 1 : 0);
    if (yieldOperandIdx >= numBundles)
      return false;
    Value parallelResult = parallelOp.getResult(yieldOperandIdx);
    if (!parallelResult.hasOneUse())
      return false;
    unbundleOp = dyn_cast<MO::TensorUnbundleOp>(
        parallelResult.getUses().begin()->getOwner());
    if (!unbundleOp)
      return false;
    auto bundleTy = dyn_cast<MO::BundleType>(parallelResult.getType());
    if (!bundleTy)
      return false;
    size_t numLaunches = unbundleOp.getNumResults();
    if (numLaunches == 0 || bundleTy.getNumElements() != numLaunches)
      return false;

    // 4. Walk the per-launch add -> rms_norm -> quantize chain on each
    //    unbundled result using the existing helpers.  Track claimed ops
    //    so a single op cannot be nominated as the peer for more than
    //    one launch.
    addOps.assign(numLaunches, nullptr);
    rmsNormOps.assign(numLaunches, nullptr);
    quantizeOps.assign(numLaunches, nullptr);
    residualValues.assign(numLaunches, Value{});
    gammaValues.assign(numLaunches, Value{});
    llvm::SmallDenseSet<Operation *> claimedAdds, claimedNorms, claimedQuants;
    for (size_t i = 0; i < numLaunches; ++i) {
      Value launchAllredOut = unbundleOp.getResult(i);
      MO::AddOp addOp;
      Value residualValue;
      MO::ReduceRMSNormOp rmsNormOp;
      MO::CustomOp quantizeOp;
      if (!matchResidualOp(launchAllredOut, addOp, residualValue))
        return false;
      if (!matchRMSNormOp(addOp.getResult(), rmsNormOp))
        return false;
      if (!matchCustomOp(rmsNormOp.getResult(), quantizeOp,
                         "mo.quantize_dynamic_scaled_float8"))
        return false;
      auto launchTy = dyn_cast<TensorType>(launchAllredOut.getType());
      if (!launchTy)
        return false;
      if (!verifyFusedAllReducePath(launchTy, addOp, rmsNormOp, quantizeOp))
        return false;
      if (!claimedAdds.insert(addOp.getOperation()).second ||
          !claimedNorms.insert(rmsNormOp.getOperation()).second ||
          !claimedQuants.insert(quantizeOp.getOperation()).second)
        return false;
      addOps[i] = addOp;
      rmsNormOps[i] = rmsNormOp;
      quantizeOps[i] = quantizeOp;
      residualValues[i] = residualValue;
      gammaValues[i] = rmsNormOp.getWeight();
    }

    // 5. Cross-launch structural match (modulo per-op `device` attr).
    for (size_t i = 1; i < numLaunches; ++i) {
      if (!opsMatchModuloDevice(addOps[0].getOperation(),
                                addOps[i].getOperation()))
        return false;
      if (!opsMatchModuloDevice(rmsNormOps[0].getOperation(),
                                rmsNormOps[i].getOperation()))
        return false;
      if (!opsMatchModuloDevice(quantizeOps[0].getOperation(),
                                quantizeOps[i].getOperation()))
        return false;
    }

    // 6. Captured operands: epsilon, weight_offset, scale_ub must be the
    //    same SSA value on every launch.
    epsilonValue = rmsNormOps[0].getEpsilon();
    weightOffsetValue = rmsNormOps[0].getWeightOffset();
    scaleUbValue = quantizeOps[0].getOperand(1);
    for (size_t i = 1; i < numLaunches; ++i) {
      if (rmsNormOps[i].getEpsilon() != epsilonValue ||
          rmsNormOps[i].getWeightOffset() != weightOffsetValue ||
          quantizeOps[i].getOperand(1) != scaleUbValue)
        return false;
    }

    // 7. Per-launch bundle operands must be device-coherent with the
    //    corresponding launch device (taken from the bundle result's
    //    element type).
    for (size_t i = 0; i < numLaunches; ++i) {
      auto launchTy = cast<MO::TensorType>(bundleTy.getElementTypes()[i]);
      auto resTy = dyn_cast<MO::TensorType>(residualValues[i].getType());
      auto gammaTy = dyn_cast<MO::TensorType>(gammaValues[i].getType());
      if (!resTy || resTy.getDeviceRef() != launchTy.getDeviceRef())
        return false;
      if (!gammaTy || gammaTy.getDeviceRef() != launchTy.getDeviceRef())
        return false;
    }
    return true;
  }

  void rewrite() {
    auto *ctx = builder.getContext();
    Location loc = parallelOp.getLoc();
    size_t numLaunches = unbundleOp.getNumResults();

    // 1. Build residual and gamma bundles immediately before the parallel.
    builder.setInsertionPoint(parallelOp);
    SmallVector<Type> residualElemTypes, gammaElemTypes;
    residualElemTypes.reserve(numLaunches);
    gammaElemTypes.reserve(numLaunches);
    for (size_t i = 0; i < numLaunches; ++i) {
      residualElemTypes.push_back(residualValues[i].getType());
      gammaElemTypes.push_back(gammaValues[i].getType());
    }
    auto residualBundleTy = MO::BundleType::get(ctx, residualElemTypes);
    auto gammaBundleTy = MO::BundleType::get(ctx, gammaElemTypes);
    auto residualBundle = MO::TensorBundleOp::create(
        builder, loc, residualBundleTy, residualValues);
    auto gammaBundle =
        MO::TensorBundleOp::create(builder, loc, gammaBundleTy, gammaValues);

    // 2. Plan the new parallel's inputs and result types.
    SmallVector<Value> newInputs(parallelOp.getInputs().begin(),
                                 parallelOp.getInputs().end());
    newInputs.push_back(residualBundle.getResult());
    newInputs.push_back(gammaBundle.getResult());

    size_t numOldBundles =
        parallelOp.getNumResults() - (parallelOp.hasChain() ? 1 : 0);
    SmallVector<Type> newResultTypes;
    SmallVector<std::optional<size_t>> oldResultToNewIdx(
        parallelOp.getNumResults(), std::nullopt);
    for (size_t i = 0; i < numOldBundles; ++i) {
      if (i == yieldOperandIdx) {
        // Replace the seed's bundle with three new bundles
        // (output, scale, residual).
        oldResultToNewIdx[i] = newResultTypes.size();
        SmallVector<Type> outElemTypes, scaleElemTypes, residualOutElemTypes;
        for (size_t j = 0; j < numLaunches; ++j) {
          outElemTypes.push_back(quantizeOps[j].getResult(0).getType());
          scaleElemTypes.push_back(quantizeOps[j].getResult(1).getType());
          residualOutElemTypes.push_back(addOps[j].getResult().getType());
        }
        newResultTypes.push_back(MO::BundleType::get(ctx, outElemTypes));
        newResultTypes.push_back(MO::BundleType::get(ctx, scaleElemTypes));
        newResultTypes.push_back(
            MO::BundleType::get(ctx, residualOutElemTypes));
      } else {
        oldResultToNewIdx[i] = newResultTypes.size();
        newResultTypes.push_back(parallelOp.getResult(i).getType());
      }
    }
    if (parallelOp.hasChain()) {
      oldResultToNewIdx[parallelOp.getNumResults() - 1] = newResultTypes.size();
      newResultTypes.push_back(MO::ChainType::get(ctx));
    }

    auto newParallel =
        MO::ParallelOp::create(builder, loc, newInputs, parallelOp.getBuffers(),
                               parallelOp.getInChain(), newResultTypes);

    // 3. Splice the body via IRMapping.  Clone every op in the old body
    //    except the seed (which we replace with the fused op).
    Block &oldBody = parallelOp.getBodyRegion().front();
    Block &newBody = newParallel.getBodyRegion().front();
    IRMapping mapping;
    // Map the per-bundle block args 1:1 (same leading positions in both
    // bodies). The new body appends residual+gamma args before the trailing
    // !mo.chain block arg, so map the chain arg explicitly (old last -> new
    // last) rather than via a contiguous take_front, which would misalign the
    // chain arg with residual/gamma.
    size_t numOrigInputs = parallelOp.getInputs().size();
    for (size_t i = 0; i < numOrigInputs; ++i)
      mapping.map(oldBody.getArgument(i), newBody.getArgument(i));
    Value residualArg = newBody.getArgument(numOrigInputs);
    Value gammaArg = newBody.getArgument(numOrigInputs + 1);
    if (parallelOp.hasChain())
      mapping.map(oldBody.getArguments().back(), newBody.getArguments().back());

    OpBuilder bodyBuilder(ctx);
    bodyBuilder.setInsertionPointToEnd(&newBody);
    for (Operation &op : oldBody.without_terminator()) {
      if (&op == allReduceOp.getOperation())
        continue;
      bodyBuilder.clone(op, mapping);
    }

    // Build the fused op in place of the seed.  Peer inputs and the
    // chain operand come through the IRMapping; signal buffers, eps, wo
    // and scale_ub are captured from outer scope (the graph region
    // permits direct outer-SSA references).
    SmallVector<Value> fusedInputs;
    fusedInputs.reserve(allReduceOp.getInputs().size());
    for (Value peer : allReduceOp.getInputs())
      fusedInputs.push_back(mapping.lookupOrDefault(peer));
    SmallVector<Value> fusedSignals;
    fusedSignals.reserve(allReduceOp.getSignalBuffers().size());
    for (Value sig : allReduceOp.getSignalBuffers())
      fusedSignals.push_back(mapping.lookupOrDefault(sig));
    Value fusedChain = mapping.lookupOrDefault(allReduceOp.getInChain());

    auto fusedOp = MO::Composite::BundledAllReduceAddRMSNormQuantFP8Op::create(
        bodyBuilder, allReduceOp.getLoc(),
        /*output=*/quantizeOps[0].getResult(0).getType(),
        /*outScale=*/quantizeOps[0].getResult(1).getType(),
        /*outResidual=*/addOps[0].getResult().getType(),
        /*outChain=*/MO::ChainType::get(ctx),
        /*inputs=*/fusedInputs,
        /*signalBuffers=*/fusedSignals,
        /*residual=*/residualArg,
        /*gamma=*/gammaArg,
        /*epsilon=*/epsilonValue,
        /*weightOffset=*/weightOffsetValue,
        /*scaleUb=*/scaleUbValue,
        /*inChain=*/fusedChain);

    // The body yields the seed collective's out-chain (chained parallels).
    // The fused op subsumes the seed, so route the yielded chain to it; the
    // rebuilt yield below then threads the correct completion chain.
    if (parallelOp.hasChain())
      mapping.map(allReduceOp.getOutChain(), fusedOp.getOutChain());

    // Build the new yield: keep all old yield operands, but in the slot
    // for the seed substitute the fused op's three tensor outputs.
    SmallVector<Value> newYieldOperands;
    for (auto [i, v] : llvm::enumerate(yieldOp.getOperands())) {
      if (i == yieldOperandIdx) {
        newYieldOperands.push_back(fusedOp.getOutput());
        newYieldOperands.push_back(fusedOp.getOutScale());
        newYieldOperands.push_back(fusedOp.getOutResidual());
      } else {
        newYieldOperands.push_back(mapping.lookupOrDefault(v));
      }
    }
    MO::YieldOp::create(bodyBuilder, yieldOp.getLoc(), newYieldOperands);

    // 4. External rewire: unbundle each new bundle result and rewrite the
    //    per-device users.  The post-add residual may have non-rms_norm
    //    consumers (e.g. an external `mo.add(%other, %add)` reading the
    //    residual); replaceAllUsesExcept skips the rms_norm we're about
    //    to erase.
    builder.setInsertionPointAfter(newParallel);
    size_t newOutIdx = *oldResultToNewIdx[yieldOperandIdx];
    Value outBundle = newParallel.getResult(newOutIdx);
    Value scaleBundle = newParallel.getResult(newOutIdx + 1);
    Value residualBundleOut = newParallel.getResult(newOutIdx + 2);
    auto outUnbundle = MO::TensorUnbundleOp::create(
        builder, loc,
        cast<MO::BundleType>(outBundle.getType()).getElementTypes(), outBundle);
    auto scaleUnbundle = MO::TensorUnbundleOp::create(
        builder, loc,
        cast<MO::BundleType>(scaleBundle.getType()).getElementTypes(),
        scaleBundle);
    auto residualUnbundle = MO::TensorUnbundleOp::create(
        builder, loc,
        cast<MO::BundleType>(residualBundleOut.getType()).getElementTypes(),
        residualBundleOut);

    for (size_t i = 0; i < numLaunches; ++i) {
      quantizeOps[i].getResult(0).replaceAllUsesWith(outUnbundle.getResult(i));
      quantizeOps[i].getResult(1).replaceAllUsesWith(
          scaleUnbundle.getResult(i));
      addOps[i].getResult().replaceAllUsesExcept(residualUnbundle.getResult(i),
                                                 rmsNormOps[i]);
    }

    // Route any non-seed bundle results and the chain to the new parallel.
    for (size_t i = 0; i < parallelOp.getNumResults(); ++i) {
      if (i == yieldOperandIdx)
        continue; // seed bundle is already routed via the new unbundles
      parallelOp.getResult(i).replaceAllUsesWith(
          newParallel.getResult(*oldResultToNewIdx[i]));
    }

    // 5. Erase the now-dead per-device chain ops, the old unbundle, and
    //    the old parallel.
    for (size_t i = 0; i < numLaunches; ++i) {
      quantizeOps[i].erase();
      rmsNormOps[i].erase();
      addOps[i].erase();
    }
    unbundleOp.erase();
    parallelOp.erase();
  }
};

/// Follow `v` through broadcast_to and rebind ops (each must have exactly one
/// use) until a MulOp is reached. Intermediate ops are appended to `chain`.
/// Returns nullptr if the chain is broken or ambiguous.
static MO::MulOp followToMul(Value v, SmallVectorImpl<Operation *> &chain) {
  while (true) {
    if (!v.hasOneUse())
      return nullptr;
    Operation *user = *v.getUsers().begin();
    if (auto mulOp = dyn_cast<MO::MulOp>(user))
      return mulOp;
    if (isa<MO::RebindOp, MO::BroadcastToOp>(user)) {
      chain.push_back(user);
      v = user->getResult(0);
      continue;
    }
    return nullptr;
  }
}

/// Follow `v` through rebind ops (each must have exactly one use) until an
/// AddOp is reached. Intermediate rebind ops are appended to `chain`.
/// Returns nullptr if the chain is broken.
static MO::AddOp followToAdd(Value v, SmallVectorImpl<Operation *> &chain) {
  while (true) {
    if (!v.hasOneUse())
      return nullptr;
    Operation *user = *v.getUsers().begin();
    if (auto addOp = dyn_cast<MO::AddOp>(user))
      return addOp;
    if (auto rebindOp = dyn_cast<MO::RebindOp>(user)) {
      chain.push_back(rebindOp);
      v = rebindOp.getResult();
      continue;
    }
    return nullptr;
  }
}

/// Fuses RMS normalization with a subsequent Rotary Position Embedding (RoPE):
///
///   normed = rms_norm(input, weight, eps, offset)
///   x1, x2 = split/slice(normed, axis=-1)   // two equal halves
///   rotated = concat(-x2, x1, axis=-1)
///   result  = normed * cos + rotated * sin   (with optional up/downcasts)
///
/// into a single mo.composite.rms_norm_rope op.
///
/// The split must be a SplitOp (created by SliceToSplitPattern).
/// Between casts and muls, and between muls and the add, there may be chains
/// of broadcast_to/rebind ops (introduced by RMO legalization) that are
/// traversed transparently.
struct ReduceRMSNormRoPEPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern,
                                           MO::ReduceRMSNormOp> {
  ReduceRMSNormRoPEPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::ReduceRMSNormOp rmsNormOp,
                             MO::DeviceAwareRewriter &rewriter) const override {
    Value normed = rmsNormOp.getResult();
    auto normedTy = cast<MO::TensorType>(normed.getType());

    // Only fuse on GPU — no GPU kernel exists for other devices.
    DeviceRefAttr device = normedTy.getDeviceRef();
    if (!device || device.getLabel() != M::kGPULabel) {
      return rewriter.notifyMatchFailure(
          rmsNormOp, "rms_norm.RoPE fusion only supported on GPU");
    }

    int64_t rank = static_cast<int64_t>(normedTy.getRank());
    int64_t rmsAxis = rank - 1; // RoPE splits along the last axis

    int64_t normedLastDim = normedTy.getIntDim(rmsAxis);
    if (normedLastDim == ShapedType::kDynamic || normedLastDim % 2 != 0) {
      return rewriter.notifyMatchFailure(
          rmsNormOp, "rms_norm last dimension must be a known even number");
    }

    // ---- 0. Step over a narrowing downcast between rms_norm and RoPE -------
    // Model code commonly computes RMSNorm "in float32" and casts back to the
    // activation dtype *before* RoPE:
    //
    //   normed_hi = reduce.rms_norm(cast(x_lo, f32), w, ...)   // f32
    //   normed_lo = cast(normed_hi, lo)                         // bf16/fp16
    //   x1, x2 = split(normed_lo, -1); ...
    //
    // The split/mul then consume `normed_lo`, not the rms_norm result, so the
    // RoPE ops below must be matched against the cast result. The resulting
    // composite is f32-in / lo-out (the kernel decouples output dtype), keeping
    // weight/epsilon in f32. The input upcast is left in place for downstream
    // prologue (FusedInput) fusion to absorb. RMSNorm accumulates in f32
    // regardless of I/O dtype, so for an f32 norm the internal pre-cast is a
    // no-op and the up/down-cast idiom is exactly `multiply_before_cast =
    // true`.
    Value ropeRoot = normed;
    MO::CastOp rmsDowncast = nullptr;
    bool forceMultiplyBeforeCast = false;
    if (normed.hasOneUse()) {
      if (auto castOp = dyn_cast<MO::CastOp>(*normed.getUsers().begin())) {
        auto castTy = cast<MO::TensorType>(castOp.getResult().getType());
        auto from = normedTy.getDtype();
        auto to = castTy.getDtype();
        if (from.isFloat() && to.isFloat() &&
            to.getSizeInBytes() < from.getSizeInBytes()) {
          ropeRoot = castOp.getResult();
          rmsDowncast = castOp;
          forceMultiplyBeforeCast = true;
        }
      }
    }

    // ---- 1. Find x1/x2 as a SplitOp on the last axis ----------------------
    MO::SplitOp normSplit = nullptr;
    Value x1, x2;

    for (Operation *user : ropeRoot.getUsers()) {
      auto splitOp = dyn_cast<MO::SplitOp>(user);
      if (!splitOp || splitOp.getNumResults() != 2)
        continue;
      // `axis` is a compile-time `index` attribute.
      if (normalizeAxis(splitOp.getAxisAttr().getInt(), rank) != rmsAxis)
        continue;
      normSplit = splitOp;
      break;
    }

    if (!normSplit) {
      return rewriter.notifyMatchFailure(
          rmsNormOp, "no 2-way split on last axis found after rms_norm");
    }

    // split result(0) has smaller start → x1 (first half)
    x1 = normSplit.getResult(0);
    x2 = normSplit.getResult(1);

    // ---- 2. x2 must feed a NegativeOp ------------------------------------
    if (!x2.hasOneUse())
      return rewriter.notifyMatchFailure(rmsNormOp, "x2 has multiple uses");
    // x2 having a single use does not guarantee that use is a negate (e.g. the
    // Gemma4 vision RoPE feeds the split halves elsewhere). dyn_cast + bail
    // rather than an unchecked cast, which would abort on a non-negate user.
    auto negOp = dyn_cast<MO::NegativeOp>(*x2.getUsers().begin());
    if (!negOp)
      return rewriter.notifyMatchFailure(rmsNormOp,
                                         "x2's single use is not a negate");

    // ---- 3. neg(x2) and x1 feed a ConcatOp on the last axis -------------
    if (!negOp.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(rmsNormOp,
                                         "neg(x2) has multiple uses");
    }
    auto concatOp =
        dyn_cast<MO::ConcatOp>(*negOp.getResult().getUsers().begin());
    if (!concatOp) {
      return rewriter.notifyMatchFailure(rmsNormOp,
                                         "neg(x2) not consumed by concat");
    }

    int64_t concatAxis = concatOp.getAxisAttr().getInt();
    if (normalizeAxis(concatAxis, rank) != rmsAxis) {
      return rewriter.notifyMatchFailure(rmsNormOp,
                                         "concat axis differs from rms axis");
    }

    // Verify operand order: concat(-x2, x1) — NOT concat(x1, -x2)
    if (concatOp.getInputs().size() != 2) {
      return rewriter.notifyMatchFailure(
          rmsNormOp, "concat does not have exactly 2 inputs");
    }
    if (concatOp.getInputs()[0] != negOp.getResult() ||
        concatOp.getInputs()[1] != x1) {
      return rewriter.notifyMatchFailure(
          rmsNormOp, "concat inputs are not in expected order (-x2, x1)");
    }

    // ---- 4. Optional upcast of the rotated tensor ------------------------
    Value rotated = concatOp.getResult();
    MO::CastOp rotatedCastOp = nullptr;
    Value rotatedMaybeUpcast = rotated;
    if (rotated.hasOneUse()) {
      if (auto castOp = dyn_cast<MO::CastOp>(*rotated.getUsers().begin())) {
        rotatedCastOp = castOp;
        rotatedMaybeUpcast = castOp.getResult();
      }
    }

    // ---- 5. rotated (or its cast) feeds mul(rotated, sin_vals) ----------
    // Allow broadcast_to/rebind chains between the cast and the mul.
    SmallVector<Operation *> rotatedToMulChain;
    auto mulRotatedOp = followToMul(rotatedMaybeUpcast, rotatedToMulChain);
    if (!mulRotatedOp)
      return rewriter.notifyMatchFailure(rmsNormOp, "rotated not fed into mul");
    Value rotatedMulInput = rotatedToMulChain.empty()
                                ? rotatedMaybeUpcast
                                : rotatedToMulChain.back()->getResult(0);
    Value sinVals = (mulRotatedOp.getOperand(0) == rotatedMulInput)
                        ? mulRotatedOp.getOperand(1)
                        : mulRotatedOp.getOperand(0);

    // ---- 6. normed (or its cast) feeds mul(normed, cos_vals) -------------
    // Allow broadcast_to/rebind chains between the cast and the mul.
    MO::MulOp mulNormedOp = nullptr;
    MO::CastOp normedCastOp = nullptr;
    Value cosVals;
    SmallVector<Operation *> normedToMulChain;
    for (Operation *user : ropeRoot.getUsers()) {
      if (user == normSplit)
        continue;
      if (auto mulOp = dyn_cast<MO::MulOp>(user)) {
        cosVals = (mulOp.getOperand(0) == ropeRoot) ? mulOp.getOperand(1)
                                                    : mulOp.getOperand(0);
        mulNormedOp = mulOp;
        break;
      }
      if (auto castOp = dyn_cast<MO::CastOp>(user)) {
        SmallVector<Operation *> chain;
        auto mulOp = followToMul(castOp.getResult(), chain);
        if (!mulOp)
          continue;
        Value chainEnd = chain.empty() ? Value(castOp.getResult())
                                       : chain.back()->getResult(0);
        cosVals = (mulOp.getOperand(0) == chainEnd) ? mulOp.getOperand(1)
                                                    : mulOp.getOperand(0);
        normedCastOp = castOp;
        normedToMulChain = std::move(chain);
        mulNormedOp = mulOp;
        break;
      }
    }
    if (!mulNormedOp) {
      return rewriter.notifyMatchFailure(rmsNormOp,
                                         "no mul(normed, cos_vals) found");
    }

    // ---- 7. add(normed*cos, rotated*sin) — commutative ------------------
    // There may be rebind ops between each mul and the add.
    SmallVector<Operation *> normedCosChain, rotatedSinChain;
    auto addOp = followToAdd(mulNormedOp.getResult(), normedCosChain);
    if (!addOp) {
      return rewriter.notifyMatchFailure(rmsNormOp,
                                         "normedCos not fed into add");
    }
    auto addOp2 = followToAdd(mulRotatedOp.getResult(), rotatedSinChain);
    if (addOp2 != addOp) {
      return rewriter.notifyMatchFailure(
          rmsNormOp, "normedCos and rotatedSin do not feed the same add");
    }

    Value normedCosInput = normedCosChain.empty()
                               ? Value(mulNormedOp.getResult())
                               : normedCosChain.back()->getResult(0);
    Value rotatedSinInput = rotatedSinChain.empty()
                                ? Value(mulRotatedOp.getResult())
                                : rotatedSinChain.back()->getResult(0);
    bool addValid = (addOp.getOperand(0) == normedCosInput &&
                     addOp.getOperand(1) == rotatedSinInput) ||
                    (addOp.getOperand(0) == rotatedSinInput &&
                     addOp.getOperand(1) == normedCosInput);
    if (!addValid) {
      return rewriter.notifyMatchFailure(
          rmsNormOp, "add does not consume both normed*cos and rotated*sin");
    }

    // ---- 8. Optional downcast back to input dtype -------------------------
    Value addResult = addOp.getResult();
    Value finalResult = addResult;
    MO::CastOp downcastOp = nullptr;
    if (addResult.hasOneUse()) {
      if (auto castOp = dyn_cast<MO::CastOp>(*addResult.getUsers().begin())) {
        auto castResultTy = cast<MO::TensorType>(castOp.getResult().getType());
        if (castResultTy.getElementType() == normedTy.getElementType()) {
          downcastOp = castOp;
          finalResult = castOp.getResult();
        }
      }
    }

    // ---- 9. Device consistency check ------------------------------------
    // RebindOps are MO_ControlOps (MOExecutesOnHost) so they always have CPU
    // device. Only push GPU computation ops into the device consistency check.
    *rewriter.pushOp() = normSplit;
    *rewriter.pushOp() = negOp;
    *rewriter.pushOp() = concatOp;
    if (rotatedCastOp)
      *rewriter.pushOp() = rotatedCastOp;
    for (Operation *op : rotatedToMulChain) {
      if (!isa<MO::RebindOp>(op))
        *rewriter.pushOp() = op;
    }
    *rewriter.pushOp() = mulRotatedOp;
    // rotatedSinChain: only rebind ops, skip all.
    if (normedCastOp)
      *rewriter.pushOp() = normedCastOp;
    for (Operation *op : normedToMulChain) {
      if (!isa<MO::RebindOp>(op))
        *rewriter.pushOp() = op;
    }
    *rewriter.pushOp() = mulNormedOp;
    // normedCosChain: only rebind ops, skip all.
    *rewriter.pushOp() = addOp;
    if (downcastOp)
      *rewriter.pushOp() = downcastOp;
    // The downcast we fused through (rms_norm result -> RoPE input) is a GPU
    // op.
    if (rmsDowncast)
      *rewriter.pushOp() = rmsDowncast;
    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          rmsNormOp, "matched ops have inconsistent devices");
    }

    // ---- 10. Create fused op and replace uses ----------------------------
    auto outputTy = downcastOp ? downcastOp.getResult().getType()
                               : cast<MO::TensorType>(addResult.getType());

    // When we fused through an f32->lo downcast, the rms_norm's internal cast
    // to its (f32) result dtype was a no-op, so the up/down-cast idiom is
    // exactly `multiply_before_cast = true` regardless of the original flag.
    bool multiplyBeforeCast =
        forceMultiplyBeforeCast
            ? true
            : rmsNormOp.getMultiplyBeforeCastAttr().getValue();

    auto fusedOp = rewriter.create<MO::Composite::ReduceRMSNormRoPEOp>(
        rmsNormOp.getLoc(), outputTy, rmsNormOp.getInput(),
        rmsNormOp.getWeight(), rmsNormOp.getEpsilon(),
        rmsNormOp.getWeightOffset(), cosVals, sinVals, multiplyBeforeCast,
        rmsNormOp.getOutputParamDecls());

    rewriter.replaceAllUsesWith(finalResult, fusedOp.getResult());

    // Erase ops from outermost (consumers) to innermost (producers).
    if (downcastOp)
      rewriter.eraseOp(downcastOp);
    rewriter.eraseOp(addOp);
    // Erase chains between muls and add (reverse = consumer first).
    for (auto *op : llvm::reverse(normedCosChain))
      rewriter.eraseOp(op);
    for (auto *op : llvm::reverse(rotatedSinChain))
      rewriter.eraseOp(op);
    rewriter.eraseOp(mulNormedOp);
    // Erase chains between normedCast and normedMul (reverse).
    for (auto *op : llvm::reverse(normedToMulChain))
      rewriter.eraseOp(op);
    if (normedCastOp)
      rewriter.eraseOp(normedCastOp);
    rewriter.eraseOp(mulRotatedOp);
    // Erase chains between rotatedCast/concat and rotatedMul (reverse).
    for (auto *op : llvm::reverse(rotatedToMulChain))
      rewriter.eraseOp(op);
    if (rotatedCastOp)
      rewriter.eraseOp(rotatedCastOp);
    rewriter.eraseOp(concatOp);
    rewriter.eraseOp(negOp);
    rewriter.eraseOp(normSplit);
    // The downcast we fused through is now dead (its only uses were the RoPE
    // ops just erased); remove it before the rms_norm that feeds it. The input
    // upcast is intentionally left for downstream prologue fusion.
    if (rmsDowncast)
      rewriter.eraseOp(rmsDowncast);
    rewriter.eraseOp(rmsNormOp);

    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Matmul + Add Fusion Pattern (mo.matmul_add)
//===----------------------------------------------------------------------===//
//
// Fuses:
//   %mm  = mo.matmul(%a, %b) {transpose_b = ...}
//   %out = mo.add(%mm, %residual)   (or mo.add(%residual, %mm))
//
// Into:
//   %out = mo.matmul_add(%a, %b, %residual) {transpose_b = ...}
//
// Only a 1D bias is fused: the addend must be a rank-1 tensor broadcast to the
// matmul's [M, N] output (`add(matmul, broadcast(bias_1d))`). A full rank-2
// residual is left as a separate `mo.add`. The fused op is GPU-only and
// requires a single consumer for the matmul result.
//
//===----------------------------------------------------------------------===//

struct MatmulAddPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern, MO::AddOp> {
  MatmulAddPattern(MLIRContext *ctx, MO::DeviceInfoMapping deviceInfoMapping)
      : DeviceAwareRewritePattern(ctx, 2),
        _deviceInfoMapping(std::move(deviceInfoMapping)) {}

private:
  MO::DeviceInfoMapping _deviceInfoMapping;

  static MO::MatmulOp matchMatmulOp(Value val,
                                    MO::Static::ReshapeOp &reshapeOp) {
    auto reshape = val.getDefiningOp<MO::Static::ReshapeOp>();
    if (!reshape)
      return val.getDefiningOp<MO::MatmulOp>();

    if (!reshape.getResult().hasOneUse())
      return nullptr;
    auto matmulOp = reshape.getInput().getDefiningOp<MO::MatmulOp>();
    if (!matmulOp)
      return nullptr;

    reshapeOp = reshape;
    return matmulOp;
  }

  static bool wouldBlockPrologueFusion(Value value) {
    Operation *op = value.getDefiningOp();
    // Only block if the op has a single use: multi-use values cannot be
    // prologue-fused without recomputation, so there is no opportunity to
    // preserve by deferring matmul+add fusion.
    return op && value.hasOneUse() &&
           isa<MO::ElementWiseLike, MO::ViewLike>(op);
  }

  // Returns true if there are downstream elementwise ops that won't get
  // epilogue fused because fused_matmul_add does not support epilogue fusion.
  static bool wouldBlockEpilogueFusion(Value val) {
    while (true) {
      // We can't fuse if the value has multiple uses.
      if (!val.hasOneUse())
        return false;
      Operation *user = *val.getUsers().begin();
      // Swim through reshapes.
      if (isa<MO::Static::ReshapeOp>(user)) {
        val = user->getResult(0);
        continue;
      }

      return isa<MO::ElementWiseLike>(user);
    }
  }

  static bool is1DTo2DBroadcast(MO::Static::BroadcastToOp broadcastOp) {
    TensorType inputTy = broadcastOp.getInput().getType();
    TensorType resultTy = broadcastOp.getResult().getType();
    if (inputTy.getRank() != 1 || resultTy.getRank() != 2)
      return false;

    auto inputInnerDim = inputTy.getDim(0);
    auto outputInnerDim = resultTy.getDim(1);
    return inputInnerDim == outputInnerDim;
  }

  static LogicalResult
  checkKernelConstraints(MO::MatmulOp matmulOp, MO::AddOp addOp,
                         MO::DeviceAwareRewriter &rewriter) {
    Type cType = matmulOp.getResult().getType().getElementType();
    Type aType = matmulOp.getInputA().getType().getElementType();
    Type bType = matmulOp.getInputB().getType().getElementType();
    if (!cType.isBF16() || !aType.isBF16() || !bType.isBF16())
      return rewriter.notifyMatchFailure(
          addOp, "fused_matmul_add requires bfloat16 input and output types");

    int64_t dimN = matmulOp.getResult().getType().getIntDim(1);
    int64_t dimK = matmulOp.getInputA().getType().getIntDim(1);
    if (dimN == mlir::ShapedType::kDynamic ||
        dimK == mlir::ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(
          addOp, "fused_matmul_add does not support dynamic N or K dimensions");

    return success();
  }

  LogicalResult
  deviceAwareMatchAndRewrite(MO::AddOp addOp,
                             MO::DeviceAwareRewriter &rewriter) const override {
    // Only fuse on GPU.
    auto resultTy = cast<MO::TensorType>(addOp.getResult().getType());
    DeviceRefAttr device = resultTy.getDeviceRef();
    if (!device || device.getLabel() != M::kGPULabel) {
      return rewriter.notifyMatchFailure(addOp,
                                         "matmul+add fusion only on GPU");
    }

    // Only apply for SM100 devices.
    DeviceInfoAttr deviceInfo = _deviceInfoMapping.get(device);
    if (deviceInfo.getArch() != "sm_100a") {
      return rewriter.notifyMatchFailure(
          addOp, "matmul+add fusion only applies to NVIDIA sm_100a");
    }

    if (wouldBlockEpilogueFusion(addOp.getResult())) {
      return rewriter.notifyMatchFailure(
          addOp, "fused_matmul_add does not support epilogue fusion of "
                 "downstream elementwise ops");
    }

    // One input must be a matmul result; the other is the residual.
    Value lhs = addOp.getInputX();
    Value rhs = addOp.getInputY();
    MO::Static::ReshapeOp reshapeOp;
    MO::MatmulOp matmulOp;
    Value residual;

    if ((matmulOp = matchMatmulOp(lhs, reshapeOp))) {
      residual = rhs;
    } else if ((matmulOp = matchMatmulOp(rhs, reshapeOp))) {
      residual = lhs;
    } else {
      return rewriter.notifyMatchFailure(addOp,
                                         "neither input is a matmul result");
    }

    // Only fuse if matmul has a single use (add).
    if (!matmulOp.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(addOp,
                                         "matmul result has multiple uses");
    }

    if (failed(checkKernelConstraints(matmulOp, addOp, rewriter)))
      return failure();

    // Only fuse a 1D bias (`matmul + broadcast(bias_1d)`).
    // TODO: Enable fusion of a full 2D residual
    bool is1DBias = false;
    if (auto broadcastOp = residual.getDefiningOp<MO::Static::BroadcastToOp>();
        broadcastOp && !reshapeOp && is1DTo2DBroadcast(broadcastOp)) {
      residual = broadcastOp.getInput();
      is1DBias = true;
    }
    if (!is1DBias) {
      return rewriter.notifyMatchFailure(
          addOp, "fused_matmul_add only fuses a 1D bias, not a 2D residual");
    }

    if (wouldBlockPrologueFusion(residual)) {
      return rewriter.notifyMatchFailure(
          addOp, "residual cannot be an elementwise op");
    }

    // Collect device annotations from both matched ops.
    if (!rewriter.addOps({matmulOp})) {
      return rewriter.notifyMatchFailure(
          addOp, "matched ops have inconsistent devices");
    }

    // Replace add(matmul(a, b), residual) -> matmul_add(a, b, residual).

    if (reshapeOp) {
      // If there was a reshape, the residual must be reshaped back to the
      // matmul output.
      residual = rewriter.create<MO::Static::ReshapeOp>(
          residual.getLoc(), matmulOp.getResult().getType(), residual);
    }

    Value newOutput = rewriter.create<MO::Composite::FusedMatmulAddOp>(
        addOp.getLoc(), matmulOp.getResult().getType(), matmulOp.getInputA(),
        matmulOp.getInputB(), residual,
        rewriter.getBoolAttr(matmulOp.getTransposeB()));

    // If there was a reshape, we need to insert a a new one for the fused
    // matmul output.
    if (reshapeOp) {
      newOutput = rewriter.create<MO::Static::ReshapeOp>(
          reshapeOp.getLoc(), reshapeOp.getResult().getType(), newOutput);
    }

    rewriter.replaceOp(addOp, newOutput);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// MegaFFN NVFP4 Fusion Pattern
//===----------------------------------------------------------------------===//
//
// Collapses the two-op NVFP4 MoE FFN chain
//
//   %down_in, %silu_scales =
//       mo.composite.grouped_matmul_swiglu_nvfp4(...)        // L1: gate-up
//   %out = mo.composite.grouped_matmul_block_scaled(
//       %down_in, %down_w, %silu_scales, ...)                // L2: down
//       (anchor)
//
// into a single `mo.composite.mega_ffn_nvfp4`. Anchored on the down op (L2);
// the gate-up op (L1) is reached as the producer of L2's activations. The
// intermediate (`down_in`, `silu_scales`) is consumed only by L2 and is
// dropped.
//
// SM100/B200 only — arch-gated internally via DeviceInfoMapping (mirroring
// MatmulAddPattern) and additionally behind the `mega-ffn-enable` driver pass
// option (see populateFusionMegaFFNPatterns and PatternFusion). The
// host-contract invariants
// the kernel relies on (front-packed active experts, ids[e] >= 0 => tokens > 0,
// device-only routing buffers) are upstream EP-gate guarantees: the pattern
// does not — and cannot statically — prove them. It only fires when the
// producer is the matching gate-up composite, which is the path that already
// satisfies them.
struct MegaFFNNvfp4Pattern
    : public MO::DeviceAwareRewritePattern<
          OpRewritePattern, MO::Composite::GroupedMatmulBlockScaledOp> {
  // Hard scheduler limit: the kernel reads ids[lane] and ids[lane + 32] across
  // the scheduler warp's 32 lanes, so the active-expert array is at most 64.
  static constexpr int64_t kMaxNumExperts = 64;

  MegaFFNNvfp4Pattern(MLIRContext *ctx, MO::DeviceInfoMapping deviceInfoMapping)
      : DeviceAwareRewritePattern(ctx, 2),
        _deviceInfoMapping(std::move(deviceInfoMapping)) {}

private:
  MO::DeviceInfoMapping _deviceInfoMapping;

public:
  LogicalResult
  deviceAwareMatchAndRewrite(MO::Composite::GroupedMatmulBlockScaledOp downOp,
                             MO::DeviceAwareRewriter &rewriter) const override {
    // Fuse only on GPU.
    auto resultTy = cast<MO::TensorType>(downOp.getOutput().getType());
    DeviceRefAttr device = resultTy.getDeviceRef();
    if (!device || device.getLabel() != M::kGPULabel)
      return rewriter.notifyMatchFailure(downOp, "mega_ffn fusion only on GPU");

    // Only apply for SM100 devices; the fused kernel is sm_100a-only.
    DeviceInfoAttr deviceInfo = _deviceInfoMapping.get(device);
    if (deviceInfo.getArch() != "sm_100a")
      return rewriter.notifyMatchFailure(
          downOp, "mega_ffn fusion only applies to NVIDIA sm_100a");

    // The down op's activations must come from a gate-up SwiGLU+quant op (L1).
    auto l1 = downOp.getHiddenStates()
                  .getDefiningOp<MO::Composite::GroupedMatmulSwigluNvfp4Op>();
    if (!l1)
      return rewriter.notifyMatchFailure(
          downOp, "down activations not from grouped_matmul_swiglu_nvfp4");

    // The two outputs of L1 must be paired into L2: down_in == L1#0 (packed
    // activations) and the down a_scales == L1#1 (SwiGLU scale tile). Guards
    // against an unrelated block-scaled GMM that reads only one L1 output.
    if (downOp.getHiddenStates() != l1.getCPacked())
      return rewriter.notifyMatchFailure(
          downOp, "down activations are not L1's packed output");
    if (downOp.getAScales() != l1.getCSwigluScales())
      return rewriter.notifyMatchFailure(
          downOp, "down a_scales are not L1's SwiGLU scale output");

    // Both L1 outputs must feed only this down op; otherwise fusing would have
    // to re-materialize the intermediate.
    if (!l1.getCPacked().hasOneUse() || !l1.getCSwigluScales().hasOneUse())
      return rewriter.notifyMatchFailure(
          downOp, "L1 intermediate has more than one use");

    // Both legs must be driven by the same EP routing tensors so the fused
    // kernel's single scheduler can enumerate both legs' tiles. (num_active_
    // experts and expert_scales are per-leg and legitimately differ.)
    if (downOp.getExpertStartIndices() != l1.getExpertStartIndices() ||
        downOp.getExpertIds() != l1.getExpertIds() ||
        downOp.getAScaleOffsets() != l1.getAScaleOffsets() ||
        downOp.getEstimatedTotalM() != l1.getEstimatedTotalM())
      return rewriter.notifyMatchFailure(
          downOp, "L1 and L2 are driven by different routing tensors");

    // num_experts is the leading dim of the rank-3 down weight; bail if it is
    // statically known to exceed the scheduler's hard limit.
    int64_t numExperts =
        cast<MO::TensorType>(downOp.getWeight().getType()).getIntDim(0);
    if (numExperts != mlir::ShapedType::kDynamic && numExperts > kMaxNumExperts)
      return rewriter.notifyMatchFailure(downOp, "num_experts exceeds 64");

    // Output must be bf16 (the fused kernel writes bf16 hidden states).
    if (!resultTy.getElementType().isBF16())
      return rewriter.notifyMatchFailure(downOp,
                                         "mega_ffn output must be bfloat16");

    // Device-consistency check across both fused ops.
    if (!rewriter.addOps({l1, downOp}))
      return rewriter.notifyMatchFailure(
          downOp, "matched ops have inconsistent devices");

    Location loc = downOp.getLoc();
    MLIRContext *ctx = downOp.getContext();

    // ---- Mint the persistent `arrival_count` pool-slot buffer. ----
    //
    // The kernel uses cross-CTA pool-slot arrival counters (strided by
    // ATOMIC_PAD=32) to hand off between the two legs. Under the registration's
    // `POST_SELF_CLEAN_UP` policy every touched slot is reset in band, so a
    // buffer that starts all-zero stays all-zero at every launch boundary. We
    // mint it here as a PERSISTENT buffer via `mo.buffer.create` with a zero
    // `initValue`: the runtime allocates and zeroes it ONCE at model load and
    // reuses it across executions -- no per-launch alloc/memset, and no
    // in-graph fill that would race under ExecInvariantCodeMotion or break
    // under stream capture.
    //
    // Sized to a static upper bound on `total_m_blocks` (sum over experts of
    // `ceildiv(tokens_e, MMA_N)`, which the kernel indexes in
    // `[0, total_m_blocks)`): with the smallest decode tile MMA_N=8 and at most
    // `numExperts` partial blocks, `ceildiv(M, 8) + numExperts + 1` bounds it.
    // Over-allocation is safe (untouched slots are never read). This needs a
    // static token bound, so bail if either M or numExperts is dynamic (the
    // >64 guard above only rejects a statically-known oversize). The EP path
    // that emits these composites uses statically-shaped activation buffers.
    int64_t m =
        cast<MO::TensorType>(l1.getHiddenStates().getType()).getIntDim(0);
    if (m == mlir::ShapedType::kDynamic ||
        numExperts == mlir::ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(
          downOp, "mega_ffn needs a static token count and num_experts to size "
                  "the arrival_count buffer");

    constexpr int64_t kAtomicPad = 32;
    constexpr int64_t kMinTokenBlock = 8;
    int64_t arrivalCountSize =
        ((m + kMinTokenBlock - 1) / kMinTokenBlock + numExperts + 1) *
        kAtomicPad;

    // uint32 element dtype, taken from a routing tensor (all ui32).
    DType u32 = cast<MO::TensorType>(downOp.getExpertStartIndices().getType())
                    .getDtype();
    auto arrivalTy = MO::BufferType::get(
        MO::TensorType::get(ctx, {arrivalCountSize}, u32, device));
    Value arrivalBuf = rewriter.create<MO::BufferCreateOp>(
        loc, arrivalTy,
        /*initValue=*/
        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), 0));
    // Fresh chain: the buffer is minted here and touched only by the fused op,
    // so there is no prior mutation to sequence against.
    Value inChain = rewriter.create<MO::ChainCreateOp>(loc, ValueRange{});

    auto megaFFN = rewriter.create<MO::Composite::MegaFFNNvfp4Op>(
        loc, /*output=*/resultTy, /*outChain=*/MO::ChainType::get(ctx),
        /*hidden_states=*/l1.getHiddenStates(),
        /*gate_up_weight=*/l1.getWeight(),
        /*gate_up_a_scales=*/l1.getAScales(),
        /*gate_up_b_scales=*/l1.getBScales(),
        /*down_weight=*/downOp.getWeight(),
        /*down_b_scales=*/downOp.getBScales(),
        /*expert_start_indices=*/downOp.getExpertStartIndices(),
        /*expert_ids=*/downOp.getExpertIds(),
        /*a_scale_offsets=*/downOp.getAScaleOffsets(),
        /*gate_up_expert_scales=*/l1.getExpertScales(),
        /*down_expert_scales=*/downOp.getExpertScales(),
        /*c_input_scales=*/l1.getCInputScales(),
        /*estimated_total_m=*/downOp.getEstimatedTotalM(),
        /*gate_up_num_active_experts=*/l1.getNumActiveExperts(),
        /*down_num_active_experts=*/downOp.getNumActiveExperts(),
        /*arrival_count=*/arrivalBuf,
        /*inChain=*/inChain);

    // Forward the clamped-SwiGLU (`swigluoai`) activation selector from L1; the
    // kernel supplies the canonical alpha/limit constants when it is set.
    if (auto clampAttr = l1->getAttr("clamp_activation"))
      megaFFN->setAttr("clamp_activation", clampAttr);
    else
      megaFFN->setAttr("clamp_activation", rewriter.getBoolAttr(false));

    // downOp has a single result; the fused op's outChain is unused (the buffer
    // is touched only here).
    rewriter.replaceOp(downOp, megaFFN.getOutput());
    // L1 is now dead (single-use intermediate, checked above).
    rewriter.eraseOp(l1);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// GEMV + Partial RMS Norm Fusion Pattern
//===----------------------------------------------------------------------===//
//
// Pattern: matmul(x, W.T) where x is [M=1, K] and W is [N, K]
//          followed by RMS norm on the first N_normed columns,
//          with the remaining columns passing through unchanged.
//
// Graph pattern detected:
//   %gemv = mo.matmul(%x, %W) {transpose_b = true} : [1, K] x [N, K] -> [1, N]
//   %normed_slice = mo.slice(%gemv) : [1, N] -> [1, N_normed]
//   %normed = mo.reduce.rms_norm(%normed_slice, %gamma, ...) : [1, N_normed]
//   %unnormed_slice = mo.slice(%gemv) : [1, N] -> [1, N - N_normed]
//
// Fused into:
//   %normed, %unnormed = mo.matmul_fused_partial_rms_norm(%x, %W, %gamma, ...)
//
//===----------------------------------------------------------------------===//

struct MatmulFusedPartialRMSNormPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern,
                                           MO::ReduceRMSNormOp> {
  MatmulFusedPartialRMSNormPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::ReduceRMSNormOp rmsNormOp,
                             MO::DeviceAwareRewriter &rewriter) const override {
    // Only fuse on GPU - the kernel is GPU-only
    auto normedOutputTy = cast<MO::TensorType>(rmsNormOp.getResult().getType());
    DeviceRefAttr device = normedOutputTy.getDeviceRef();
    if (!device || device.getLabel() != M::kGPULabel) {
      return rewriter.notifyMatchFailure(
          rmsNormOp, "GEMV+partial RMS norm fusion only supported on GPU");
    }

    // Check that RMS norm input comes from a split of a matmul output
    Value rmsInput = rmsNormOp.getInput();
    auto rmsInputSplitOp = rmsInput.getDefiningOp<MO::SplitOp>();
    if (!rmsInputSplitOp) {
      return rewriter.notifyMatchFailure(rmsNormOp,
                                         "RMS norm input is not from a split");
    }

    // Check that the split input is from a matmul
    Value splitInput = rmsInputSplitOp.getInput();
    auto matmulOp = splitInput.getDefiningOp<MO::MatmulOp>();
    if (!matmulOp) {
      return rewriter.notifyMatchFailure(rmsNormOp,
                                         "Split input is not from a matmul");
    }

    // Verify matmul has transpose_b = true (required for the kernel)
    if (!matmulOp.getTransposeB()) {
      return rewriter.notifyMatchFailure(
          rmsNormOp, "matmul must have transpose_b=true for fusion");
    }

    // The kernel hardcodes multiply_before_cast=True, so only fuse when
    // the RMS norm has this setting (Gemma-style).
    if (!rmsNormOp.getMultiplyBeforeCast()) {
      return rewriter.notifyMatchFailure(
          rmsNormOp,
          "multiply_before_cast=true required (kernel hardcodes this)");
    }

    // Check matmul output is M=1 (GEMV)
    auto matmulResultTy = cast<MO::TensorType>(matmulOp.getResult().getType());
    auto matmulShape = matmulResultTy.getIntDims();
    if (matmulShape.size() != 2 || matmulShape[0] != 1) {
      return rewriter.notifyMatchFailure(
          rmsNormOp, "matmul must be GEMV (M=1) for fusion");
    }

    // The split must produce exactly 2 results
    if (rmsInputSplitOp.getNumResults() != 2) {
      return rewriter.notifyMatchFailure(
          rmsNormOp, "split must produce exactly 2 results");
    }

    // Check that the split is on the last axis (axis=1 for rank-2). `axis` is
    // a compile-time `index` attribute.
    int64_t axis = normalizeAxis(rmsInputSplitOp.getAxisAttr().getInt(), 2);
    if (axis != 1) {
      return rewriter.notifyMatchFailure(rmsNormOp,
                                         "split must be on the last axis");
    }

    // Get the split sizes - must be constant with 2 elements
    auto splitSizes =
        MO::tryGetConstantIntVector(rmsInputSplitOp.getSplitSizes());
    assert(splitSizes && splitSizes->size() == 2 &&
           "split must have 2 constant sizes");

    // The RMS norm input must be the first result of the split (normed portion)
    // Check that rmsInput is result(0) of the split
    if (rmsInput != rmsInputSplitOp.getResult(0)) {
      return rewriter.notifyMatchFailure(
          rmsNormOp, "RMS norm must consume the first split result");
    }

    // Check normed split result has single use (the rms_norm op)
    if (!rmsInputSplitOp.getResult(0).hasOneUse()) {
      return rewriter.notifyMatchFailure(
          rmsNormOp, "normed split result must have exactly one use");
    }

    // The unnormed output is the second result of the split
    Value unnormedOutput = rmsInputSplitOp.getResult(1);

    // Verify matmul users are exactly the split op
    // This prevents fusion when the GEMV output has other consumers
    for (Operation *user : matmulOp.getResult().getUsers()) {
      if (user != rmsInputSplitOp) {
        return rewriter.notifyMatchFailure(
            rmsNormOp, "matmul has unexpected user (not the split op)");
      }
    }

    // Check device consistency (include all matched ops)
    if (!rewriter.addOps({rmsInputSplitOp, matmulOp})) {
      return rewriter.notifyMatchFailure(
          rmsNormOp, "matched ops have inconsistent devices");
    }

    // Create the fused custom op at the RMSNorm location (default insertion
    // point)
    auto normedResultTy = rmsNormOp.getResult().getType();
    auto unnormedResultTy = cast<MO::TensorType>(unnormedOutput.getType());

    // Create the fused composite op. The kernel hardcodes
    // multiply_before_cast=True and transpose_b=True, so we don't need to pass
    // these as attributes.
    auto fusedOp = rewriter.create<MO::Composite::MatmulFusedPartialRMSNormOp>(
        rmsNormOp->getLoc(), normedResultTy, unnormedResultTy,
        matmulOp.getInputA(), matmulOp.getInputB(), rmsNormOp.getWeight(),
        rmsNormOp.getEpsilon(), rmsNormOp.getWeightOffset());

    // Replace uses - CSE will clean up unused ops
    rewriter.replaceAllUsesWith(rmsNormOp.getResult(), fusedOp.getNormed());
    rewriter.replaceAllUsesWith(unnormedOutput, fusedOp.getUnnormed());

    return mlir::success();
  }
};

struct ConcatSlicePattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern, MO::SliceOp> {
  ConcatSlicePattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::SliceOp sliceOp,
                             MO::DeviceAwareRewriter &rewriter) const override {

    Value input = sliceOp.getInput();
    MO::ConcatOp concatOp = input.getDefiningOp<MO::ConcatOp>();
    if (!concatOp) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "slice input must be a concat");
    }

    // The fused op computes both the concat and the slice output.
    // It's only useful if the concat is used somewhere else.
    if (input.hasOneUse()) {
      return rewriter.notifyMatchFailure(
          sliceOp,
          "the input (ConcatOp) must have another use to be beneficial");
    }

    std::optional<SmallVector<int64_t>> startValues =
        MO::tryGetConstantIntVector(sliceOp.getStart());
    std::optional<SmallVector<int64_t>> stepValues =
        MO::tryGetConstantIntVector(sliceOp.getStep());
    if (!startValues.has_value() || !stepValues.has_value()) {
      return rewriter.notifyMatchFailure(
          sliceOp, "start / step operands must be constants");
    }

    if (isTrivialSlice(sliceOp, *stepValues)) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "for trivial slice we shouldn't fuse");
    }

    for (int64_t s : *stepValues) {
      if (s < 0) {
        return rewriter.notifyMatchFailure(sliceOp,
                                           "negative step isn't supported");
      }
    }

    auto innerInputDim =
        sliceOp.getInput().getType().getTypedShapeAttr().getValues().back();
    auto innerOutputDim =
        sliceOp.getResult().getType().getTypedShapeAttr().getValues().back();
    if (innerInputDim != innerOutputDim) {
      return rewriter.notifyMatchFailure(
          sliceOp, "slicing the inner dimension isn't supported");
    }

    auto startsAttr = rewriter.rewriter.getI64ArrayAttr(*startValues);
    auto stepsAttr = rewriter.rewriter.getI64ArrayAttr(*stepValues);

    auto fusedOp = rewriter.create<MO::Composite::FusedConcatSliceOp>(
        sliceOp->getLoc(), concatOp.getResult().getType(),
        sliceOp.getResult().getType(), concatOp.getAxisAttr().getValue(),
        concatOp.getInputs(), startsAttr, stepsAttr,
        concatOp.getOutputParamDecls());

    rewriter.replaceAllUsesWith(concatOp.getResult(),
                                fusedOp.getConcatResult());
    rewriter.replaceOp(sliceOp, fusedOp.getSliceResult());
    return mlir::success();
  }

private:
  // returns true if the slice can be implemented with a buffer view operation
  // (update offset).
  static bool isTrivialSlice(MO::SliceOp sliceOp, ArrayRef<int64_t> step) {
    for (int64_t s : step) {
      if (s != 1)
        return false;
    }

    TensorType inputTy = sliceOp.getInput().getType();
    TensorType resultTy = sliceOp.getResult().getType();
    unsigned rank = inputTy.getRank();
    bool nonOneDim = false;

    for (unsigned i = 0; i < rank; ++i) {
      bool dimMightBeSliced = inputTy.getTypedShapeAttr().getValues()[i] !=
                              resultTy.getTypedShapeAttr().getValues()[i];

      if (dimMightBeSliced && nonOneDim)
        return false;

      if (inputTy.getIntDim(i) != 1)
        nonOneDim = true;
    }

    return true;
  }
};

// Fuse matmuls -> reduce_scatter pattern (with optional residual_add) into a
// single mo.composite.distributed.matmul_reduce_scatter.sum op.
//
// At most one input to the reduce_scatter may carry a residual add (the
// asymmetric DeepseekV3/KimiK2.5 TP+EP pattern). The fused kernel
// hardcodes `transpose_b=true` for the matmul and expects each B in [N, K].
//
// GPU-only: the fused kernel currently exists only for SM100 (B200). For now,
// we only refuse to fuse on non-GPU targets, but ideally we would filter by
// arch.
struct FuseMatmulReduceScatterPattern
    : public OpRewritePattern<MO::ReduceScatterSumOp> {
  DeviceRefAttr targetDevice;
  FuseMatmulReduceScatterPattern(MLIRContext *ctx, DeviceRefAttr targetDevice)
      // This could inherit from DeviceAwareRewritePattern but it trips a
      // single-device assert
      : OpRewritePattern(ctx, 2), targetDevice(targetDevice) {}

  LogicalResult matchAndRewrite(MO::ReduceScatterSumOp rsOp,
                                PatternRewriter &rewriter) const override {
    // Check we're building for GPU
    DeviceRefAttr device;
    if (!rsOp.getInputs().empty())
      device = cast<TensorValue>(rsOp.getInputs()[0]).getType().getDeviceRef();
    if (!device)
      device = targetDevice;
    if (!device || device.getLabel() != M::kGPULabel)
      return rewriter.notifyMatchFailure(rsOp, "fused op requires GPU");

    unsigned numDevices = rsOp.getInputs().size();
    if (numDevices == 0)
      return rewriter.notifyMatchFailure(rsOp, "no inputs");
    if (rsOp.getAxis() != 0)
      return rewriter.notifyMatchFailure(rsOp, "axis != 0");
    // Grouped reduce-scatter is not supported yet for the fused kernel.
    if (rsOp.getGroupSizeAttr().getInt() != numDevices)
      return rewriter.notifyMatchFailure(rsOp,
                                         "grouped reduce-scatter not fused");

    // Match each input to the reduce_scatter. Each must come from either a
    // bare matmul or an add(residual, matmul). At most one input may have
    // a residual add.
    llvm::DenseMap<Value, MO::MatmulOp> matmulOps;
    Value residualValue;
    // Index of the input (and thus device) carrying the residual add. Only
    // meaningful when residualValue is set; defaults to 0 otherwise.
    int64_t residualPeer = 0;
    for (auto [idx, in] : llvm::enumerate(rsOp.getInputs())) {
      auto tv = cast<TensorValue>(in);
      if (matchMatmulInput(tv, matmulOps))
        continue;
      // matchAddMatmulInput allows at most one residual
      // across all inputs (it returns false once one is claimed), so a second
      // residual-bearing input fails to match here and we bail out of fusion.
      if (!matchAddMatmulInput(tv, matmulOps, residualValue))
        return rewriter.notifyMatchFailure(rsOp, "input not matchable");
      // Reached only on the single input that carries the residual.
      residualPeer = static_cast<int64_t>(idx);
    }
    if (matmulOps.size() != numDevices)
      return rewriter.notifyMatchFailure(rsOp, "not all inputs matched");

    llvm::SmallVector<Value> inputsA;
    llvm::SmallVector<Value> inputsB;
    llvm::SmallVector<Type> outputTypes;
    inputsA.reserve(numDevices);
    inputsB.reserve(numDevices);
    outputTypes.reserve(numDevices);
    for (Value in : rsOp.getInputs()) {
      MO::MatmulOp matmul = matmulOps.at(in);
      inputsA.push_back(matmul.getInputA());
      inputsB.push_back(matmul.getInputB());
    }
    for (Value out : rsOp.getOutputs())
      outputTypes.push_back(out.getType());

    // Single residual tensor: the real residual if matched, otherwise a
    // dummy (first matmul result). has_residual indicates whether to use it.
    bool hasResidual = residualValue != nullptr;
    Value residual = hasResidual
                         ? residualValue
                         : matmulOps.at(rsOp.getInputs()[0]).getResult();

    auto fusedOp = MO::Composite::DistributedMatmulReduceScatterSumOp::create(
        rewriter, rsOp.getLoc(),
        /*outputs=*/outputTypes,
        /*outChain=*/rsOp.getOutChain().getType(),
        /*inputsA=*/inputsA,
        /*inputsB=*/inputsB,
        /*residual=*/residual,
        /*has_residual=*/rewriter.getBoolAttr(hasResidual),
        /*residual_peer=*/rewriter.getI64IntegerAttr(residualPeer),
        /*signalBuffers=*/rsOp.getSignalBuffers(),
        /*inChain=*/rsOp.getInChain());

    SmallVector<Value> replacements(fusedOp.getOutputs().begin(),
                                    fusedOp.getOutputs().end());
    replacements.push_back(fusedOp.getOutChain());
    rewriter.replaceOp(rsOp, replacements);
    return mlir::success();
  }

private:
  static bool isValidMatmul(MO::MatmulOp matmul) {
    if (matmul.getPackedB())
      return false;
    if (matmul.getTransposeB())
      return true;
    // MatmulOp::canonicalize doesn't run between rewriter iterations,
    // so transpose_b won't be set. Check NK layout instead, set by
    // MatmulTranspBPattern
    auto bLayout = matmul.getInputB().getType().getLayoutAttr();
    return bLayout && bLayout.getFormat() == "NK";
  }

  static bool matchMatmulInput(TensorValue reduceScatterIn,
                               llvm::DenseMap<Value, MO::MatmulOp> &out) {
    auto matmul = reduceScatterIn.getDefiningOp<MO::MatmulOp>();
    if (!matmul || !isValidMatmul(matmul))
      return false;
    // Single-use: fusing would drop a use otherwise.
    if (!reduceScatterIn.hasOneUse())
      return false;
    out[reduceScatterIn] = matmul;
    return true;
  }

  // Match add(residual, matmul) or add(matmul, residual) feeding the RS.
  static bool matchAddMatmulInput(TensorValue reduceScatterIn,
                                  llvm::DenseMap<Value, MO::MatmulOp> &out,
                                  Value &residualValue) {
    if (residualValue)
      return false; // already claimed.
    auto addOp = reduceScatterIn.getDefiningOp<MO::AddOp>();
    if (!addOp)
      return false;
    if (!reduceScatterIn.hasOneUse())
      return false;
    for (auto [matmulCandidate, residualCandidate] :
         {std::pair{addOp.getLhsInput(), addOp.getRhsInput()},
          std::pair{addOp.getRhsInput(), addOp.getLhsInput()}}) {
      auto matmul = matmulCandidate.getDefiningOp<MO::MatmulOp>();
      if (!matmul || !isValidMatmul(matmul))
        continue;
      if (!matmulCandidate.hasOneUse())
        continue;
      out[reduceScatterIn] = matmul;
      residualValue = residualCandidate;
      return true;
    }
    return false;
  }
};

} // namespace

void applyFuseAllReduceAddRMSNormQuantFP8(MO::AllReduceSumOp allReduceOp,
                                          M::DeviceRefAttr targetDeviceRef) {
  FuseAllReduceAddRMSNormQuantFP8PatternMatcher fuser(allReduceOp,
                                                      targetDeviceRef);
  fuser.apply();
}

void applyFuseParallelBundledAllReduceAddRMSNormQuantFP8(
    MO::BundledAllReduceSumOp bundledAllReduceOp) {
  FuseParallelBundledAllReduceAddRMSNormQuantFP8PatternMatcher(
      bundledAllReduceOp)
      .apply();
}

void populateFusionPatterns(mlir::RewritePatternSet &patterns,
                            PackParamCache &cache,
                            M::DeviceRefAttr targetDeviceRef,
                            const MHAOptions &mhaOpts) {
  mlir::MLIRContext *ctx = patterns.getContext();

  patterns.add<MatmulConstBPattern>(ctx, cache, targetDeviceRef);
  patterns.add<ConvConstFilterPattern>(ctx, targetDeviceRef);

  patterns.add<GatherSumPattern, DecomposeSmallReduceAddPattern,
               MatmulTranspBPattern, ConcatZeroAxisPattern,
               ConvTransposeConstFilterPattern, SplitZeroAxisPattern,
               LayerNormPattern, RMSNormFusedResidualAddPattern,
               RMSNormResidualAddPattern, ReduceRMSNormRoPEPattern,
               ConcatSlicePattern, MatmulFusedPartialRMSNormPattern>(ctx);

  patterns.add<ForceMatmulTransposeBForGPU, BatchMatmulAdjBPattern,
               RMSNormFusedQuantizeDynamicScaledFP8Pattern,
               RMSNormFusedQuantizeDynamicBlockScaledPattern,
               FuseMatmulReduceScatterPattern>(ctx, targetDeviceRef);

  if (mhaOpts.enable) {
    patterns.add<MultiHeadAttentionPattern>(ctx, mhaOpts.allowUnsafeBCast,
                                            mhaOpts.allowInternalReshape,
                                            mhaOpts.debug, targetDeviceRef);
  }
}

void populateFusionMatmulAddPatterns(const DeviceInfoMapping &deviceInfoMapping,
                                     mlir::RewritePatternSet &patterns) {
  patterns.add<MatmulAddPattern>(patterns.getContext(), deviceInfoMapping);
}

void populateFusionMegaFFNPatterns(const DeviceInfoMapping &deviceInfoMapping,
                                   mlir::RewritePatternSet &patterns) {
  patterns.add<MegaFFNNvfp4Pattern>(patterns.getContext(), deviceInfoMapping);
}

} // namespace M::MO
