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
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GenericML/GraphCompiler/Matchers.h"
#include "GenericML/Support/DataflowUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/DeviceSpecs.h"
#include "Support/ML/Permutation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>

using namespace M;
using namespace M::MO;
using namespace GraphCompiler::Matchers;

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

// TODO(#29430) make this more general and apply to all shape function calls.
struct PackParamCache {
  /// Associate `outputTy` with `(inputTy, kgenParams)` as a new cache entry.
  void put(MO::TensorType inputTy, DictionaryAttr kgenParams,
           MO::TensorType outputTy) {
    cache[inputTy][kgenParams] = outputTy;
  }

  /// Return the output type associated with `(inputTy, kgenParams)`, or null if
  /// the entry doesn't exist.
  MO::TensorType get(MO::TensorType inputTy, DictionaryAttr kgenParams) {
    auto cacheIt = cache.find(inputTy);
    if (cacheIt == cache.end())
      return {};

    auto &kgenParamsToOutTy = cacheIt->second;
    auto it = kgenParamsToOutTy.find(kgenParams);
    if (it == kgenParamsToOutTy.end())
      return {};

    return it->second;
  }

private:
  // Output tensor type of a packing operation is determined solely by the input
  // tensor type and the extra kgen parameters. Therefore we can cache the
  // output tensor type based on the (input_ty, kgen_params) tuple.
  DenseMap<MO::TensorType, DenseMap<DictionaryAttr, MO::TensorType>> cache;
};

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
    rewriter.modifyOpInPlace(op, [&] { op.setOperand(1, rebWeight); });

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
    std::optional<int64_t> axis =
        MO::tryGetConstantInt(cast<MO::TensorValue>(concat.getAxis()));
    if (!axis.has_value() || axis.value() != 0)
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

    if (filterLayout.getFormat() == "FCRS")
      return rewriter.notifyMatchFailure(op,
                                         "Packing FCRS format isn't supported");

    if (!dependsOnNoAliasConstantLikesOnly(filter.getDefiningOp()))
      return rewriter.notifyMatchFailure(op, "filter isn't constant");

    auto maybeNumGroups = MO::tryGetConstantInt(op.getNumGroups());
    if (!maybeNumGroups) {
      return rewriter.notifyMatchFailure(
          op, "currently only pack constant num_groups");
    }

    if (!rewriter.deviceAnnots.isConsistentWith(getHostDevice(op))) {
      // TODO(#25114): Support outside of host.
      return rewriter.notifyMatchFailure(
          op, "packing is only supported on the host device");
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
    std::optional<int64_t> axis =
        MO::tryGetConstantInt(cast<MO::TensorValue>(op.getAxis()));
    if (!axis.has_value() || axis.value() != 0)
      return rewriter.notifyMatchFailure(op, "axis not constant 0");

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

    auto lastAxis = m_Optional(m_ScalarConstant(2), m_ScalarConstant(-1));
    // TODO: Current capture system can't distinguish multiple uses of the same
    // value in the pattern, need to fix, this might be slightly risky
    Value input;
    auto x = m_Ranked(3, m_Any(&input));
    auto mean = m_Op<MO::Static::BroadcastToOp>(
        po(), m_Op<MO::MeanOp>(po(), x, lastAxis));
    auto diff = m_Op<MO::SubOp>(po(), x, mean);
    auto square = m_Optional(
        m_Op<MO::MulOp>(po(), diff, diff),
        m_Op<MO::PowOp>(
            po(), diff,
            m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                2)));
    auto var = m_Op<MO::MeanOp>(po(), square, lastAxis);
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
  TensorValue softmaxAxis;
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
    auto pSoftmaxAttnMatrix = m_Op<SoftmaxOp>(
        &matcher.softmaxProducer, maybeBCast1, m_Any(&matcher.softmaxAxis));

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

    std::optional<int64_t> axis = MO::tryGetConstantInt(matcher.softmaxAxis);
    if (!axis.has_value() ||
        size_t(axis.value()) != matcher.softmaxInput.getType().getRank() - 1)
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

    // Inserts a runtime guard checking (beforeShape == afterShape)
    IntegerType boolTy = rewriter.rewriter.getI1Type();
    TypedAttr cmpAttr =
        KGEN::emitMLIROperationCall(MOSH::EqualOp::getOperationName(), {},
                                    {beforeShape, afterShape}, boolTy);
    KGEN::ParamDeclAttr condAttr = graphOp.getNewParamOfType(checkName, boolTy);
    rewriter.create<KGEN::ParamDeclareOp>(loc, condAttr, cmpAttr);

    // Add runtime assertion into graph
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
    SmallVector<Value> customOpInputs = {query, key, value};

    if (matcher.hasAttentionMask()) {
      customOpInputs.push_back(matcher.getAttentionMask());
    }

    // Kernel expects FP32 scale
    auto normScalar = cast<TensorValue>(matcher.getNormScalar());
    TensorValue normScalarAsFP32 = rewriter.create<CastOp>(
        rootOp.getLoc(), normScalar.getType().cloneWith(DType::f32),
        normScalar);

    // The scale operand to all variants of masked_flash_attention are scalar
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

    customOpInputs.push_back(normScalarAsFP32);

    StringRef kernelName;
    if (isCPUDevice) {
      kernelName = matcher.hasAttentionMask() ? "with_mask_flash_attention_cpu"
                                              : "no_mask_flash_attention_cpu";
    } else {
      kernelName = "masked_flash_attention_gpu";
    }

    auto customOp = rewriter.create<CustomOp>(
        rootOp.getLoc(), TypeRange{rootOp.getType().cloneWith(mhaOutputShape)},
        customOpInputs, StringAttr::get(getContext(), kernelName));
    auto mhaResult = cast<TensorValue>(customOp.getResult(0));

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
    // we assume the transpose is to BSHD and insert a dynamic runtime check
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

    if (dimD != 128 || !isDimHStatic)
      return Error("Right now must prove D == 128 && static H for llama2.");

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

    // If we have unsafe broadcasts, we want to add runtime guards against them.
    for (Static::BroadcastToOp unsafeBCast : matcher->getUnsafeBroadcasts())
      guardUnsafeBCast(unsafeBCast, rewriter);

    rewriteRootOp(op, rewriter, *matcher, isCPUDevice);

    return mlir::success();
  }
};

struct RMSNormFusedResidualAddPattern
    : public MO::DeviceAwareRewritePattern<OpRewritePattern, MO::CustomOp> {
  RMSNormFusedResidualAddPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 3) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::CustomOp op,
                             MO::DeviceAwareRewriter &rewriter) const override {
    // Match the pattern: rms_norm(rms_norm(input) + residual_input)
    // This means we're looking for a CustomOp with symbol "rms_norm" that takes
    // as input the result of (rms_norm(input) + residual_input)

    if (op.getSymbol() != "rms_norm")
      return rewriter.notifyMatchFailure(op, "not an rms_norm operation");

    // Check that we have the right number of operands for rms_norm
    if (op.getOperands().size() != 4)
      return rewriter.notifyMatchFailure(op, "rms_norm expects 4 operands");

    // The first operand should be the result of (rms_norm(input) +
    // residual_input)
    Value firstOperand = op.getOperand(0);

    // Look for an add operation
    auto addOp = firstOperand.getDefiningOp<MO::AddOp>();
    if (!addOp)
      return rewriter.notifyMatchFailure(
          op, "first operand not from add operation");

    // One of the add operands should be the original input
    if (addOp.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "add operation expects 2 operands");

    Value innerRmsNormResult = addOp.getOperand(0);
    Value residualInput = addOp.getOperand(1);
    auto innerRmsNorm = innerRmsNormResult.getDefiningOp<MO::CustomOp>();
    if (!innerRmsNorm || innerRmsNorm.getSymbol() != "rms_norm") {
      innerRmsNormResult = addOp.getOperand(1);
      residualInput = addOp.getOperand(0);
      innerRmsNorm = innerRmsNormResult.getDefiningOp<MO::CustomOp>();
      if (!innerRmsNorm || innerRmsNorm.getSymbol() != "rms_norm")
        return rewriter.notifyMatchFailure(op, "inner rms_norm not found");
    }

    // Check that we have the right number of operands for inner rms_norm
    if (innerRmsNorm.getOperands().size() != 4)
      return rewriter.notifyMatchFailure(op,
                                         "inner rms_norm expects 4 operands");

    // Now we have the pattern: rms_norm(rms_norm(input, gamma1, eps1,
    // weight_offset1) + residual_input) We need to extract the parameters from
    // both rms_norm operations
    Value originalInput = innerRmsNorm.getOperand(0);

    // Extract parameters from outer rms_norm (gamma2, eps2, weight_offset2)
    Value gamma2 = op.getOperand(1);
    Value eps2 = op.getOperand(2);
    Value weightOffset2 = op.getOperand(3);

    // Extract parameters from inner rms_norm (gamma1, eps1, weight_offset1)
    Value gamma1 = innerRmsNorm.getOperand(1);
    Value eps1 = innerRmsNorm.getOperand(2);
    Value weightOffset1 = innerRmsNorm.getOperand(3);

    // Check device consistency
    if (!rewriter.addOps({addOp, innerRmsNorm})) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    // Create the fused operation
    SmallVector<Value> fusedOperands = {
        originalInput, // input
        residualInput, // residual_input
        gamma1,        // gamma1
        gamma2,        // gamma2
        eps1,          // epsilon1
        eps2,          // epsilon2
        weightOffset1, // weight_offset1
        weightOffset2  // weight_offset2
    };

    // The fused operation returns two outputs: [output, intermediate]
    // where output = rms_norm(rms_norm(input) + residual)
    // and intermediate = rms_norm(input) + residual
    SmallVector<Type> fusedResultTypes = {
        op.getResultTypes()[0],    // output type (same as outer rms_norm)
        addOp->getResultTypes()[0] // intermediate type (same as add result)
    };

    auto fusedOp = rewriter.create<MO::CustomOp>(
        op.getLoc(), fusedResultTypes, fusedOperands,
        StringAttr::get(getContext(), "rms_norm_fused_residual_add"),
        op.getDevice(), op.getParameters(), op.getOutputParamDecls());

    // Replace the outer rms_norm with the first output of the fused operation
    rewriter.replaceOp(op, fusedOp.getResult(0));

    // Replace all consumers of the add operation with the second output of the
    // fused operation
    rewriter.replaceAllUsesWith(addOp.getResult(), fusedOp.getResult(1));

    return mlir::success();
  }
};

} // namespace

struct PatternFusionPass : public impl::PatternFusionBase<PatternFusionPass> {
  using PatternFusionBase::PatternFusionBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns{ctx};

    DeviceRefAttr targetDeviceRef =
        targetDevice == "" ? nullptr
                           : DeviceRefAttr::get(ctx, targetDevice.getValue(),
                                                targetDeviceId.getValue());

    PackParamCache cache;
    patterns.add<MatmulConstBPattern>(ctx, cache, targetDeviceRef);

    patterns.add<ConvConstFilterPattern>(ctx, targetDeviceRef);

    patterns.add<GatherSumPattern, MatmulTranspBPattern, ConcatZeroAxisPattern,
                 ConvTransposeConstFilterPattern, SplitZeroAxisPattern,
                 LayerNormPattern, RMSNormFusedResidualAddPattern>(ctx);

    patterns.add<ForceMatmulTransposeBForGPU, BatchMatmulAdjBPattern>(
        ctx, targetDeviceRef);

    if (enableMHAPattern.getValue()) {
      patterns.add<MultiHeadAttentionPattern>(
          ctx, mhaAllowUnsafeBCast.getValue(),
          mhaAllowInternalReshape.getValue(), debugMHAPattern.getValue(),
          targetDeviceRef);
    }

    MO::GraphOp graphOp = getOperation();
    if (failed(applyPatternsGreedily(graphOp, std::move(patterns))))
      return signalPassFailure();
    MO::synchronizeMetadataUpdates(graphOp);
  }
};

} // namespace M::MOGG
