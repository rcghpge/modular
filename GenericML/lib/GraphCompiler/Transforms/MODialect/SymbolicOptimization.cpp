//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "GenericML/GraphCompiler/Analysis/MODialect/StaticReshapeDimensionsMappingAnalysis.h"
#include "GenericML/GraphCompiler/MODialect/MODevices.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/GraphCompiler/Matchers.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "GenericML/Support/DataflowUtils.h"
#include "GenericML/Support/Dims.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/AssertStream.h"
#include "Support/ML/Permutation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "./Utils.h"

#include <numeric>

#define DEBUG_TYPE "symbolic-optimization"

using namespace M;
using namespace MO;
using namespace graph_compiler::matchers;

namespace M::MO {
#define GEN_PASS_DEF_SYMBOLICOPTIMIZATION
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

} // namespace M::MO

namespace {

/// A pattern that detects `mo.arg_nonzero` ops preceded by a flattening reshape
/// and a rank-increasing broadcast, and rewrites them so that the broadcast
/// happens after the `mo.arg_nonzero` op.
///
/// This decreases the workload on `mo.arg_nonzero` and improves the compilers
/// ability to constant fold before and fuse after the arg_nonzero op.
///
/// Example:
///
/// input : tensor<[2, 3]>
///
///   bcast = broadcast(input) : ... -> tensor<[5, 2, 3]>
///   flat  = reshape(bcast)   : ... -> tensor<[30]>
///   res   = arg_nonzero(flat)      : ...
/// ===>
///   flat        = reshape(input)   : ... -> tensor<[6]>
///   arg_nonzero = arg_nonzero(flat)      : ...
///   res         = arg_nonzero + broadcast(range(...)))
struct CommuteBroadcastWithFlatennedArgNonzero
    : DeviceAwareRewritePattern<OpRewritePattern, ArgNonzeroOp> {
  CommuteBroadcastWithFlatennedArgNonzero(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 4) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::ArgNonzeroOp argNonzeroOp,
                             DeviceAwareRewriter &rewriter) const override {
    Location loc = argNonzeroOp.getLoc();

    TensorValue inputVal = argNonzeroOp.getInput();
    TensorType inputTy = inputVal.getType();
    if (!inputTy.hasRank() || inputTy.getRank() != 1)
      return rewriter.notifyMatchFailure(loc, "input is not rank-1");

    auto reshapeOp = inputVal.getDefiningOp<Static::ReshapeOp>();
    if (!reshapeOp)
      return rewriter.notifyMatchFailure(loc, "not preceded by static reshape");

    TensorValue reshapeInput = reshapeOp.getInput();
    TensorType reshapeInputTy = reshapeInput.getType();
    if (!reshapeInputTy.hasRank()) {
      return rewriter.notifyMatchFailure(
          loc, "preceding reshape has dynamic input rank");
    }

    auto bcastOp = reshapeInput.getDefiningOp<Static::BroadcastToOp>();
    if (!bcastOp)
      return rewriter.notifyMatchFailure(loc, "not preceded by static bcast");

    TensorValue bcastInput = bcastOp.getInput();
    TensorType bcastInputTy = bcastInput.getType();
    if (!bcastInputTy.hasRank()) {
      return rewriter.notifyMatchFailure(
          loc, "preceding static bcast has dynamic input rank");
    }

    // We check that the broadcast increases rank. This is the exit condition.
    int64_t rankDiff =
        (int64_t)reshapeInputTy.getRank() - (int64_t)bcastInputTy.getRank();
    if (rankDiff <= 0) {
      return rewriter.notifyMatchFailure(
          loc, "preceding static bcast does not increase rank");
    }

    if (!rewriter.addOps({reshapeOp, bcastOp})) {
      return rewriter.notifyMatchFailure(
          loc, "matched ops have inconsistent devices");
    }

    // The pattern is now matched. We start by flattening the input of the
    // broadcast, i.e. we roughly implement this:
    // flattened = bcastInput.flatten()
    // NOTE: FS/fs stands for flat size.
    GraphOp graphOp = argNonzeroOp->getParentOfType<GraphOp>();
    KGEN::ParamDeclAttr fsDecl = graphOp.getNewDimParam("FS");
    ArrayRef<TypedAttr> reshapeInputDims = reshapeInputTy.getDims();
    ArrayRef<TypedAttr> trailingDims = reshapeInputDims.drop_front(rankDiff);
    auto fsAttr = KGEN::ParamOperatorAttr::get(KGEN::POC::MulNuw, trailingDims);
    rewriter.create<KGEN::ParamDeclareOp>(loc, fsDecl, fsAttr);

    auto flattenedTy = TensorType::get(
        SmallVector<TypedAttr>{KGEN::ParamDeclRefAttr::get(fsDecl)},
        bcastInputTy.getElementType());
    auto flattened =
        rewriter.create<Static::ReshapeOp>(loc, flattenedTy, bcastInput);

    // We now find the nonzero elements in the flattened (unbroadcasted) input.
    // We roughly implement the following:
    // argNonzero_res = np.expand_dims(np.argNonzero(flattened)[0], axis=1)
    // Note we refer to the number of unique elements as W.
    KGEN::ParamDeclAttr wDecl = graphOp.getNewDimParam("W");
    auto wRef = KGEN::ParamDeclRefAttr::get(wDecl);
    IntegerType si64Ty = rewriter.getIntegerType(64, /*isSigned=*/true);
    SmallVector<TypedAttr> argNonzeroDims{wRef, rewriter.getIndexAttr(1)};
    auto argNonzeroTy = TensorType::get(argNonzeroDims, si64Ty);
    auto newArgNonzeroOp = rewriter.create<ArgNonzeroOp>(
        loc, argNonzeroTy, flattened, ParamDeclArrayAttr::get(wDecl));

    // Next we create the first auxiliary value, roughly the following range:
    // range = np.arange(0, FS*BS, FS)
    // First, we need a constant zero for the start index.
    auto start = rewriter.create<ConstantOp>(loc, (int64_t)0);

    // Second, we calculate the batch size symbolically (BS/bs).
    ArrayRef<TypedAttr> batchDims = reshapeInputDims.take_front(rankDiff);
    auto bsAttr = KGEN::ParamOperatorAttr::get(KGEN::POC::MulNuw, batchDims);

    // We can now calculate the range limit (FS*BS) symbolically.
    SmallVector<TypedAttr> limFactors{bsAttr, fsAttr};
    auto limAttr = KGEN::ParamOperatorAttr::get(KGEN::POC::MulNuw, limFactors);
    auto lim = rewriter.create<MOSH::ParamToValueOp>(loc, limAttr);
    auto limT = rewriter.create<IndexToTensorOp>(loc, start.getType(), lim);

    // Finally, we use the flat size as the step size.
    auto step = rewriter.create<MOSH::ParamToValueOp>(loc, fsAttr);
    auto stepT = rewriter.create<IndexToTensorOp>(loc, start.getType(), step);

    // We can now construct the range.
    KGEN::ParamDeclAttr bsDecl = graphOp.getNewDimParam("BS");
    rewriter.create<KGEN::ParamDeclareOp>(loc, bsDecl, bsAttr);
    auto bsRef = KGEN::ParamDeclRefAttr::get(bsDecl);
    auto rangeTy = TensorType::get(SmallVector<TypedAttr>{bsRef}, si64Ty);
    auto range = rewriter.create<RangeOp>(loc, rangeTy, start, limT, stepT);

    // We add a trailing 1-dimension to the range, roughly:
    // rangeExpanded = np.expand_dims(range, axis=1)
    auto expandedTy = TensorType::get(
        SmallVector<TypedAttr>{bsRef, rewriter.getIndexAttr(1)}, si64Ty);
    auto rangeExpanded =
        rewriter.create<Static::ReshapeOp>(loc, expandedTy, range);

    // We will now add the expanded range to the transpose of the new argNonzero
    // op's output, more or less like this: commonShape =
    // np.broadcast_shapes(argNonzero_res.T.shape, aux2.shape) indexes =
    // np.broadcast_to(argNonzero_res.T, commonShape)
    //            + np.broadcast_to(aux2, commonShape)
    // Notice that the common shape is always [BS, W].

    // We start with transposing the result of argNonzero.
    auto perm = rewriter.create<ConstantOp>(loc, SmallVector<int64_t>{1, 0});
    auto transposeTy = TensorType::get(
        SmallVector<TypedAttr>{argNonzeroDims[1], argNonzeroDims[0]}, si64Ty);
    auto wTranspose =
        rewriter.create<TransposeOp>(loc, transposeTy, newArgNonzeroOp, perm);

    // Then we add the broadcasted transposed indexed and the broadcasted
    // expanded range.
    auto bcastResTy =
        TensorType::get(SmallVector<TypedAttr>{bsRef, wRef}, si64Ty);
    auto indexes = rewriter.create<AddOp>(
        loc,
        rewriter.create<Static::BroadcastToOp>(loc, bcastResTy, wTranspose),
        rewriter.create<Static::BroadcastToOp>(loc, bcastResTy, rangeExpanded));

    // Make sure we don't drop parameter declarations from the original
    // argNonzero.
    ArrayRef<ParamDeclAttr> oldDecls = argNonzeroOp.getOutputParamDecls();
    ASSERT_STREAM(oldDecls.size() == 1,
                  << "mo.argNonzero declares more than one new param");
    rewriter.modifyOpInPlace(argNonzeroOp,
                             [&]() { argNonzeroOp.setOutputParamDecls({}); });
    auto numResAttr = KGEN::ParamOperatorAttr::get(
        KGEN::POC::MulNuw, SmallVector<TypedAttr>{bsAttr, wRef});
    rewriter.create<KGEN::ParamDeclareOp>(loc, oldDecls[0], numResAttr);

    // Replace the original argNonzero op, with the new result.
    rewriter.replaceOpWithNewOp<Static::ReshapeOp>(
        argNonzeroOp, argNonzeroOp.getType(), indexes);
    return success();
  }
};

/// A pattern that detects batch matmul with the rhs coming from a broadcast on
/// the batch axis, and rewrites them into a regular 2D matmul with a reshape
/// that flattens the batch axis of lhs and another reshape that restores the
/// batch axis of the final result.
///
/// This enables more optimizations on matmul.
///
/// Example:
///
/// lhs : tensor<[4, 3, 2]>
/// rhs : tensor<[2, 3]>
///
///   bcast = broadcast(rhs) : ... -> tensor<[4, 2, 3]>
///   res   = batch_matmul(lhs, rhs) : ... -> tensor<[4, 3, 3]>
/// ===>
///   flat_lhs = reshape(lhs) : ... -> tensor<[12, 2]>
///   res_2d   = matmul(flat_lhs, rhs) : ... -> tensor<[12, 3]>
///   res      = reshape(res_2d) : ... -> tensor<[4, 3, 3]>
struct ConvertBatchMatmulToMatmulPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, BatchMatmulOp> {
  ConvertBatchMatmulToMatmulPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 4) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::BatchMatmulOp op,
                             DeviceAwareRewriter &rewriter) const override {
    Value input, weight;
    Operation *broadcast = nullptr;
    auto weightBcastP =
        m_Op<Static::BroadcastToOp>(&broadcast, m_MOMatrix(m_Any(&weight)));
    if (!m_Op<BatchMatmulOp>(
             m_Any(&input),
             m_Optional(m_Op<RebindOp>(rewriter.pushOp(), weightBcastP),
                        weightBcastP))
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

    if (!rewriter.addOps({broadcast})) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    ArrayRef<TypedAttr> oldShape = inputTy.getDims();
    ArrayRef<TypedAttr> kVec = oldShape.take_back();
    TypedAttr batch =
        KGEN::ParamOperatorAttr::get(KGEN::POC::MulNuw, oldShape.drop_back(1));
    SmallVector<TypedAttr> newShape{batch, kVec[0]};

    auto shapeTy = rewriter.getType<ShapeType>();
    MO::TensorType reshapeTy =
        inputTy.cloneWith(ShapeAttr::get(newShape, shapeTy));
    Value reshape =
        rewriter.create<Static::ReshapeOp>(op->getLoc(), reshapeTy, input)
            .getResult();

    ArrayRef<TypedAttr> weightShape = weightTy.getDims();
    MO::TensorType matmulTy =
        bmmTy.cloneWith(ShapeAttr::get({newShape[0], weightShape[1]}, shapeTy));

    Value matmul =
        rewriter.create<MatmulOp>(op->getLoc(), matmulTy, reshape, weight)
            .getResult();

    rewriter.replaceOpWithNewOp<Static::ReshapeOp>(op, bmmTy, matmul);

    return mlir::success();
  }
};

/// A pattern that detects batch matmul batch size one and converts the op to
/// a normal matmul. Note if the broadcast cannot be sunk, it will be converted
/// to a reshape.
///
/// Example:
///
///   lhs : tensor<[1, 1, 3, 2]>
///   rhs : tensor<[1, 1, 2, 3]>
///
///   res   = batch_matmul(lhs, rhs) : ... -> tensor<[1, 1, 2, 2]>
/// ===>
///   reshape_lhs = reshape(lhs) : ... -> tensor<[3, 2]>
///   reshape_rhs = reshape(rhs) : ... -> tensor<[2, 3]>
///   matmul      = matmul(reshape_lhs, reshape_rhs) : ... -> tensor<[2, 2]>
///   res         = broadcast_to(matmul) : ... -> tensor<[1, 1, 2, 2]>
struct ConvertBatchMatmulBatchSizeOneToMatmulPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, BatchMatmulOp> {
  ConvertBatchMatmulBatchSizeOneToMatmulPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 5) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::BatchMatmulOp op,
                             DeviceAwareRewriter &rewriter) const override {

    TensorType aType = op.getTensorA().getType();
    TensorType bType = op.getTensorB().getType();
    TensorType cType = op.getResult().getType();

    if (!aType.hasRank())
      return rewriter.notifyMatchFailure(
          op, "rank must be known in order to match");

    auto rank = aType.getRank();
    // Verifier ensures rank >= 3, so requiredOneDims >= 1
    auto requiredOneDims = rank - 2;

    for (size_t i = 0, e = (requiredOneDims); i < e; i++)
      if ((aType.getIntDim(i) != 1) || (bType.getIntDim(i) != 1))
        return rewriter.notifyMatchFailure(op, "failed to match pattern");

    auto resTy = op.getResult().getType();

    SmallVector<TypedAttr> aShapeDims;
    SmallVector<TypedAttr> bShapeDims;
    SmallVector<TypedAttr> resShapeDims;
    for (auto i = requiredOneDims; i < rank; i++) {
      aShapeDims.push_back(aType.getDim(i));
      bShapeDims.push_back(bType.getDim(i));
      resShapeDims.push_back(resTy.getDim(i));
    }

    auto shapeTy = rewriter.getType<ShapeType>();
    auto eltATy = aType.getElementType();
    auto eltBTy = bType.getElementType();
    auto eltCTy = cType.getElementType();

    auto newAType =
        TensorType::get(MOSH::ShapeAttr::get(aShapeDims, shapeTy), eltATy);
    auto newBType =
        TensorType::get(MOSH::ShapeAttr::get(bShapeDims, shapeTy), eltBTy);
    auto newResType =
        TensorType::get(MOSH::ShapeAttr::get(resShapeDims, shapeTy), eltCTy);

    auto newATensor = rewriter.create<Static::ReshapeOp>(op.getLoc(), newAType,
                                                         op.getTensorA());
    auto newBTensor = rewriter.create<Static::ReshapeOp>(op.getLoc(), newBType,
                                                         op.getTensorB());

    auto matmul = rewriter.create<MatmulOp>(op.getLoc(), newResType, newATensor,
                                            newBTensor);
    rewriter.replaceOpWithNewOp<Static::BroadcastToOp>(op, resTy, matmul);
    return success();
  }
};

/// A pattern that detects `mo.conv` with 1x1 filter, 1s for strides and
/// dilations, 0s for paddings, and num_groups=1 and rewrites it into a matmul
/// with some static reshapes.
///
/// Example:
///
/// input  : tensor<[N, H, W, Ci]>
/// filter : tensor<[1, 1, Ci, Co]>
///
///   res = conv(input, filter)[
///     strides = {1, ...}, dilations = {1, ...}, paddings = {0, ...},
///     num_groups = 1] : Sh
/// ===>
///   input'  = reshape(input)  : tensor<[N, H, W, Ci]> -> tensor<[N*H*W, Ci]>
///   filter' = reshape(filter) : tensor<[1, 1, Ci, Co]> -> tensor<[Ci, Co]>
///   matmul  = matmul(input', filter')
///   res     = reshape(matmul) : ... -> Sh
struct ConvToMatmulPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, ConvOp> {
  ConvToMatmulPattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(ConvOp op,
                             DeviceAwareRewriter &rewriter) const override {

    // Declare some meta variables.
    TensorValue input = op.getInput();
    TensorValue filter = op.getFilter();
    TensorType opTy = op.getType();
    Location loc = op.getLoc();

    // Check rank
    TensorType inputTy = input.getType();
    TensorType filterTy = filter.getType();
    if (!inputTy.hasRank() || inputTy.getRank() != 4)
      return rewriter.notifyMatchFailure(op, "input rank must be 4");

    // Check filter size
    ArrayRef<TypedAttr> inputDims = inputTy.getTypedShapeAttr().getValues();
    ArrayRef<TypedAttr> filterDims = filterTy.getTypedShapeAttr().getValues();
    IntegerAttr intOneAttr = rewriter.getIndexAttr(1);
    if (filterDims[0] != intOneAttr || filterDims[1] != intOneAttr)
      return rewriter.notifyMatchFailure(op, "filter size must be 1x1");

    // Check strides and paddings, note that we don't need to check dilations
    // because it's irrelevant for 1x1 filter.
    auto areConstant = [](TensorValue val, int64_t check) {
      std::optional<SmallVector<int64_t>> maybeInts =
          tryGetConstantIntVector(val);
      return maybeInts.has_value() &&
             llvm::all_of(maybeInts.value(),
                          [&check](int64_t n) { return n == check; });
    };
    auto isConstant = [](TensorValue val, int64_t check) {
      std::optional<int64_t> maybeInt = tryGetConstantInt(val);
      return maybeInt && (maybeInt == check);
    };
    if (!areConstant(op.getStrides(), 1))
      return rewriter.notifyMatchFailure(op, "strides must be all 1s");
    if (!areConstant(op.getPaddings(), 0))
      return rewriter.notifyMatchFailure(op, "paddings must be all 0s");
    if (!isConstant(op.getNumGroups(), 1))
      return rewriter.notifyMatchFailure(op, "num_groups must be 1");

    // Insert reshapes for conv input and filter
    auto shapeTy = rewriter.getType<MOSH::ShapeType>();
    // TODO provide a better API for symbolic dimension manipulations
    auto matmulDimM =
        KGEN::ParamOperatorAttr::get(KGEN::POC::MulNuw, inputDims.slice(0, 3));
    TypedAttr matmulDimK = inputDims.back();
    TypedAttr matmulDimN = filterDims.back();
    auto matmulInputShape =
        MOSH::ShapeAttr::get({matmulDimM, matmulDimK}, shapeTy);
    auto matmulWeightShape =
        MOSH::ShapeAttr::get({matmulDimK, matmulDimN}, shapeTy);
    auto matmulShape = MOSH::ShapeAttr::get({matmulDimM, matmulDimN}, shapeTy);
    auto matmulInputTy = inputTy.cloneWith(matmulInputShape);
    auto matmulWeightTy = filterTy.cloneWith(matmulWeightShape);
    auto matmulTy = opTy.cloneWith(matmulShape);
    auto matmulInput =
        rewriter.create<MO::Static::ReshapeOp>(loc, matmulInputTy, input);
    auto matmulWeight =
        rewriter.create<MO::Static::ReshapeOp>(loc, matmulWeightTy, filter);

    // Replace convolution with a newly created matmul
    auto matmul =
        rewriter.create<MO::MatmulOp>(loc, matmulTy, matmulInput, matmulWeight);
    rewriter.replaceOpWithNewOp<MO::Static::ReshapeOp>(op, opTy, matmul);
    return mlir::success();
  }
};

/// A pattern that detects nested `mo.concat` with the same axis, and flatten
/// them into a single `mo.concat` with all the concatenated input values, if
/// the nested concat isn't used by anything else.
///
/// Example:
///
/// t0 : tensor<[1, ...]>
/// t1 : tensor<[2, ...]>
/// t2 : tensor<[3, ...]>
///
///   res0 = concat[axis_0](t0, t1)
///   res1 = concat[axis_0](res0, t2)
/// ===>
///   res1 = concat[axis_0](t0, t1, t2)
struct FlattenNestedConcatPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, ConcatOp> {
  FlattenNestedConcatPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(ConcatOp op,
                             DeviceAwareRewriter &rewriter) const override {

    // Requires constant axis
    std::optional<int64_t> maybeAxis = tryGetConstantInt(op.getAxis());
    if (!maybeAxis.has_value())
      return rewriter.notifyMatchFailure(op, "axis is not constant");
    int64_t axis = maybeAxis.value();

    // Flatten the single-user nested concats by collecting their inputs, along
    // with other non-concat inputs.
    bool foundSingleUserNestedConcat = false;
    SmallVector<Value> flattenedInputs;
    for (Value input : op.getInputs()) {
      auto nestedConcatOp = input.getDefiningOp<ConcatOp>();
      if (nestedConcatOp && nestedConcatOp->hasOneUse() &&
          tryGetConstantInt(nestedConcatOp.getAxis()) == axis) {
        rewriter.pushOp(nestedConcatOp);
        llvm::append_range(flattenedInputs, nestedConcatOp.getInputs());
        foundSingleUserNestedConcat = true;
      } else {
        flattenedInputs.push_back(input);
      }
    }

    // Nothing to merge if we didn't encounter any single-user nested concat
    if (!foundSingleUserNestedConcat) {
      return rewriter.notifyMatchFailure(
          op, "failed to find single-user nested concat");
    }

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    // Replace the old concat with a new concat that operate on the flattened
    // input values. Note that the types are guaranteed to match (shape
    // inference either verified it statically, or dropped runtime asserts).
    rewriter.replaceOpWithNewOp<ConcatOp>(op, op.getType(), op.getAxis(),
                                          flattenedInputs);
    return mlir::success();
  }
};

/// A pattern that detects a power op whose exponent is a constant floating
/// point 3, and rewrites it into two multiplications.
///
/// Example:
///
/// input : tensor<[...]>
///
///   res = pow(input, 3)
/// ===>
///   res = input * input * input
struct Pow3Pattern : public DeviceAwareRewritePattern<OpRewritePattern, PowOp> {
  Pow3Pattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::PowOp op,
                             DeviceAwareRewriter &rewriter) const override {
    Value input;
    if (!m_Op<MO::PowOp>(
             m_Any(&input),
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 2))
             .match(op)) {
      return failure();
    }

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    Value sq = rewriter.create<MulOp>(op->getLoc(), input, input).getResult();
    rewriter.replaceOpWithNewOp<MulOp>(op, sq, input);

    return mlir::success();
  }
};

/// A pattern that detects explicit zero padding before convolution
/// The pattern will only fire if the pad op has constant padding dimensions
/// and a constant zero padding value
///
/// The pattern also ensure that Batch/Channels are not explicitly padded
/// before fusing the pad with the conv
///
/// TODO : Longer term, should probably extend this to pooling
///
/// Example:
///
/// input : tensor<[...]>
///   tmp = pad(input, pads, 0.0)
///   res = conv(tmp, weight, strides, dilations, paddings)
/// ===>
///   res = conv(tmp, weight, strides, dilations, pads[2:-2] + paddings)
struct PadConvPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, ConvOp> {
  PadConvPattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::ConvOp op,
                             DeviceAwareRewriter &rewriter) const override {
    Value input, filter, strides, dilations, pads, pads2, groups;
    if (!m_Op<ConvOp>(m_Op<PadConstantOp>(rewriter.pushOp(), m_Any(&input),
                                          m_Any(&pads), m_ScalarConstant(0.0)),
                      m_Any(&filter), m_Any(&strides), m_Any(&dilations),
                      m_Any(&pads2), m_Any(&groups))
             .match(op)) {
      return rewriter.notifyMatchFailure(op, "failed to match pattern");
    }

    if (!input || !filter || !strides || !dilations || !pads || !pads2 ||
        !groups) {
      return rewriter.notifyMatchFailure(op, "null input value");
    }

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    std::optional<SmallVector<int64_t>> maybePads =
        tryGetConstantIntVector(cast<TensorValue>(pads));

    if (!maybePads.has_value())
      return rewriter.notifyMatchFailure(op, "can't rewrite dynamic padding");

    auto constPads = maybePads.value();
    if (constPads[0] != 0 || constPads[1] != 0 || constPads.end()[-2] != 0 ||
        constPads.end()[-1] != 0) {
      return rewriter.notifyMatchFailure(
          op, "is padding more than the image dimensions");
    }

    auto loc = op.getLoc();
    Value new_pads = rewriter.create<ConstantOp>(
        loc, SmallVector<int64_t>{constPads.begin() + 2, constPads.end() - 2});
    // Cast, sometimes this comes in as si32
    pads2 = rewriter.create<CastOp>(
        loc, cast<TensorType>(pads2.getType()).cloneWith(M::DType::si64),
        pads2);

    new_pads = rewriter.create<AddOp>(loc, new_pads, pads2);

    op.getInputMutable().assign(input);
    op.getPaddingsMutable().assign(new_pads);
    return mlir::success();
  }
};

/// A pattern that detects a series of gathers that are equivalent to a single
/// split, and rewrite them with a new split.
///
/// Example:
///
/// input : tensor<[N, ...]>
///
///   res0 = gather(input, /*indices=*/0, /*axis=*/0)
///   res1 = gather(input, /*indices=*/1, /*axis=*/0)
///   ...
///   resN = gather(input, /*indices=*/N, /*axis=*/0)
/// ===>
///   res  = split(input, /*splitSizes=*/[1, 1, ..., 1], /*axis=*/0)
///   res0 = res#0
///   res1 = res#1
///   ...
///   resN = res#N
struct GathersToSplitPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, GatherOp> {
  GathersToSplitPattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(GatherOp op,
                             DeviceAwareRewriter &rewriter) const override {

    TensorValue input = op.getInput();
    TensorType inputTy = input.getType();
    if (!inputTy.hasRank())
      return rewriter.notifyMatchFailure(op, "unranked input");

    // Get input's 0-axis dimension
    int64_t inputAxis0Dim = inputTy.getIntDim(0);
    if (inputAxis0Dim <= 0) {
      return rewriter.notifyMatchFailure(
          op, "input axis 0 isn't static and positive");
    }

    // Collect N gather users of input (with axis = 0, indices = [0] or ...)
    SmallDenseMap<size_t, GatherOp> idxToGather;
    MO::GatherOp dominantGatherOp; // one that domindates all other gather ops
    for (Operation *user : input.getUsers()) {
      auto userGatherOp = dyn_cast<GatherOp>(user);
      if (!userGatherOp)
        continue;

      // Axis must be 0
      std::optional<int64_t> maybeAxis =
          tryGetConstantInt(userGatherOp.getAxis());
      if (!maybeAxis.has_value() || maybeAxis.value() != 0)
        continue;

      // Indices must be a singleton within [0, inputAxis0Dim)
      std::optional<SmallVector<int64_t>> maybeIndices =
          tryGetConstantIntVector(userGatherOp.getIndices());
      if (!maybeIndices.has_value() || maybeIndices.value().size() != 1)
        continue;

      int64_t idx = maybeIndices.value().front();
      if (idx < 0 || idx >= inputAxis0Dim || idxToGather.contains(idx))
        continue;

      // Update dominant gather and make sure all gathers are in the same block
      if (!dominantGatherOp)
        dominantGatherOp = userGatherOp;

      // TODO account for control-flow
      if (userGatherOp->getBlock() != dominantGatherOp->getBlock())
        continue;

      if (userGatherOp->isBeforeInBlock(dominantGatherOp))
        dominantGatherOp = userGatherOp;

      rewriter.pushOp(userGatherOp);
      idxToGather.try_emplace(idx, userGatherOp);
    }

    // We must have input gather users that cover the entire 0-axis
    if (idxToGather.size() != inputAxis0Dim) {
      return rewriter.notifyMatchFailure(
          op, "not enough qualified gather users for input");
    }

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    // Construct split
    SmallVector<Type> resultTypes;
    for (int64_t idx = 0; idx < inputAxis0Dim; idx++)
      resultTypes.push_back(idxToGather[idx].getType());

    // Insert gather and new constants before dominant gather op; the former is
    // to maintain dominance over input users, the latter is for debuggability.
    Location loc = dominantGatherOp.getLoc();
    rewriter.setInsertionPoint(dominantGatherOp);
    Value splitSizes = rewriter.create<ConstantOp>(
        loc, SmallVector<int64_t>(inputAxis0Dim, 1));
    Value axis = rewriter.create<ConstantOp>(loc, 0);
    auto splitOp =
        rewriter.create<SplitOp>(loc, resultTypes, input, splitSizes, axis);

    // Replace each gather user of input with split result
    for (auto [idx, gatherOp] : idxToGather)
      rewriter.replaceOp(gatherOp, splitOp.getResult(idx));

    return success();
  }
};

/// A pattern that detects a series of algebraic operations that are equivalent
/// to a gelu op, and rewrites them into a new gelu op.
///
/// TODO remove once jit fusion can fuse this into a FC.
///
/// Example:
///
/// input : tensor<[...]>
///
///   res = input * 1/2 * (1 + erf(input/sqrt(2)))
/// ===>
///   res = gelu(input)
struct GeluPattern : public DeviceAwareRewritePattern<OpRewritePattern, MulOp> {
  GeluPattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 4) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::MulOp op,
                             DeviceAwareRewriter &rewriter) const override {
    // Abbreviate the helper to try to keep things readable.
    auto po = [&rewriter]() { return rewriter.pushOp(); };

    Value input, input_erf;
    // gelu = x * 1/2 * (1 + erf(x/sqrt(2)))
    auto sq2 = m_Op<Static::BroadcastToOp>(
        po(), m_ScalarConstant(llvm::numbers::sqrt2f));
    auto divsq2 = m_Op<Static::BroadcastToOp>(
        po(), m_ScalarConstant(llvm::numbers::inv_sqrt2f));

    auto x_erf = m_Any(&input_erf);
    auto erf = m_Op<ErfOp>(
        po(), m_Optional(m_Op<DivOp>(po(), x_erf, sq2),
                         m_BinaryCommutativeOp<MulOp>(po(), x_erf, divsq2)));
    auto one = m_Op<Static::BroadcastToOp>(po(), m_ScalarConstant(1.0f));
    auto add = m_BinaryCommutativeOp<AddOp>(po(), one, erf);

    auto half = m_Op<Static::BroadcastToOp>(po(), m_ScalarConstant(0.5f));
    auto x = m_Any(&input);
    // TODO: write an associative commutative matcher?
    auto gelu =
        m_Optional(m_BinaryCommutativeOp<MulOp>(
                       po(), m_BinaryCommutativeOp<MulOp>(po(), x, half), add),
                   m_BinaryCommutativeOp<MulOp>(
                       po(), half, m_BinaryCommutativeOp<MulOp>(po(), x, add)));

    if (!gelu.match(op->getResult(0)))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !input_erf)
      return rewriter.notifyMatchFailure(op, "null input value");
    if (!(input == input_erf))
      return rewriter.notifyMatchFailure(op, "inputs don't match");

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    rewriter.replaceOpWithNewOp<GeluOp>(op, input);

    return mlir::success();
  }
};

/// A pattern that detects broadcast ops preceded by a transpose, and rewrites
/// them by moving the transpose after the broadcast.
///
/// Example:
///
///  input     perm       input
///    \        /           |
///    transpose     ==>  unary(*)    perm
///        |                 \       /
///      unary               transpose(*)
///
/// NOTE: (*) means result type or operands of the op were updated
struct SinkTransposeThroughUnaryOpPattern
    : public DeviceAwareRewritePattern<OpInterfaceRewritePattern,
                                       ElementWiseUnary> {
  SinkTransposeThroughUnaryOpPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(ElementWiseUnary op,
                             DeviceAwareRewriter &rewriter) const override {
    Value perm;
    Operation *opaqueTransposeOp = nullptr;
    auto unaryP = m_Op<ElementWiseUnary>(
        m_Op<TransposeOp>(&opaqueTransposeOp, m_Any(), m_Any(&perm)));
    if (!unaryP.match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!perm)
      return rewriter.notifyMatchFailure(op, "null permutation value");

    // If transpose op has multiple uses, we don't sink the transposes because
    // we'll have to recompute the original transpose op for other users.
    Value transpose = opaqueTransposeOp->getResult(0);
    if (!transpose.hasOneUse())
      return rewriter.notifyMatchFailure(op, "transpose op has multiple uses");

    if (!rewriter.addOps({opaqueTransposeOp})) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    TensorType finalOutputType = op.getResult().getType();
    auto transposeOp = cast<TransposeOp>(opaqueTransposeOp);
    TensorValue input = transposeOp.getInput();
    TensorValue opRes = op.getResult();
    rewriter.replaceAllUsesWith(op->getResults(), transposeOp.getResult());
    rewriter.modifyOpInPlace(op, [&]() {
      op.setInput(input);
      DType unaryResultDType = opRes.getType().getDtype();
      opRes.setType(input.getType().cloneWith(unaryResultDType));
    });

    rewriter.modifyOpInPlace(transposeOp, [&]() {
      transposeOp.getInputMutable().assign(opRes);
      transposeOp->moveAfter(op);
      transposeOp.getResult().setType(finalOutputType);
    });
    return mlir::success();
  }
};

/// A pattern that detects binary ops preceded by a transpose on each input, and
/// rewrites them by replacing the old transposes with a new transpose after the
/// binary op.
///
/// Example:
///
/// input1     perm   input2     perm
///    \        /        \        /
///    transpose1        transpose2
///           \           /
///               binary
///
/// =======>
///
///  input1   input2
///     \      /
///      binary(*)    perm1 (or perm2, they are the same)
///          \        /
///          transpose
///
/// NOTE (*) means result type or operands of the op were updated
struct SinkTransposeThroughBinaryOpPattern
    : public DeviceAwareRewritePattern<OpInterfaceRewritePattern,
                                       ElementWiseBinary> {
  SinkTransposeThroughBinaryOpPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(ElementWiseBinary op,
                             DeviceAwareRewriter &rewriter) const override {
    TensorValue input1, input2, perm1, perm2;
    auto transpose1P =
        m_Op<TransposeOp>(rewriter.pushOp(), m_Any(&input1), m_Any(&perm1));
    auto transpose2P =
        m_Op<TransposeOp>(rewriter.pushOp(), m_Any(&input2), m_Any(&perm2));
    auto binaryP = m_Op<ElementWiseBinary>(transpose1P, transpose2P);
    if (!binaryP.match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input1 || !input2 || !perm1 || !perm2)
      return rewriter.notifyMatchFailure(op, "null input or permutation value");

    // If any transpose op has multiple uses, we don't sink the transposes
    // because we'll end up doing extra computation overall since we must keep
    // the original transposes for other users.
    if (!op.getLhsInput().hasOneUse())
      return rewriter.notifyMatchFailure(op, "lhs transpose has multiple uses");

    if (!op.getRhsInput().hasOneUse())
      return rewriter.notifyMatchFailure(op, "rhs transpose has multiple uses");

    if (perm1 != perm2)
      return rewriter.notifyMatchFailure(op, "permutations are different");

    TensorType input1Ty = input1.getType();
    TensorType input2Ty = input2.getType();
    if (input1Ty.getShapeAttr() != input2Ty.getShapeAttr())
      return rewriter.notifyMatchFailure(op, "input shapes are different");

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    TensorValue opRes = op.getResult();
    TensorType oldOpResTy = opRes.getType();
    rewriter.modifyOpInPlace(op, [&]() {
      op.setLhsInput(input1);
      op.setRhsInput(input2);

      opRes.setType(input1Ty.cloneWith(oldOpResTy.getDtype()));
    });

    rewriter.setInsertionPointAfter(op);
    auto finalTransposeOp =
        rewriter.create<TransposeOp>(op->getLoc(), oldOpResTy, opRes, perm1);
    rewriter.replaceAllUsesExcept(opRes, finalTransposeOp, finalTransposeOp);

    return mlir::success();
  }
};

/// A pattern that detects broadcast ops preceded by a transpose, and rewrites
/// them by moving the transpose after the broadcast.
///
/// Example:
///
///  input     perm       input
///    \        /           |
///    transpose     ==>  broadcast(*)    perm
///        |                 \           /
///    broadcast              transpose(*)
///
/// NOTE (*) means result type or operands of the op were updated
struct SinkTransposeThroughStaticBroadcastToOpPattern
    : public DeviceAwareRewritePattern<OpRewritePattern,
                                       Static::BroadcastToOp> {
  SinkTransposeThroughStaticBroadcastToOpPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  // Return a permutation of `attrs` such that, if it gets permuted via `perm`,
  // it'll become `attrs` again.
  static SmallVector<TypedAttr> unpermute(ArrayRef<TypedAttr> attrs,
                                          ArrayRef<int64_t> perm) {
    // e.g., if perm == [2, 0, 1], inversePerm == [1, 2  0]
    SmallVector<TypedAttr> result(perm.size());
    for (auto [idx, permIdx] : llvm::enumerate(perm))
      result[permIdx] = attrs[idx];
    return result;
  }

  LogicalResult
  deviceAwareMatchAndRewrite(Static::BroadcastToOp op,
                             DeviceAwareRewriter &rewriter) const override {
    Value input, perm;
    Operation *opaqueTransposeOp = nullptr;
    auto broadcastP = m_Op<Static::BroadcastToOp>(
        m_Op<TransposeOp>(&opaqueTransposeOp, m_Any(&input), m_Any(&perm)));
    if (!broadcastP.match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !perm)
      return rewriter.notifyMatchFailure(op, "null input or permutation value");

    // If transpose op has multiple uses, we don't sink the transposes because
    // it we'll have to recompute the original transpose op for other users.
    auto transposeOp = cast<TransposeOp>(opaqueTransposeOp);
    if (!transposeOp->hasOneUse())
      return rewriter.notifyMatchFailure(op, "transpose op has multiple uses");

    MO::TensorType oldBroadcastResTy = op.getType();
    MO::TensorType transposeResTy = transposeOp.getType();
    if (!oldBroadcastResTy.hasRank())
      return rewriter.notifyMatchFailure(op, "broadcast shape is unranked");
    if (!transposeResTy.hasRank())
      return rewriter.notifyMatchFailure(op, "transpose shape is unranked");

    auto maybePermVec = tryGetConstantIntVector(cast<TensorValue>(perm));
    if (!maybePermVec.has_value())
      return rewriter.notifyMatchFailure(op, "permutation isn't constant");

    if (!rewriter.addOps({opaqueTransposeOp})) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    SmallVector<int64_t> permVec = maybePermVec.value();
    // Broadcast is rank-increasing, so we must pad the permutation vector to
    // account for the added dimensions. For instance, if we have
    //   input shape      = [1, 3, 2]
    //   permutation      = [0, 2, 1]
    //   transposed shape = [1, 2, 3]
    //   bcast shape      = [9, 1, 2, 3]
    // then after sinking the transpose, we must have
    //   input shape      = [1, 3, 2]
    //   bcast shape      = [9, 1, 3, 2]
    //   permutation      = [0, 1, 3, 2]
    //   transposed shape = [9, 1, 2, 3]
    if (oldBroadcastResTy.getRank() > transposeResTy.getRank()) {
      size_t extraDims = oldBroadcastResTy.getRank() - transposeResTy.getRank();
      for (int64_t &axis : permVec)
        axis += extraDims;
      auto paddedAxes =
          llvm::iota_range<int64_t>(0, extraDims, /*Inclusive=*/false);
      permVec.insert(permVec.begin(), paddedAxes.begin(), paddedAxes.end());
      auto newPerm = rewriter.create<ConstantOp>(op.getLoc(), permVec);

      rewriter.modifyOpInPlace(
          transposeOp, [&]() { transposeOp.getPermMutable().assign(newPerm); });
    }
    SmallVector<TypedAttr> newBroadcastDims =
        unpermute(oldBroadcastResTy.getDims(), permVec);

    auto shapeTy = rewriter.getType<ShapeType>();
    MO::TensorType newBroadcastResTy =
        oldBroadcastResTy.cloneWith(ShapeAttr::get(newBroadcastDims, shapeTy));

    rewriter.replaceAllUsesWith(op, transposeOp.getResult());
    rewriter.modifyOpInPlace(op, [&]() {
      op.getInputMutable().assign(input);
      op.getResult().setType(newBroadcastResTy);
    });

    rewriter.modifyOpInPlace(transposeOp, [&]() {
      transposeOp.getInputMutable().assign(op.getResult());
      transposeOp->moveAfter(op);
      transposeOp.getResult().setType(oldBroadcastResTy);
    });

    return mlir::success();
  }
};

/// Check whether `val` comes from a reshape that eliminates the trailing 1s
/// from input's shape.
static bool isValueFromReshapeThatSqueezesFinalDims(Value val) {
  TensorType inputTy, outputTy;
  if (auto reshapeOp = val.getDefiningOp<MO::Static::ReshapeOp>()) {
    inputTy = reshapeOp.getInput().getType();
    outputTy = reshapeOp.getType();
  } else if (auto reshapeOp = val.getDefiningOp<MO::Static::ReshapeOp>()) {
    inputTy = reshapeOp.getInput().getType();
    outputTy = reshapeOp.getType();
  } else {
    return false;
  }

  auto inputShape = dyn_cast<MO::ShapeAttr>(inputTy.getShapeAttr());
  auto outputShape = dyn_cast<MO::ShapeAttr>(outputTy.getShapeAttr());

  auto inputShapeHasTrailingOne = inputShape && inputShape.getRank() > 0 &&
                                  inputShape.getIntDims().back() == 1;

  auto outputShapeHasNoTrailingOne =
      outputShape &&
      (outputShape.getRank() == 0 || outputShape.getIntDims().back() != 1);

  return inputShapeHasTrailingOne && outputShapeHasNoTrailingOne;
}

static bool isStaticConcat(ConcatOp op) {
  auto inputType = op.getType();
  if (!inputType.hasRank())
    return false;

  auto axisOr = tryGetConstantInt(op.getAxis());
  if (!axisOr)
    return false;

  uint64_t axis = normalizeAxis(*axisOr, inputType.getRank());
  if (axis >= inputType.getRank())
    return false;

  if (!isa<IntegerAttr>(inputType.getDim(axis)))
    return false;

  return true;
}

template <typename... OpTypes>
static bool allOperandsOfThisTypeHaveOneUse(Operation::operand_range operands) {
  return llvm::all_of(operands, [&](Value operand) {
    auto op = operand.getDefiningOp();
    if (!op || !isa<OpTypes...>(op))
      return false;

    return llvm::range_size(operand.getUsers()) == 1;
  });
}

// Given the input and output tensor types of a reshape operation try to
// determine whether the reshape corresponds to a squeeze
// (cases where the input and output dims match exactly other than
// one extra dim in the input that has value 1)
//
// Will return an optional wrapping the axis of the squeeze in such cases and a
// std::nullopt otherwise
//
// e.g.
// inputType = tensor<[D1, 2, 1, D3]>
// outputType = tensor<[D1, 2, D3]>
// res = optional{2}
//
// inputType = tensor<[D1, 2, 1, D4]>
// outputType = tensor<[D1, 2, D3]>
// res = nullopt
static std::optional<int64_t> findSqueezeAxis(MO::TensorType inputType,
                                              MO::TensorType outputType) {

  auto inputDims = inputType.getDims();
  auto outputDims = outputType.getDims();

  // check that we have static rank first
  if (inputDims.size() != outputDims.size() + 1)
    return std::nullopt;

  auto zipIter = llvm::zip_longest(inputDims, outputDims);

  auto it = llvm::find_if(zipIter, [](const auto &ioDims) {
    return std::get<0>(ioDims) != std::get<1>(ioDims);
  });

  auto [inputDim, outputDim] = *it;
  auto inputDimInt = dyn_cast<mlir::IntegerAttr>(*inputDim);

  if (!inputDimInt)
    return std::nullopt;

  if (inputDimInt.getInt() != 1)
    return std::nullopt;

  auto offset = std::distance(zipIter.begin(), it);

  if (!std::equal(inputDims.begin(), inputDims.begin() + offset,
                  outputDims.begin()))
    return std::nullopt;

  if (!std::equal(inputDims.begin() + offset + 1, inputDims.end(),
                  outputDims.begin() + offset))
    return std::nullopt;

  return offset;
}

/// This pattern will swim a concat operation through a ReshapeOpType (usually a
/// static reshape or a rebind op).
/// CONDITIONS:
/// - Concat axis is statically known.
/// - Concat dimension is statically known.
/// - All concat providers are only used by the concat.
/// - Concat must be uniform (same size).
/// - All concat providers must be `ReshapeOpType`.
/// - The reshape must not mix dynamic and static dimensions.
/// - The concat axis must not be mixed with left dimensions after the reshape.
/// - [Unchecked] Row major-like memory layout.
/// BEFORE:
/// -> reshape -> concat ->
/// -> reshape ----| |
/// -> reshape ------|
/// AFTER:
/// -> concat -> reshape
template <typename ReshapeOpType>
struct SwimConcatThroughReshapeLikeOpsPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, ConcatOp> {
  SwimConcatThroughReshapeLikeOpsPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(ConcatOp concatOp,
                             DeviceAwareRewriter &rewriter) const override {
    if (!isStaticConcat(concatOp) ||
        !allOperandsOfThisTypeHaveOneUse<ReshapeOpType>(concatOp.getInputs())) {
      return rewriter.notifyMatchFailure(
          concatOp, "Unsupported concat. Must be static and providing reshapes "
                    "must have no other user");
    }

    rewriter.pushOp(concatOp);
    SmallVector<ReshapeOpType> reshapeOps;
    SmallVector<Value> reshapeOperands;
    for (Value operand : concatOp.getInputs()) {
      auto op = operand.getDefiningOp();
      auto reshapeLikeOp = dyn_cast<ReshapeOpType>(op);
      if (!reshapeLikeOp.getType().hasRank()) {
        return rewriter.notifyMatchFailure(concatOp,
                                           "Reshape inputs must be ranked");
      }
      reshapeOps.push_back(reshapeLikeOp);
      reshapeOperands.push_back(reshapeLikeOp.getInput());
      rewriter.pushOp(op);
    }

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          concatOp, "matched ops have inconsistent devices");
    }

    auto firstBeforeType =
        cast<TensorType>(reshapeOps.front().getInput().getType());
    for (auto reshapeLikeOp : ArrayRef(reshapeOps).slice(1)) {
      auto beforeType = cast<TensorType>(reshapeLikeOp.getInput().getType());
      if (beforeType != firstBeforeType) {
        return rewriter.notifyMatchFailure(concatOp,
                                           "Only supports uniform concat");
      }
    }

    auto firstAfterType = cast<TensorType>(reshapeOps.front().getType());
    const StaticReshapeDimensionsMappingAnalysis
        staticReshapeDimensionsMappingAnalysis(firstAfterType.getDims(),
                                               firstBeforeType.getDims());
    if (!staticReshapeDimensionsMappingAnalysis) {
      return rewriter.notifyMatchFailure(concatOp,
                                         "Unsupported reshape operand");
    }

    auto concatInputType = firstAfterType;
    const int64_t concatAxis = normalizeAxis(
        *tryGetConstantInt(concatOp.getAxis()), concatInputType.getRank());

    uint64_t mappedConcatAxis = concatAxis;
    // Will stop at the first non-1 dimension, dynamic dimension included. This
    // is because 1s are ignored by the analysis and they serve no real purpose
    // in the reshape.
    while (mappedConcatAxis < concatInputType.getRank() &&
           concatInputType.getIntDim(mappedConcatAxis) == 1) {
      ++mappedConcatAxis;
    }

    // If there is no non-1 dimension and or if that dimension is dynamic, then
    // the reshape is unsupported. This is because there is no way to concat it
    // with the new dimensions.
    const bool couldNotFindAnotherAxis =
        mappedConcatAxis == concatInputType.getRank();
    if (couldNotFindAnotherAxis ||
        !isa<IntegerAttr>(concatInputType.getDim(mappedConcatAxis))) {
      return rewriter.notifyMatchFailure(
          concatOp,
          "Trailing ones or trailing dynamic dimensions are unsupported");
    }

    // If the dimension was expanded and it was not mixed with other dimensions
    // or if it was contracted but the mixing was done with dimensions that are
    // on the right, then it can proceed. This assumes a row major memory
    // layout.
    // What mixing means is that two dimensions from the original shape
    // contributed to the same dimension in the reshaped shape. This prevents a
    // concat unless the concat axis is the leftmost of the mixed dimensions (in
    // row major).
    const bool couldMix =
        staticReshapeDimensionsMappingAnalysis
            .backwardMapping()[staticReshapeDimensionsMappingAnalysis
                                   .forwardMapping()[mappedConcatAxis]
                                   .front()]
            .size() > 1;
    const bool mixedAndConcatAxisIsNotFirst =
        staticReshapeDimensionsMappingAnalysis
            .backwardMapping()[staticReshapeDimensionsMappingAnalysis
                                   .forwardMapping()[mappedConcatAxis]
                                   .front()]
            .front() != static_cast<int64_t>(mappedConcatAxis);
    if (couldMix && mixedAndConcatAxisIsNotFirst) {
      return rewriter.notifyMatchFailure(concatOp,
                                         "Reshape is mixing static dimensions");
    }

    // The new concat axis is the left most axis in the shape after the reshape.
    const int64_t newConcatAxis = staticReshapeDimensionsMappingAnalysis
                                      .forwardMapping()[mappedConcatAxis]
                                      .front();

    Location loc = concatOp->getLoc();
    rewriter.setInsertionPoint(concatOp);

    SmallVector<TypedAttr> newConcatOutputShape(firstBeforeType.getDims());
    newConcatOutputShape[newConcatAxis] = rewriter.getIndexAttr(
        cast<IntegerAttr>(newConcatOutputShape[newConcatAxis]).getInt() *
        reshapeOps.size());
    auto newConcatOutputType =
        TensorType::get(ShapeAttr::get(newConcatOutputShape,
                                       rewriter.getType<MOSH::ShapeType>()),
                        concatInputType.getDtype());
    SmallVector<Value> newConcatOperands = {
        rewriter.create<ConstantOp>(loc, newConcatAxis)};
    newConcatOperands.append(reshapeOperands.begin(), reshapeOperands.end());
    OperationState concatOpState(loc, concatOp->getName(), newConcatOperands,
                                 newConcatOutputType, concatOp->getAttrs());
    auto newConcatOp = rewriter.create(concatOpState);

    auto newReshapeOp = rewriter.create<ReshapeOpType>(
        loc, concatOp.getType(), newConcatOp->getResult(0));

    rewriter.replaceAllUsesWith(concatOp, newReshapeOp);
    rewriter.eraseOp(concatOp);
    for (auto reshapeOp : reshapeOps)
      rewriter.eraseOp(reshapeOp);

    return success();
  }
};

/// A pattern that detects binary operator where one operand is a reshape
/// and the other a broadcast. We attempt to move the binary operator to
/// occur before the reshape for better fusion opportunities.
///
/// Example:
///
/// ┌─────┐         ┌────────┐               ┌─────┐  ┌────────┐
/// │input│         │elemwise│               │input│  │elemwise│
/// └──┬──┘         └───┬────┘               └──┬──┘  └───┬────┘
///    │                │                       │         │
///    │                │                       │         │
/// reshape          broadcast                  │     broadcast*
///    │                │                       │         │
///    │                │         ───────►      └─binOp*──┘
///    └──── binOp ─────┘                         (add)
///          (add)                                  │
///            │                                    │
///            │                                  reshape*
///         ┌──┴───┐                                │
///         │output│                             ┌──┴───┐
///         └──────┘                             │output│
///                                              └──────┘
/// Note reshape can potentially do nasty things so this optimization cannot
/// always be done.
///
/// We only do this optimization if out of the values `elemwise`, `input`, and
/// `reshape` we can prove `elemwise` has the least rank `r`. Then we can look
/// at the last `r` dims of `broadcast`, `input`, and `reshape` and see if the
/// dimensions are the same.
template <typename ReshapeOpType>
struct SinkStaticReshapeLikeOpThroughBroadcastedElementwiseBinaryOp
    : public DeviceAwareRewritePattern<OpInterfaceRewritePattern,
                                       ElementWiseBinary> {
  SinkStaticReshapeLikeOpThroughBroadcastedElementwiseBinaryOp(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(ElementWiseBinary op,
                             DeviceAwareRewriter &rewriter) const override {

    // Match pattern and get results.
    TensorValue inputTensor, elemwiseTensor;
    auto reshapeP = m_Op<ReshapeOpType>(rewriter.pushOp(), m_Any(&inputTensor));
    auto broadcastP =
        m_Op<Static::BroadcastToOp>(rewriter.pushOp(), m_Any(&elemwiseTensor));
    auto elemwiseOpP1 =
        m_Op<ElementWiseBinary>(rewriter.pushOp(), reshapeP, broadcastP);
    auto elemwiseOpP2 =
        m_Op<ElementWiseBinary>(rewriter.pushOp(), broadcastP, reshapeP);

    // whether the reshaped argument was the LHS or RHS
    bool reshapedWasLeftOriginally;
    TensorValue reshapedTensor, broadcastTensor;
    if (elemwiseOpP1.match(op)) { // handle different orderings for binary op
      reshapedTensor = op.getLhsInput();
      broadcastTensor = op.getRhsInput();
      reshapedWasLeftOriginally = true;
    } else if (elemwiseOpP2.match(op)) {
      broadcastTensor = op.getLhsInput();
      reshapedTensor = op.getRhsInput();
      reshapedWasLeftOriginally = false;
    } else {
      return rewriter.notifyMatchFailure(op, "Failed to match the pattern.");
    }
    if (!inputTensor || !elemwiseTensor || !reshapedTensor || !broadcastTensor)
      return rewriter.notifyMatchFailure(op, "Null values found in pattern.");

    // TODO(#18823) find a more general solution
    if (isValueFromReshapeThatSqueezesFinalDims(reshapedTensor)) {
      return rewriter.notifyMatchFailure(
          op, "Don't sink reshape that squeezes final dim.");
    }

    TensorType inputTensorTy = inputTensor.getType();
    // This avoids creating a binary op with unknown shapes.
    if (inputTensorTy.hasUnknownShape())
      return rewriter.notifyMatchFailure(op, "Shape must be known");

    // Ensure the smallest rank tensor is `elemwise`
    TensorType reshapedTensorTy = reshapedTensor.getType();
    TensorType elemwiseTensorTy = elemwiseTensor.getType();
    TensorType broadcastTensorTy = broadcastTensor.getType();
    if (!reshapedTensorTy.hasRank() || !elemwiseTensorTy.hasRank() ||
        !inputTensorTy.hasRank() || !broadcastTensorTy.hasRank())
      return rewriter.notifyMatchFailure(
          op, "Cannot prove correctness in case of dynamic rank.");
    size_t smallestRank =
        std::min(reshapedTensorTy.getRank(), elemwiseTensorTy.getRank());
    smallestRank = std::min(smallestRank, inputTensorTy.getRank());
    smallestRank = std::min(smallestRank, broadcastTensorTy.getRank());

    if (smallestRank != elemwiseTensorTy.getRank())
      return rewriter.notifyMatchFailure(
          op,
          "Original shape of broadcasted operand does not have smallest rank.");

    // Check trailing dims of `input`, `reshape` and `broadcast` match
    for (size_t i = 0; i < smallestRank; i++) {
      TypedAttr reshapedTensorDim =
          reshapedTensorTy.getDim(reshapedTensorTy.getRank() - i - 1);
      TypedAttr broadcastTensorDim =
          broadcastTensorTy.getDim(broadcastTensorTy.getRank() - i - 1);
      TypedAttr inputTensorDim =
          inputTensorTy.getDim(inputTensorTy.getRank() - i - 1);

      if (reshapedTensorDim != broadcastTensorDim ||
          broadcastTensorDim != inputTensorDim)
        return rewriter.notifyMatchFailure(
            op, "Trailing dimensions of the reshape, input, and elemwise "
                "tensor do not match.");
    }

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    // TODO (#36567):
    // This following early exit is a bit of a hack used to
    // avoid an infinite loop that can occur somtimes occure between
    // this pattern and the SwimViewOpThroughBinaryOpWithConstantSubtree
    // patterns
    if (op.getResult().hasOneUse()) {
      auto user = op.getResult().getUses().begin().getUser();
      if (auto userReshapeOp = dyn_cast<ReshapeOpType>(user)) {
        return rewriter.notifyMatchFailure(
            op,
            "Consumer operation is a reshape which would undo this pattern");
      }
    }

    // Now we are sure moving the broadcast up will not change results.
    // Intuition:
    //  - we know slices across the trailing dimension will have the same
    //     corresponding elemwise operands.
    //  - we know the trailing dimensions before and after reshape are the same
    DType broadcastDType = elemwiseTensor.getType().getDtype();
    TensorValue newBroadcast = rewriter.create<Static::BroadcastToOp>(
        op->getLoc(), inputTensor.getType().cloneWith(broadcastDType),
        elemwiseTensor);
    TensorType oldType = op.getResult().getType();

    auto newOp = cast<ElementWiseBinary>(rewriter.clone(*op));

    // make sure to preserve ordering for ops like mo.div
    if (reshapedWasLeftOriginally) {
      newOp.setLhsInput(inputTensor);
      newOp.setRhsInput(newBroadcast);
    } else {
      newOp.setLhsInput(newBroadcast);
      newOp.setRhsInput(inputTensor);
    }
    newOp.getResult().setType(inputTensor.getType());

    TensorValue output = rewriter.create<ReshapeOpType>(op->getLoc(), oldType,
                                                        newOp.getResult());

    rewriter.replaceAllUsesWith(op.getResult(), output);
    return mlir::success();
  }
};

/// Drops reshapes-like ops through Elementwise Unary Ops.
///                    ┌─────┐     ┌─────┐
///                    │input│     │input│
///                    └──┬──┘     └──┬──┘
///                       │           │
///                    reshape     unaryOp*
///                       │         (erf)
///                       │    ──►    │
///                    unaryOp        │
///                     (erf)      reshape*
///                       │           │
///                    ┌──┴───┐    ┌──┴───┐
///                    │output│    │output│
///                    └──────┘    └──────┘
template <typename ReshapeOpType>
struct SinkStaticReshapeLikeOpThroughBroadcastedElementwiseUnaryOp
    : public DeviceAwareRewritePattern<OpInterfaceRewritePattern,
                                       ElementWiseUnary> {
  SinkStaticReshapeLikeOpThroughBroadcastedElementwiseUnaryOp(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(ElementWiseUnary op,
                             DeviceAwareRewriter &rewriter) const override {
    TensorValue inputTensor;
    auto reshapeP = m_Op<ReshapeOpType>(rewriter.pushOp(), m_Any(&inputTensor));
    auto elemwiseOpP = m_Op<ElementWiseUnary>(reshapeP);

    if (!elemwiseOpP.match(op))
      return rewriter.notifyMatchFailure(op, "Pattern not matched.");

    if (!inputTensor)
      return rewriter.notifyMatchFailure(op, "Null values found in pattern.");

    // TODO(#18823) find a more general solution
    if (isValueFromReshapeThatSqueezesFinalDims(op.getInput())) {
      return rewriter.notifyMatchFailure(
          op, "Don't sink reshape that squeezes final dim.");
    }

    auto inputType = llvm::cast<TensorType>(inputTensor.getType());
    // This avoids creating a unary op with unknown shapes.
    if (inputType.hasUnknownShape())
      return rewriter.notifyMatchFailure(op, "Shape must be known");

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    auto newOp = cast<ElementWiseUnary>(rewriter.clone(*op));

    // We cannot assume output will be the same dtype as input (e.g. mo.cast)
    DType outputDType = newOp.getResult().getType().getDtype();
    newOp.setInput(inputTensor);
    newOp.getResult().setType(inputTensor.getType().cloneWith(outputDType));
    TensorValue output = rewriter.create<ReshapeOpType>(
        op->getLoc(), op.getResult().getType(), newOp.getResult());

    rewriter.replaceAllUsesWith(op.getResult(), output);
    return mlir::success();
  }
};

/// Drops reshape ops through SoftmaxOp as long as the reshape doesn't affect
/// the last axis (the compute axis of softmax).
///                    ┌─────┐     ┌─────┐
///                    │input│     │input│
///                    └──┬──┘     └──┬──┘
///                       │           │
///                    reshape        │
///                       │           │
///                    softmax ──► softmax
///                       │           │
///                       │        reshape
///                       │           │
///                    ┌──┴───┐    ┌──┴───┐
///                    │output│    │output│
///                    └──────┘    └──────┘
struct SinkStaticReshapeOpThroughSoftmaxOp
    : public DeviceAwareRewritePattern<OpRewritePattern, SoftmaxOp> {
  SinkStaticReshapeOpThroughSoftmaxOp(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(SoftmaxOp op,
                             DeviceAwareRewriter &rewriter) const override {
    TensorValue inputTensor;
    auto reshapeP =
        m_Op<Static::ReshapeOp>(rewriter.pushOp(), m_Any(&inputTensor));
    auto softmaxOpP = m_Op<SoftmaxOp>(reshapeP);

    if (!softmaxOpP.match(op))
      return rewriter.notifyMatchFailure(op, "Pattern not matched.");

    if (!inputTensor)
      return rewriter.notifyMatchFailure(op, "Null values found in pattern.");

    // TODO(#18823) find a more general solution
    if (isValueFromReshapeThatSqueezesFinalDims(op.getInput())) {
      return rewriter.notifyMatchFailure(
          op, "Don't sink reshape that squeezes final dim.");
    }

    TensorType inputTy = inputTensor.getType();
    TensorType reshapedTy = op.getInput().getType();
    // This avoids sinking reshape that modifies the last axis, which is the
    // compute axis of softmax.
    if (!inputTy.hasRank() || !reshapedTy.hasRank())
      return rewriter.notifyMatchFailure(op, "Shape must be ranked");
    if (inputTy.getRank() > 0 &&
        inputTy.getTypedShapeAttr().getValues().back() !=
            reshapedTy.getTypedShapeAttr().getValues().back())
      return rewriter.notifyMatchFailure(op, "Last axis is modified");

    // This avoids creating a softmax op with unknown shapes.
    if (inputTy.hasUnknownShape())
      return rewriter.notifyMatchFailure(op, "Shape must be known");

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    auto newOp = cast<SoftmaxOp>(rewriter.clone(*op));

    // We cannot assume output will be the same dtype as input (e.g. mo.cast)
    DType outputDType = newOp.getResult().getType().getDtype();
    newOp.getInputMutable().set(inputTensor);
    newOp.getResult().setType(inputTensor.getType().cloneWith(outputDType));
    TensorValue output = rewriter.create<Static::ReshapeOp>(
        op->getLoc(), op.getResult().getType(), newOp.getResult());

    rewriter.replaceAllUsesWith(op.getResult(), output);
    return mlir::success();
  }
};

/// Sinks operands which are reshapes through binary operators if
/// it can be proven `input1` and `input2` are the same shape
/// ┌──────┐     ┌──────┐    ┌──────┐   ┌──────┐
/// │input1│     │input2│    │input1│   │input2│
/// └──┬───┘     └──┬───┘    └──┬───┘   └───┬──┘
///    │            │           │           │
/// Reshape      Reshape        └───binOp───┘
///    │            │               (add)
///    │            │    ─────►       │
///    └────binOp───┘                 │
///         (add)                  Reshape
///           │                       │
///       ┌───┴──┐                 ┌──┴───┐
///       │output│                 │output│
///       └──────┘                 └──────┘
template <typename ReshapeOpType>
struct SinkSameStaticReshapeLikeOpsThroughElementwiseBinaryOp
    : public DeviceAwareRewritePattern<OpInterfaceRewritePattern,
                                       ElementWiseBinary> {
  SinkSameStaticReshapeLikeOpsThroughElementwiseBinaryOp(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(ElementWiseBinary op,
                             DeviceAwareRewriter &rewriter) const override {
    TensorValue input1Tensor, input2Tensor;
    auto reshapeP1 =
        m_Op<ReshapeOpType>(rewriter.pushOp(), m_Any(&input1Tensor));
    auto reshapeP2 =
        m_Op<ReshapeOpType>(rewriter.pushOp(), m_Any(&input2Tensor));
    auto elemwiseOpP = m_Op<ElementWiseBinary>(reshapeP1, reshapeP2);

    if (!elemwiseOpP.match(op))
      return rewriter.notifyMatchFailure(op, "Pattern not matched.");

    if (!input1Tensor || !input2Tensor)
      return rewriter.notifyMatchFailure(op, "Null values found in pattern.");

    TensorType input1TensorTy = input1Tensor.getType();
    TensorType input2TensorTy = input2Tensor.getType();

    if (input1TensorTy.getShapeAttr() != input2TensorTy.getShapeAttr())
      return rewriter.notifyMatchFailure(op, "Inputs shapes must be equal");

    // This avoids creating a binary op with unknown shapes.
    if (input1TensorTy.hasUnknownShape())
      return rewriter.notifyMatchFailure(op, "Input shapes must be known");

    // TODO(#18823) find a more general solution
    if (isValueFromReshapeThatSqueezesFinalDims(op.getLhsInput()) ||
        isValueFromReshapeThatSqueezesFinalDims(op.getRhsInput())) {
      return rewriter.notifyMatchFailure(
          op, "Don't sink reshape that squeezes final dim.");
    }

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    TensorType finalOutputType = op.getResult().getType();
    auto newOp = cast<ElementWiseBinary>(rewriter.clone(*op));
    newOp.setLhsInput(input1Tensor);
    newOp.setRhsInput(input2Tensor);
    newOp.getResult().setType(
        input1TensorTy.cloneWith(finalOutputType.getDtype()));
    TensorValue output = rewriter.create<ReshapeOpType>(
        op->getLoc(), finalOutputType, newOp.getResult());

    rewriter.replaceAllUsesWith(op.getResult(), output);
    return mlir::success();
  }
};

/// Construct a new TensorType based on `ty` with the shape dimensions
/// permuted by `perm`.
TensorType permuteDims(TensorType ty, ArrayRef<int64_t> perm) {
  SmallVector<TypedAttr> permutedDims =
      permute(ty.getTypedShapeAttr().getValues(), perm);
  auto shapeTy = ShapeType::get(ty.getContext());
  return ty.cloneWith(ShapeAttr::get(permutedDims, shapeTy));
}

/// A pattern that detects a split with nonzero axis followed by a squeeze
/// (expressed as reshape) on each result, and rewrites them into a transpose
/// followed by a axis-0 split and the reshapes.
///
/// This is benefitial because
/// 1. split on axis-0 can be lowered into in-place split.
/// 2. the transpose kernel is much faster than the split kernel.
///
/// Example:
///
/// input : tensor<[..., N, ...]>
///   ress = split(input, /*splitSizes=*/[1, 1, ..., 1], /*axis=*/...)
///   res0 = reshape(ress#0) : tensor<[..., 1, ...]> -> tensor<[...]>
///   res1 = reshape(ress#1) : tensor<[..., 1, ...]> -> tensor<[...]>
///   ...
///   resN = reshape(ress#1) : tensor<[..., 1, ...]> -> tensor<[...]>
/// ===>
///   inputT = transpose(input) : ... -> tensor<[N, ...]>
///   ress   = split(inputT, /*splitSizes=*/[1, 1, ..., 1], /*axis=*/0)
///   res0   = reshape(ress#0) : tensor<[1, ...]> -> tensor<[...]>
///   res1   = reshape(ress#1) : tensor<[1, ...]> -> tensor<[...]>
///   ...
///   resN   = reshape(ress#1) : tensor<[1, ...]> -> tensor<[...]>
struct SplitAndSqueezesPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, SplitOp> {
  // TODO(#13349) Generalize or remove this pattern
  SplitAndSqueezesPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  // Check whether `op` is a `mo.static.reshape` (or rebind followed by a
  // reshape) that effectively squeezes the given axis `squeezeAxis`.
  static bool isSqueezingReshape(DeviceAwareRewriter &rewriter, Operation *op,
                                 ArrayRef<TypedAttr> inferredInDims,
                                 size_t squeezeAxis) {
    rewriter.pushOp(op);

    // TODO(#13348) remove this check once we have shape inference for SplitOp
    if (auto rebindOp = dyn_cast<RebindOp>(op)) {
      if (!rebindOp->hasOneUse())
        return false;
      op = *rebindOp->user_begin();
      rewriter.pushOp(op);
    }

    auto reshapeOp = dyn_cast<Static::ReshapeOp>(op);
    if (!reshapeOp)
      return false;

    TensorType inTy = reshapeOp.getInput().getType();
    TensorType outTy = reshapeOp.getResult().getType();
    if (!inTy.hasRank() || !outTy.hasRank())
      return false;

    // TODO(#13348) simplify all these once we have shape inference for SplitOp
    size_t inRank = inTy.getRank();
    if (inRank != outTy.getRank() + 1 || inRank <= squeezeAxis)
      return false;

    auto indexType = IndexType::get(op->getContext());
    auto intOneAttr = IntegerAttr::get(indexType, 1);
    SmallVector<TypedAttr> expectedInDims(
        outTy.getTypedShapeAttr().getValues());
    expectedInDims.insert(expectedInDims.begin() + squeezeAxis, intOneAttr);
    ArrayRef<TypedAttr> inDims = inTy.getTypedShapeAttr().getValues();
    for (auto [inDim, inferredInDim, expectedInDim] :
         llvm::zip(inDims, inferredInDims, expectedInDims)) {
      TypedAttr resolvedInDim =
          isa<IntegerAttr>(inferredInDim) ? inferredInDim : inDim;
      if (resolvedInDim != expectedInDim)
        return false;
    }

    return true;
  }

  LogicalResult
  deviceAwareMatchAndRewrite(SplitOp op,
                             DeviceAwareRewriter &rewriter) const override {
    TensorValue input = op.getInput();
    TensorType inputTy = input.getType();
    if (!inputTy.hasRank())
      return rewriter.notifyMatchFailure(op, "unranked input");
    int64_t inputRank = inputTy.getRank();

    // Split axis must be a constant
    std::optional<int64_t> maybeSplitAxis = tryGetConstantInt(op.getAxis());
    if (!maybeSplitAxis.has_value())
      return rewriter.notifyMatchFailure(op, "split axis isn't a constant");

    // Normalized split axis, and check it's positive.
    int64_t splitAxis = maybeSplitAxis.value();
    splitAxis = (splitAxis < 0) ? inputRank + splitAxis : splitAxis;
    if (splitAxis <= 0) {
      return rewriter.notifyMatchFailure(
          op, "normalized split axis isn't positive");
    }

    // Input split axis dimension must be static
    int64_t inputSplitAxisDim = inputTy.getIntDim(splitAxis);
    if (mlir::ShapedType::isDynamic(inputSplitAxisDim)) {
      return rewriter.notifyMatchFailure(
          op, "input split axis dimension isn't static");
    }

    // Split sizes must be all ones of length `inputSplitAxisDim`
    TensorValue splitSizes = op.getSplitSizes();
    std::optional<SmallVector<int64_t>> maybeSplitSizes =
        tryGetConstantIntVector(splitSizes);
    if (maybeSplitSizes != SmallVector<int64_t>(inputSplitAxisDim, 1)) {
      return rewriter.notifyMatchFailure(
          op, "split sizes isn't all ones of inputSplitAxisDim");
    }

    // NOTE this is basically doing shape inference, because we currently don't
    // have parameterization and shape function for `mo.split`.
    ArrayRef<TypedAttr> inputDims = inputTy.getTypedShapeAttr().getValues();
    SmallVector<TypedAttr> inferredResDims(inputDims);
    auto indexType = rewriter.getType<IndexType>();
    inferredResDims[splitAxis] = IntegerAttr::get(indexType, 1);

    // Each result must be consumed by a single reshape that squeezes
    for (Value res : op.getResults()) {
      if (!res.hasOneUse())
        return rewriter.notifyMatchFailure(op, "a result has multiple users");

      if (!isSqueezingReshape(rewriter, *res.user_begin(), inferredResDims,
                              splitAxis)) {
        return rewriter.notifyMatchFailure(
            op, "a result isn't consumed by a reshape that squeezes");
      }
    }

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    // Construct transpose permutation and zero axis value (before split op)
    Location loc = op.getLoc();
    SmallVector<int64_t> permVec(inputRank);
    std::iota(permVec.begin() + 1, permVec.begin() + splitAxis + 1, 0);
    std::iota(permVec.begin() + splitAxis + 1, permVec.end(), splitAxis + 1);
    permVec[0] = splitAxis;
    Value perm = rewriter.create<ConstantOp>(loc, permVec);
    Value zeroAxis = rewriter.create<ConstantOp>(loc, 0);

    // Construct transpose (before split op)
    TensorType transposedInputTy = permuteDims(inputTy, permVec);
    auto transposeOp =
        rewriter.create<TransposeOp>(loc, transposedInputTy, input, perm);

    // Construct new split op
    SmallVector<Type> newSplitResTypes;
    for (Type oldResTy : op.getResultTypes()) {
      auto oldResTensorTy = cast<TensorType>(oldResTy);
      TensorType newResTy = permuteDims(oldResTensorTy, permVec);
      newSplitResTypes.push_back(newResTy);
    }
    auto newSplitOp = rewriter.replaceOpWithNewOp<SplitOp>(
        op, newSplitResTypes, transposeOp, splitSizes, zeroAxis,
        op.getOutputParamDecls());

    // Rectify rebind consumers because the results dimensions have been
    // permuted. Note that since all rebind users are squeezing reshapes, they
    // are compatible with the new result types.
    // TODO(#13348) remove once we have shape inference for SplitOp
    for (Value res : newSplitOp.getResults()) {
      ASSERT_STREAM(res.hasOneUse(), << "guards should've enforced this");
      if (auto rebindOp = dyn_cast<RebindOp>(*res.user_begin())) {
        TensorType oldResTy = rebindOp.getResult().getType();
        TensorType newResTy = permuteDims(oldResTy, permVec);
        rebindOp.getResult().setType(newResTy);
      }
    }

    return success();
  }
};

/// A pattern that detects reshapes that correspond to a squeeze operation on
/// just a single axis followed by a transpose and re-orders the reshape after
/// the transpose:
///
/// (See findSqueezeAxis documentation on how exactly these reshapes are
/// identified)
///
/// Example:
///
///   input : tensor<[D1, 1, D2]>
///   reshaped : tensor<[D1, D2]>
///   transposed : tensor<[D2, D1]>
///
///   reshaped = reshape(input)
///   transposed = transpose(reshaped)
///
/// ========>
///
///   input : tensor<[D1, 1, D2]>
///   transposed : tensor<[D2, 1, D1]>
///   reshaped : tensor<[D2, D1]>
///
///   transposed = transpose(input)
///   reshaped = reshape(transposed)
///
/// See https://github.com/modularml/modular/issues/27800 for the motivating
/// context
struct SinkSqueezeLikeReshapeThroughTranspose
    : public DeviceAwareRewritePattern<OpRewritePattern, TransposeOp> {

  SinkSqueezeLikeReshapeThroughTranspose(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(TransposeOp op,
                             DeviceAwareRewriter &rewriter) const override {

    MO::TensorValue transposeInputType = op.getInput();

    auto transposeDefiningOp = transposeInputType.getDefiningOp();

    if (!transposeDefiningOp)
      return rewriter.notifyMatchFailure(
          op, "Input to transpose was a block argument");

    auto reshapeOp = dyn_cast<Static::ReshapeOp>(transposeDefiningOp);

    if (!reshapeOp)
      return rewriter.notifyMatchFailure(
          op, "Did not find corresponding reshape op");

    if (!op.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(op,
                                         "Transpose op has more than one use");

    MO::TensorType inputType = reshapeOp.getInput().getType();
    MO::TensorType outputType = reshapeOp.getResult().getType();

    if (!outputType.hasRank() || !inputType.hasRank())
      return rewriter.notifyMatchFailure(op,
                                         "Input/output ranks not fully known");

    auto maybeSqueezeIdx = findSqueezeAxis(inputType, outputType);

    if (!maybeSqueezeIdx)
      return rewriter.notifyMatchFailure(
          op, "Reshape does not correspond to a squeeze-like operation");

    int64_t squeezeIdx = *maybeSqueezeIdx;

    auto maybePermVec = tryGetConstantIntVector(op.getPerm());
    if (!maybePermVec.has_value())
      return rewriter.notifyMatchFailure(
          op, "Transpose permutation is not produced by a constant op");

    SmallVector<int64_t> permVec(maybePermVec.value());
    for (int64_t i = 0, size = permVec.size(); i < size; i++) {
      if (permVec[i] >= squeezeIdx)
        permVec[i] = permVec[i] + 1;
    }
    permVec.insert(permVec.begin() + squeezeIdx, squeezeIdx);

    Location loc = op->getLoc();

    auto constantOp = rewriter.create<MO::ConstantOp>(loc, permVec);
    auto permuted = permuteDims(inputType, permVec);

    auto newTransposeOp = rewriter.create<MO::TransposeOp>(
        loc, permuted, reshapeOp.getInput(), constantOp);

    rewriter.replaceOpWithNewOp<MO::Static::ReshapeOp>(
        op, op.getResult().getType(), newTransposeOp);

    return success();
  }
};

/// A pattern that detects binary operator where one operand is the result
/// of either a transpose or static reshape and the other is ultimately
/// the result of operations on solely constant values.
///
/// Example:
///
/// ┌─────┐         ┌────────┐               ┌─────┐  ┌────────┐
/// │input│         │constant│               │input│  │constant│
/// └──┬──┘         └───┬────┘               └──┬──┘  └───┬────┘
///    │                │                       │         │
///    │                │                       │       .....
///    │                │                       │         │
///  ViewOp           .....                     │     ViewOp*
///    │                │                       │         │
///    │                │         ───────►      └─binOp*──┘
///    └──── binOp ─────┘                           │
///            │                                    │
///            │                                  ViewOp*
///         ┌──┴───┐                                │
///         │output│                             ┌──┴───┐
///         └──────┘                             │output│
///                                              └──────┘
///
template <typename ViewOp>
struct SwimViewOpThroughBinaryOpWithConstantSubtree
    : public DeviceAwareRewritePattern<OpInterfaceRewritePattern,
                                       ElementWiseBinary> {
  SwimViewOpThroughBinaryOpWithConstantSubtree(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::ElementWiseBinary op,
                             DeviceAwareRewriter &rewriter) const override {

    TensorValue lhs = op.getLhsInput();
    TensorValue rhs = op.getRhsInput();
    TensorValue result = op.getResult();

    auto lhsProducer = lhs.getDefiningOp();
    auto rhsProducer = rhs.getDefiningOp();

    if (!lhsProducer || !rhsProducer) {
      return rewriter.notifyMatchFailure(
          op, "One of the inputs is a block argument");
    }

    if (!result.hasOneUse()) {
      return rewriter.notifyMatchFailure(
          op, "Only applies to binop outputs with one use");
    }

    // TODO (#36567):
    // This following early exit is a bit of a hack used to
    // avoid an infinite loop that can occur between the interaction
    // of SwimReshapeThroughBinaryOpWithConstantSubtree,
    // BroadcastReshapePattern, and ElementwiseBinaryVectorPattern
    if (result.getType().hasRank() && result.getType().getRank() == 1) {
      return rewriter.notifyMatchFailure(
          op, "Ignoring to not conflict with binary vector pattern");
    }

    Operation *maybeViewOp{nullptr};
    Operation *maybeConstantSubtreeRootOp{nullptr};

    Input constantSubtreeSide;

    if (isa<ViewOp>(lhsProducer) && !isa<ViewOp>(rhsProducer)) {
      maybeViewOp = lhsProducer;
      maybeConstantSubtreeRootOp = rhsProducer;
      constantSubtreeSide = Input::RHS;
    } else if (isa<ViewOp>(rhsProducer) && !isa<ViewOp>(lhsProducer)) {
      maybeViewOp = rhsProducer;
      maybeConstantSubtreeRootOp = lhsProducer;
      constantSubtreeSide = Input::LHS;
    } else if (isa<ViewOp>(lhsProducer) && isa<ViewOp>(rhsProducer)) {
      constantSubtreeSide = Input::UNK;
    } else {
      return rewriter.notifyMatchFailure(
          op, "Failed to match, no appropriate view op was found");
    }

    if (constantSubtreeSide != Input::UNK &&
        !dependsOnConstantsOnly<MO::ConstantOp>(maybeConstantSubtreeRootOp)) {
      return rewriter.notifyMatchFailure(
          op, "Failed to match, the view op input is not a constant subtree");
    }

    // Not applying the pattern to constant foldable subgraphs because there is
    // no gain and it can lead to cycles with other patterns that sink reshapes
    // through elementwise operations.
    bool lhsConstantFoldable = dependsOnConstantsOnly<ConstantOp>(lhsProducer);
    bool rhsConstantFoldable = dependsOnConstantsOnly<ConstantOp>(rhsProducer);
    if (lhsConstantFoldable && rhsConstantFoldable) {
      return rewriter.notifyMatchFailure(
          op, "Not applying the pattern to a constant foldable subgraph");
    }

    if (constantSubtreeSide == Input::UNK) {
      if (lhsConstantFoldable) {
        constantSubtreeSide = Input::LHS;
        maybeViewOp = rhsProducer;
        maybeConstantSubtreeRootOp = lhsProducer;
      } else if (rhsConstantFoldable) {
        constantSubtreeSide = Input::RHS;
        maybeViewOp = lhsProducer;
        maybeConstantSubtreeRootOp = rhsProducer;
      } else {
        return rewriter.notifyMatchFailure(
            op,
            "Failed to match, no constant subtree to the op could be found");
      }
    }

    ViewOp viewOp = dyn_cast<ViewOp>(maybeViewOp);

    // TODO (#36567):
    // This following early exit is a bit of a hack used to
    // avoid an infinite loop that can occur between the interaction
    // of SwimReshapeThroughBinaryOpWithConstantSubtree,
    // BroadcastReshapePattern, and ElementwiseBinaryVectorPattern
    if (ViewOp consumerOp =
            dyn_cast<ViewOp>(result.getUses().begin()->getOwner())) {

      if (consumerOp.getResult().getType() == viewOp.getInput().getType()) {
        return rewriter.notifyMatchFailure(
            op, "Immediate consumer reshapes back to original shape, skipping "
                "application");
      }
    }

    TensorValue cso =
        constantSubtreeSide == Input::LHS ? op.getLhsInput() : op.getRhsInput();

    auto constantSubtreeViewOp =
        materializeConstantSubtreeViewOp(rewriter, op, viewOp, cso);

    if (constantSubtreeViewOp.isError()) {
      return rewriter.notifyMatchFailure(op, constantSubtreeViewOp.getError());
    }

    switch (constantSubtreeSide) {
    case Input::LHS:
      op.setLhsInput(*constantSubtreeViewOp);
      op.setRhsInput(viewOp.getInput());
      break;
    case Input::RHS:
      op.setLhsInput(viewOp.getInput());
      op.setRhsInput(*constantSubtreeViewOp);
      break;
    default:
      llvm_unreachable("Should not reach here");
    }
    op.getResult().setType(viewOp.getInput().getType());

    rewriter.setInsertionPointAfter(op);
    auto binOpViewOp = materializeBinOpViewOp(rewriter, op, viewOp);
    rewriter.replaceAllUsesExcept(result, binOpViewOp.getResult(), binOpViewOp);

    return mlir::success();
  }

  /// The pattern works by moving the given ViewOp from the non-constant subtree
  /// to the top of the binary op and the top of the constant subtree
  ///
  /// The following two functions are responsible for creating these appropriate
  /// (modified) view ops at these locations
  virtual ErrorOr<ViewOp>
  materializeConstantSubtreeViewOp(DeviceAwareRewriter &rewriter,
                                   ElementWiseBinary binOp, ViewOp viewOp,
                                   TensorValue constantSubtreeOutput) const = 0;

  virtual ViewOp materializeBinOpViewOp(DeviceAwareRewriter &rewriter,
                                        ElementWiseBinary binOp,
                                        ViewOp viewOp) const = 0;

private:
  enum class Input {
    LHS,
    RHS,
    UNK,
  };
};

/// Swims transposes through binary operators when the other operand is
/// derived from constants
///
/// See SwimViewOpThroughBinaryOpWithConstantSubtree documentation for further
/// details
struct SwimTransposeThroughBinaryOpWithConstantSubtree
    : public SwimViewOpThroughBinaryOpWithConstantSubtree<TransposeOp> {
  SwimTransposeThroughBinaryOpWithConstantSubtree(MLIRContext *ctx)
      : SwimViewOpThroughBinaryOpWithConstantSubtree<TransposeOp>(ctx) {}

  ErrorOr<TransposeOp> materializeConstantSubtreeViewOp(
      DeviceAwareRewriter &rewriter, ElementWiseBinary binOp,
      TransposeOp viewOp,
      TensorValue constantSubtreeOutput) const override final {

    auto maybePermVec = tryGetConstantIntVector(viewOp.getPerm());

    if (!maybePermVec) {
      return {"Transpose permutation tensor was not a constant input"};
    }

    // Find the inverse permutation of the target transpose to fit on top of
    // the constant subtree
    const auto &permVec = maybePermVec.value();
    SmallVector<int64_t> invertedPerm(permVec.size());
    for (auto [idx, val] : llvm::enumerate(permVec)) {
      invertedPerm[val] = static_cast<int64_t>(idx);
    }

    auto invertedPermProducer =
        rewriter.create<ConstantOp>(binOp->getLoc(), invertedPerm);

    auto constantSubtreeTranspose = rewriter.create<TransposeOp>(
        binOp->getLoc(), viewOp.getInput().getType(), constantSubtreeOutput,
        invertedPermProducer);

    return constantSubtreeTranspose;
  }

  TransposeOp materializeBinOpViewOp(DeviceAwareRewriter &rewriter,
                                     ElementWiseBinary binOp,
                                     TransposeOp viewOp) const override final {
    auto out = rewriter.create<TransposeOp>(
        binOp->getLoc(), viewOp.getResult().getType(), binOp.getResult(),
        viewOp.getPerm());
    return out;
  }
};

/// Swims static reshapes through binary operators when the other operand is
/// derived from constants
///
/// See SwimViewOpThroughBinaryOpWithConstantSubtree documentation for further
/// details
struct SwimReshapeThroughBinaryOpWithConstantSubtree
    : public SwimViewOpThroughBinaryOpWithConstantSubtree<Static::ReshapeOp> {
  SwimReshapeThroughBinaryOpWithConstantSubtree(MLIRContext *ctx)
      : SwimViewOpThroughBinaryOpWithConstantSubtree<Static::ReshapeOp>(ctx) {}

  ErrorOr<Static::ReshapeOp> materializeConstantSubtreeViewOp(
      DeviceAwareRewriter &rewriter, ElementWiseBinary binOp,
      Static::ReshapeOp viewOp,
      TensorValue constantSubtreeOutput) const override final {

    return rewriter.create<Static::ReshapeOp>(
        binOp->getLoc(), viewOp.getInput().getType(), constantSubtreeOutput);
  }

  Static::ReshapeOp
  materializeBinOpViewOp(DeviceAwareRewriter &rewriter, ElementWiseBinary binOp,
                         Static::ReshapeOp viewOp) const override final {

    return rewriter.create<Static::ReshapeOp>(
        binOp->getLoc(), viewOp.getResult().getType(), binOp.getResult());
  }
};

struct ElementWiseUnaryVectorPattern
    : public DeviceAwareRewritePattern<OpInterfaceRewritePattern,
                                       ElementWiseUnary> {
  ElementWiseUnaryVectorPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::ElementWiseUnary op,
                             DeviceAwareRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TensorValue input = op.getInput();

    auto inType = llvm::cast<TensorType>(input.getType());

    if (!(inType.hasStaticShape() && inType.getRank() == 2 &&
          inType.getIntDim(0) != 1 && inType.getIntDim(1) == 1))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    Value opRes = op.getResult();

    TensorType finalOutputType = op.getResult().getType();
    TensorType newTypeShape = TensorType::get(
        SmallVector<int64_t>{inType.getIntDim(0)}, inType.getElementType());
    auto reshape = rewriter.create<Static::ReshapeOp>(
        loc, newTypeShape.cloneWith(input.getType().getDtype()), input);
    op.setInput(reshape.getResult());
    op.getResult().setType(
        newTypeShape.cloneWith(op.getResult().getType().getDtype()));
    rewriter.setInsertionPointAfter(op);
    auto out = rewriter.create<Static::ReshapeOp>(loc, finalOutputType,
                                                  op.getResult());
    rewriter.replaceAllUsesExcept(opRes, out.getResult(), out);
    return mlir::success();
  }
};

struct ElementWiseBinaryVectorPattern
    : public DeviceAwareRewritePattern<OpInterfaceRewritePattern,
                                       ElementWiseBinary> {
  ElementWiseBinaryVectorPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::ElementWiseBinary op,
                             DeviceAwareRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TensorValue a = op.getLhsInput();
    TensorValue b = op.getRhsInput();

    auto inType = llvm::cast<TensorType>(a.getType());

    if (!(inType.hasStaticShape() && inType.getRank() == 2 &&
          inType.getIntDim(0) != 1 && inType.getIntDim(1) == 1))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    TensorValue opRes = op.getResult();
    TensorType oldTensorType = opRes.getType();
    TensorType newTypeShape = TensorType::get(
        SmallVector<int64_t>{inType.getIntDim(0)}, inType.getElementType());
    auto reshapeA = rewriter.create<Static::ReshapeOp>(
        loc, newTypeShape.cloneWith(a.getType().getDtype()), a);
    auto reshapeB = rewriter.create<Static::ReshapeOp>(
        loc, newTypeShape.cloneWith(b.getType().getDtype()), b);
    rewriter.modifyOpInPlace(op, [&]() {
      op.setLhsInput(reshapeA.getResult());
      op.setRhsInput(reshapeB.getResult());
      op.getResult().setType(newTypeShape.cloneWith(oldTensorType.getDtype()));
    });
    rewriter.setInsertionPointAfter(op);
    auto out =
        rewriter.create<Static::ReshapeOp>(loc, oldTensorType, op.getResult());
    rewriter.replaceAllUsesExcept(opRes, out.getResult(), out);
    return mlir::success();
  }
};

struct BroadcastReshapePattern
    : public DeviceAwareRewritePattern<OpRewritePattern, Static::ReshapeOp> {
  BroadcastReshapePattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(Static::ReshapeOp op,
                             DeviceAwareRewriter &rewriter) const override {
    Value input;
    if (!m_Op<Static::ReshapeOp>(
             m_Op<Static::BroadcastToOp>(rewriter.pushOp(), m_Any(&input)))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input)
      return rewriter.notifyMatchFailure(op, "null input value");

    TensorType inType = llvm::cast<TensorType>(input.getType());

    // Check if the input is a scalar
    if (!inType.hasRank())
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    for (size_t i = 0, e = inType.getRank(); i < e; ++i)
      if (inType.getIntDim(i) != 1)
        return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    TensorType scalarType =
        TensorType::get(SmallVector<int64_t>{}, inType.getElementType());
    auto squash =
        rewriter.create<Static::ReshapeOp>(op.getLoc(), scalarType, input);

    rewriter.replaceOpWithNewOp<Static::BroadcastToOp>(op, op.getType(),
                                                       squash);
    return mlir::success();
  }
};

/// A pattern that replaces a split with several slices to enable fusion.
struct SplitToSlicePattern
    : public DeviceAwareRewritePattern<OpRewritePattern, SplitOp> {
  SplitToSlicePattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::SplitOp op,
                             DeviceAwareRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value in = op.getInput();

    if (!op.getOutputParamDecls().empty())
      return rewriter.notifyMatchFailure(op,
                                         "Op needs a shape function first.");

    TensorType inTy = cast<TensorType>(in.getType());
    if (inTy.hasUnknownShape() || !inTy.hasRank())
      return rewriter.notifyMatchFailure(op, "Pattern needs known input shape");
    size_t rank = inTy.getRank();

    // If all of the users are output ops, don't run as we know they won't be
    // fused in.
    for (Value out : op.getResults()) {
      for (Operation *user : out.getUsers()) {
        if (isa<MO::OutputOp>(user))
          return rewriter.notifyMatchFailure(op, "Split used in output.");
      }
    }

    // All the output shapes must have a known shape and be of same rank (sanity
    // check).
    if (llvm::any_of(op.getResultTypes(), [&](Type type) {
          auto tensor = cast<TensorType>(type);
          return tensor.hasUnknownShape() || !tensor.hasRank() ||
                 tensor.getRank() != rank;
        }))
      return rewriter.notifyMatchFailure(op, "Output shape is unknown.");

    // We must have a constant axis.
    std::optional<int64_t> maybeAxis = tryGetConstantInt(op.getAxis());
    if (!maybeAxis.has_value())
      return rewriter.notifyMatchFailure(op, "axis is not constant");
    int64_t axis = maybeAxis.value();
    if (axis == 0)
      return rewriter.notifyMatchFailure(
          op, "Avoid clashing with MGP optimization when axis == 0");
    axis = normalizeAxis(axis, inTy.getRank());

    // We must have constant split sizes.
    std::optional<SmallVector<int64_t>> maybeSplits =
        tryGetConstantIntVector(op.getSplitSizes());
    if (!maybeSplits.has_value())
      return rewriter.notifyMatchFailure(op, "Splits are not constant");
    SmallVector<int64_t> splitSizes = *maybeSplits;

    // Step is the same for each slice.
    SmallVector<int64_t> step(rank, 1);
    Value stepVal = rewriter.create<MO::ConstantOp>(loc, step);

    int64_t startVal = 0, splitIndex = 0;

    // Replace each result with a slice.
    for (Value result : op.getResults()) {
      SmallVector<int64_t> start, stop;

      // Build a start / stop combination for each slice.
      for (int64_t i = 0; i < static_cast<int64_t>(rank); ++i) {
        if (axis == i) {
          start.push_back(startVal);
          stop.push_back(startVal + splitSizes[splitIndex]);
          startVal += splitSizes[splitIndex];
          splitIndex++;
        } else {
          start.push_back(0);
          stop.push_back(std::numeric_limits<int64_t>::max());
        }
      }
      Value startVal = rewriter.create<MO::ConstantOp>(loc, start);
      Value stopVal = rewriter.create<MO::ConstantOp>(loc, stop);

      Value split = rewriter.create<MO::SliceOp>(loc, result.getType(), in,
                                                 startVal, stopVal, stepVal);
      rewriter.replaceAllUsesWith(result, split);
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

/// A pattern that replaces several slices into one split to enable split
/// patterns. This is the converse of SplitToSlicePattern.
struct SliceToSplitPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, SliceOp> {
  SliceToSplitPattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::SliceOp op,
                             DeviceAwareRewriter &rewriter) const override {
    SmallVector<SliceOp> sliceOps;
    Value input = op.getInput();
    for (auto user : input.getUsers()) {
      if (auto sliceOp = dyn_cast<SliceOp>(user))
        sliceOps.push_back(sliceOp);
    }

    if (sliceOps.size() == 1)
      return rewriter.notifyMatchFailure(op, "Only one slice op");

    if (!llvm::all_of(sliceOps, [](SliceOp sliceOp) {
          return sliceOp.getStart().getDefiningOp<ConstantOp>() &&
                 sliceOp.getStop().getDefiningOp<ConstantOp>();
        })) {
      return rewriter.notifyMatchFailure(
          op, "Start and step operands must be constants");
    }

    auto inputType = llvm::cast<TensorType>(input.getType());
    int64_t sliceDimension;
    const auto sliceHasStaticRankKnownShapeAndSlicesOneDimension =
        [&](SliceOp sliceOp) {
          auto type = sliceOp.getResult().getType();
          if (!type.hasRank())
            return false;

          if (type.hasUnknownShape())
            return false;

          int64_t differentDimensions = 0;
          int64_t i = 0;
          for (auto [before, after] :
               llvm::zip(inputType.getDims(), type.getDims())) {
            if (before != after) {
              sliceDimension = i;
              ++differentDimensions;
            }
            ++i;
          }

          return differentDimensions == 1;
        };

    if (!llvm::all_of(sliceOps,
                      sliceHasStaticRankKnownShapeAndSlicesOneDimension)) {
      return rewriter.notifyMatchFailure(
          op, "All slices must have a static rank, a known shape and slice "
              "through only one dimension");
    }

    if (llvm::any_of(sliceOps, [&](SliceOp sliceOp) {
          return sliceOp.getType().getIntDim(sliceDimension) ==
                 ShapedType::kDynamic;
        })) {
      return rewriter.notifyMatchFailure(
          op, "Sliced dimension must be statically known for all slices");
    }

    int64_t nElements = 0;
    for (SliceOp sliceOp : sliceOps)
      nElements += sliceOp.getType().getIntDim(sliceDimension);

    if (nElements != inputType.getIntDim(sliceDimension)) {
      return rewriter.notifyMatchFailure(
          op, "Union of slices must cover all of the input");
    }

    const auto getSliceValue = [&](Value value) {
      return SmallVector<APInt>(
                 cast<IntArrayElementsAttr>(
                     value.getDefiningOp<ConstantOp>().getValue())
                     .getValues())[sliceDimension]
          .getSExtValue();
    };

    std::sort(sliceOps.begin(), sliceOps.end(), [&](SliceOp lhs, SliceOp rhs) {
      return getSliceValue(lhs.getStart()) < getSliceValue(rhs.getStop());
    });

    for (uint64_t i = 1; i < sliceOps.size(); ++i) {
      if (getSliceValue(sliceOps[i].getStart()) <
          getSliceValue(sliceOps[i - 1].getStop())) {
        return rewriter.notifyMatchFailure(op, "Slices are overlapping");
      }
    }

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);

    SmallVector<Type> splitTypes;
    for (SliceOp sliceOp : sliceOps)
      splitTypes.push_back(sliceOp.getType());

    SmallVector<int64_t> splitSizes;
    for (SliceOp sliceOp : sliceOps) {
      splitSizes.push_back(getSliceValue(sliceOp.getStop()) -
                           getSliceValue(sliceOp.getStart()));
    }
    auto splitSizesOp = rewriter.create<ConstantOp>(loc, splitSizes);

    SmallVector<Value> originalSliceValues;
    for (SliceOp sliceOp : sliceOps)
      originalSliceValues.push_back(sliceOp.getResult());

    auto splitAxis =
        rewriter.create<ConstantOp>(loc, sliceDimension).getResult();

    auto splitOp = rewriter.create<SplitOp>(
        loc, splitTypes, input, splitSizesOp.getResult(), splitAxis);

    rewriter.replaceAllUsesWith(originalSliceValues, splitOp.getResults());
    for (SliceOp sliceOp : sliceOps)
      rewriter.eraseOp(sliceOp);

    return success();
  }
};

// clang-format off
/// A pattern that detects a matmuls that use the same activations and merges
/// them to increase throughput.
/// BEFORE:
/// matmul(A, B1) -> C1
/// matmul(A, B2) -> C2
/// matmul(A, B3) -> C3
/// AFTER:
/// matmul(A, concat(B1, B2, B3)) -> C -> split -> C1
///                                            |-> C2
///                                            |-> C3
// clang-format on
template <typename MatmulOpType>
struct MergeParallelMatmulPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, MatmulOpType> {
  MergeParallelMatmulPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern<OpRewritePattern, MatmulOpType>(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MatmulOpType op,
                             DeviceAwareRewriter &rewriter) const override {
    // Gather all matmul operations that use the same activations.
    Value activations = op.getInputA();
    SmallVector<MatmulOpType> matmulUsers;
    for (Operation *user : activations.getUsers()) {
      if (auto matmulOp = dyn_cast<MatmulOpType>(user)) {
        rewriter.pushOp(matmulOp);
        matmulUsers.push_back(matmulOp);
      }
    }

    if (matmulUsers.size() == 1)
      return rewriter.notifyMatchFailure(op, "No parallel matmuls");

    if (llvm::all_of(matmulUsers, [](MatmulOpType matmul) {
          return llvm::all_of(matmul->getUsers(), [](Operation *user) {
            return isa<OutputOp>(user);
          });
        })) {
      // A return op will force the materialization of the split which ends up
      // slowing down the program.
      return rewriter.notifyMatchFailure(
          op, "Matmuls are all feeding into output ops");
    }

    // Ensure that all matmuls have the same number of rows and that weights are
    // constants.
    auto oldMatmulType = cast<TensorType>(op.getType());
    TypedAttr expectedNumberOfRows =
        oldMatmulType.getDim(oldMatmulType.getRank() - 2);
    if (!llvm::all_of(matmulUsers, [&](MatmulOpType matmulOp) {
          auto resultType = matmulOp.getType();

          if (!resultType.hasRank())
            return false;

          if (resultType.getDim(0) != expectedNumberOfRows)
            return false;

          if (mlir::ShapedType::isDynamic(resultType.getIntDim(1)))
            return false;

          if (!dependsOnConstantsOnly<ConstantOp>(
                  matmulOp.getInputB().getDefiningOp()))
            return false;

          return true;
        })) {
      return rewriter.notifyMatchFailure(
          op, "Parallel matmuls must have the same number of rows, a static "
              "number of columns and constant weights");
    }

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    // The last of matmul Users is the earliest location.
    auto firstMatmulIndex = matmulUsers.size() - 1;
    Location loc = matmulUsers[firstMatmulIndex]->getLoc();
    rewriter.setInsertionPoint(matmulUsers[firstMatmulIndex]);

    SmallVector<Value> weightsToConcatenate;
    SmallVector<Type> matmulTypes;
    SmallVector<int64_t> columnsInEachMatmul;
    int64_t totalColumns = 0;
    for (MatmulOpType matmulOp : matmulUsers) {
      weightsToConcatenate.push_back(matmulOp.getInputB());
      TensorType matmulType = matmulOp.getType();
      matmulTypes.push_back(matmulType);
      columnsInEachMatmul.push_back(matmulType.getIntDims().back());
      totalColumns += columnsInEachMatmul.back();
    }

    // Concatenate the weights.
    auto concatType =
        cast<TensorType>(op.getInputB().getType())
            .cloneWithNewDim(oldMatmulType.getRank() - 1, totalColumns);
    Value concatAxis = rewriter.create<ConstantOp>(loc, 1);
    Value concatenatedWeights =
        rewriter
            .create<ConcatOp>(loc, concatType, concatAxis, weightsToConcatenate)
            .getResult();

    // Create a single larger matmul.
    auto newMatmulType = oldMatmulType.cloneWithNewDim(
        oldMatmulType.getRank() - 1, totalColumns);
    Value newMatmul = rewriter
                          .create<MatmulOpType>(loc, newMatmulType, activations,
                                                concatenatedWeights)
                          .getResult();

    // Split the output.
    Value splitSizes = rewriter.create<ConstantOp>(loc, columnsInEachMatmul);
    auto splitOp = rewriter.create<SplitOp>(loc, matmulTypes, newMatmul,
                                            splitSizes, concatAxis);

    for (uint64_t idx = 0; idx < matmulUsers.size(); ++idx) {
      rewriter.replaceAllUsesWith(matmulUsers[idx].getResult(),
                                  splitOp.getResult(idx));
      // Erase to avoid re-doing the same optimization on the other matmuls.
      rewriter.eraseOp(matmulUsers[idx]);
    }

    return mlir::success();
  }
};

// Returns true if the split operates on the last dimension and that the
// dimension is statically known.
static bool isStaticSplit(SplitOp op) {
  auto inputType = llvm::cast<TensorType>(op.getInput().getType());
  if (!inputType.hasRank())
    return false;

  auto axisOr = tryGetConstantInt(op.getAxis());
  if (!axisOr)
    return false;

  int64_t axis = normalizeAxis(*axisOr, inputType.getRank());
  if (!isa<IntegerAttr>(inputType.getDim(axis)))
    return false;

  return true;
}

template <typename... OpTypes>
static bool allResultsAreUniquelyUsedBy(Operation *op) {
  return llvm::all_of(op->getResults(), [](Value result) {
    auto users = result.getUsers();
    if (llvm::range_size(users) != 1)
      return false;

    return isa<OpTypes...>(*users.begin());
  });
}

/// A pattern that will move a split/ operation after identical elementwise
/// operations if certain conditions are met
/// CONDITIONS:
/// - Split on the last axis.
/// - Last axis of the split is statically known.
/// - Split must be uniform (same size).
/// - Elementwise must be the same.
/// - Elementwise operand use must be the same.
/// - Elementwise are fed with a constant or a constant and a broadcast.
/// - The innermost dimension must be unchanged after the broadcast.
/// - All broadcasts must be the same.
/// BEFORE:
/// -> split -> elemwise ->
///          -> elemwise ->
///          -> elemwise ->
/// AFTER:
/// -> elemwise -> split ->
struct SinkSplitThroughElementwisePattern
    : public DeviceAwareRewritePattern<OpRewritePattern, SplitOp> {
  SinkSplitThroughElementwisePattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(SplitOp op,
                             DeviceAwareRewriter &rewriter) const override {
    if (!isStaticSplit(op) ||
        !allResultsAreUniquelyUsedBy<ElementWiseUnary, ElementWiseBinary>(op)) {
      return rewriter.notifyMatchFailure(
          op, "Split op must be statically ranked and have one use and one "
              "user per result");
    }

    auto firstResultType =
        llvm::cast<TensorType>(op.getResults().front().getType());
    int64_t splitAxis = normalizeAxis(*tryGetConstantInt(op.getAxis()),
                                      firstResultType.getRank());

    if (splitAxis != static_cast<int64_t>(firstResultType.getRank() - 1))
      return rewriter.notifyMatchFailure(op, "Split must be on last axis");

    SmallVector<Value> elementwiseValues;
    for (auto result : op.getResults()) {
      if (result.getType() != firstResultType) {
        return rewriter.notifyMatchFailure(
            op, "Split results don't have the same size");
      }

      Operation *user = result.getUses().begin()->getOwner();
      rewriter.pushOp(user);
      elementwiseValues.push_back(user->getResult(0));
    }

    auto &firstUse = *op->getResult(0).getUses().begin();
    uint64_t firstUseIndex = firstUse.getOperandNumber();
    Operation *firstOwner = firstUse.getOwner();

    SmallVector<Value> constantsToConcatenate;
    Static::BroadcastToOp broadcastOp;
    for (auto result : op.getResults()) {
      auto &use = *result.getUses().begin();

      uint64_t index = use.getOperandNumber();
      if (index != firstUseIndex) {
        return rewriter.notifyMatchFailure(
            op, "Split result use is not at the same operand index");
      }

      Operation *owner = use.getOwner();
      if (owner->getName() != firstOwner->getName()) {
        return rewriter.notifyMatchFailure(
            op, "Split results users are not all the same operation");
      }

      if (isa<ElementWiseBinary>(owner)) {
        uint64_t otherIndex = 1 - index;
        auto firstOtherOperandOp =
            firstOwner->getOperand(otherIndex).getDefiningOp();
        auto ownerOtherOperandOp =
            owner->getOperand(otherIndex).getDefiningOp();
        if (firstOtherOperandOp->getName() != ownerOtherOperandOp->getName()) {
          return rewriter.notifyMatchFailure(
              op, "Operations that feed the elementwise don't match");
        }

        if (!isa<ConstantOp, Static::BroadcastToOp>(ownerOtherOperandOp)) {
          return rewriter.notifyMatchFailure(
              op, "Operations that feed the elementwise must be constants or "
                  "static broadcasts");
        }

        auto constantOp = dyn_cast<ConstantOp>(ownerOtherOperandOp);
        if ((broadcastOp =
                 dyn_cast<Static::BroadcastToOp>(ownerOtherOperandOp))) {
          constantOp = dyn_cast<ConstantOp>(
              ownerOtherOperandOp->getOperand(0).getDefiningOp());
          if (!constantOp) {
            return rewriter.notifyMatchFailure(
                op, "Input to the static broadcast must be a constant");
          }
        }
        assert(constantOp);

        constantsToConcatenate.push_back(constantOp.getResult());
        auto constantOpType = llvm::cast<TensorType>(constantOp.getType());
        if (constantOpType.getRank() == 0 ||
            constantOpType.getDims().back() !=
                firstResultType.getDims().back()) {
          return rewriter.notifyMatchFailure(
              op, "Unsupported broadcasting that would duplicate data");
        }
      }
    }

    if (!llvm::all_of(constantsToConcatenate, [&](auto constantOp) {
          return constantOp.getType() ==
                 constantsToConcatenate.front().getType();
        })) {
      return rewriter.notifyMatchFailure(
          op, "Constants are being broadcasted differently");
    }

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    auto inputType = llvm::cast<TensorType>(op.getInput().getType());
    uint64_t lastDimSize =
        llvm::cast<IntegerAttr>(inputType.getDims().back()).getInt();

    SmallVector<Value, 2> elementwiseOperands;
    if (!constantsToConcatenate.empty()) {
      TensorType constantType =
          llvm::cast<TensorType>(constantsToConcatenate.front().getType());
      auto concatType =
          constantType.cloneWithNewDim(constantType.getRank() - 1, lastDimSize);
      Value concatAxis =
          rewriter.create<ConstantOp>(loc, int64_t(constantType.getRank() - 1));
      Value concatenatedConstants =
          rewriter
              .create<ConcatOp>(loc, concatType, concatAxis,
                                constantsToConcatenate)
              .getResult();

      Value broadcast = concatenatedConstants;
      if (broadcastOp) {
        auto broadcastType =
            broadcastOp.getType().cloneWithNewDim(splitAxis, lastDimSize);
        broadcast = rewriter
                        .create<Static::BroadcastToOp>(loc, broadcastType,
                                                       concatenatedConstants)
                        .getResult();
      }

      elementwiseOperands.resize(2);
      elementwiseOperands[1 - firstUseIndex] = broadcast;
      elementwiseOperands[firstUseIndex] = op.getInput();
    } else {
      elementwiseOperands.push_back(op.getInput());
    }

    auto elementwiseType =
        llvm::cast<TensorType>(firstOwner->getResultTypes().front())
            .cloneWithNewDim(splitAxis, lastDimSize);
    OperationState elementwiseOpState(loc, firstOwner->getName(),
                                      elementwiseOperands, elementwiseType);
    auto newElementwise = rewriter.create(elementwiseOpState)->getResult(0);

    SmallVector<Type> newSplitTypes;
    for (auto val : elementwiseValues)
      newSplitTypes.push_back(val.getType());
    SmallVector<Value> newSplitOpOperands = op.getOperands();
    newSplitOpOperands[0] = newElementwise;
    OperationState splitOpState(loc, op->getName(), newSplitOpOperands,
                                newSplitTypes, op->getAttrs(),
                                op->getSuccessors());
    auto newSplit = rewriter.create(splitOpState);

    SmallVector<Operation *> elemwiseOps(op->getUsers());
    SmallVector<Value> splitResults(newSplit->getResults());
    for (uint64_t i = 0; i < elemwiseOps.size(); ++i) {
      rewriter.replaceAllUsesWith(elemwiseOps[i]->getResult(0),
                                  splitResults[i]);
      rewriter.eraseOp(elemwiseOps[i]);
    }

    rewriter.eraseOp(op);

    return success();
  }
};

/// This pattern will sink a split operation through a ReshapeOpType (usually a
/// static reshape or a rebind op).
/// CONDITIONS:
/// - Split axis is statically known.
/// - Split dimension is statically known.
/// - All split results are used at least once.
/// - Split must be uniform (same size).
/// - All split users must be `ReshapeOpType`.
/// - The reshape must not mix dynamic and static dimensions.
/// - The split axis must not be mixed with left dimensions after the reshape.
/// - [Unchecked] Row major-like memory layout.
/// BEFORE:
/// -> split -> reshape ->
///          -> reshape ->
///          -> reshape ->
/// AFTER:
/// -> reshape -> split ->
///                     ->
///                     ->
template <typename ReshapeOpType>
struct SinkSplitThroughStaticReshapeLikeOpsPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, SplitOp> {
  SinkSplitThroughStaticReshapeLikeOpsPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(SplitOp op,
                             DeviceAwareRewriter &rewriter) const override {
    if (!isStaticSplit(op) || !allResultsAreUniquelyUsedBy<ReshapeOpType>(op)) {
      return rewriter.notifyMatchFailure(
          op, "Split op must be statically ranked, operate on last statically "
              "known axis and have one use and one user per result");
    }

    auto resultTypes = op->getResultTypes();
    if (!llvm::all_of(resultTypes, [&](Type resultType) {
          return resultType == resultTypes.front();
        })) {
      return rewriter.notifyMatchFailure(op, "Split is not uniform");
    }

    auto users = op->getUsers();
    if (!llvm::all_of(users, [&](Operation *user) {
          auto reshapeOp = llvm::dyn_cast<ReshapeOpType>(user);
          auto type = reshapeOp.getType();

          return type.getRank() &&
                 type == cast<ReshapeOpType>(*users.begin()).getType();
        })) {
      return rewriter.notifyMatchFailure(
          op, "Users must be identical non-scalar reshapes");
    }

    for (Operation *user : users)
      rewriter.pushOp(user);

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    auto splitType = llvm::cast<TensorType>(op.getType(0));
    auto shapeBeforeReshape = splitType.getDims();
    auto shapeAfterReshape =
        llvm::cast<TensorType>(users.begin()->getResults().front().getType())

            .getDims();
    const StaticReshapeDimensionsMappingAnalysis
        staticReshapeDimensionsMappingAnalysis(shapeBeforeReshape,
                                               shapeAfterReshape);
    if (!staticReshapeDimensionsMappingAnalysis)
      return rewriter.notifyMatchFailure(op, "Unsupported reshape");

    int64_t splitAxis =
        normalizeAxis(*tryGetConstantInt(op.getAxis()), splitType.getRank());

    uint64_t mappedSplitAxis = splitAxis;
    // Will stop at the first non 1 dimension, dynamic dimension included. This
    // is because 1s are ignored by the analysis and they serve no real purpose
    // in the reshape.
    while (mappedSplitAxis < splitType.getRank() &&
           splitType.getIntDim(mappedSplitAxis) == 1) {
      ++mappedSplitAxis;
    }

    // If there is no non-1 dimension and or if that dimension is dynamic, then
    // the reshape is unsupported. This is because there is no way to split it
    // with the new dimensions.
    const bool couldNotFindAnotherAxis = mappedSplitAxis == splitType.getRank();
    if (couldNotFindAnotherAxis ||
        !isa<IntegerAttr>(splitType.getDim(mappedSplitAxis)))
      return rewriter.notifyMatchFailure(op, "Trailing ones is unsupported");

    // If the dimension was expanded and it was not mixed with other dimensions
    // or if it was contracted but the mixing was done with dimensions that are
    // on the right, then it can proceed. This assumes a row major memory
    // layout.
    // What mixing means is that two dimensions from the original shape
    // contributed to the same dimension in the reshaped shape. This prevents a
    // split unless the split axis is the leftmost of the mixed dimensions (in
    // row major).
    const bool couldMix =
        staticReshapeDimensionsMappingAnalysis
            .backwardMapping()[staticReshapeDimensionsMappingAnalysis
                                   .forwardMapping()[mappedSplitAxis]
                                   .front()]
            .size() > 1;
    const bool mixedButSplitAxisIsFirst =
        staticReshapeDimensionsMappingAnalysis
            .backwardMapping()[staticReshapeDimensionsMappingAnalysis
                                   .forwardMapping()[mappedSplitAxis]
                                   .front()]
            .front() == static_cast<int64_t>(mappedSplitAxis);
    if (couldMix && !mixedButSplitAxisIsFirst) {
      return rewriter.notifyMatchFailure(op,
                                         "Reshape is mixing static dimensions");
    }

    // The new split axis is the left most axis in the shape after the reshape.
    const int64_t newSplitAxis =
        staticReshapeDimensionsMappingAnalysis.forwardMapping()[mappedSplitAxis]
            .front();

    Location loc = op->getLoc();
    rewriter.setInsertionPoint(op);

    auto firstReshapeType = cast<ReshapeOpType>(*users.begin()).getType();
    SmallVector<TypedAttr> newReshapeShape(firstReshapeType.getDims());
    newReshapeShape[newSplitAxis] = rewriter.getIndexAttr(
        llvm::cast<IntegerAttr>(newReshapeShape[newSplitAxis]).getInt() *
        op->getNumResults());
    auto newReshapeType = TensorType::get(
        ShapeAttr::get(newReshapeShape, rewriter.getType<MOSH::ShapeType>()),
        firstReshapeType.getDtype());
    auto newReshape =
        rewriter.create<ReshapeOpType>(loc, newReshapeType, op.getInput())
            .getResult();

    SmallVector<Type> newSplitTypes(op.getNumResults(), firstReshapeType);
    SmallVector<int64_t> newSplitSizes(
        op.getNumResults(), firstReshapeType.getIntDim(newSplitAxis));
    SmallVector<Value> newSplitOpOperands = op.getOperands();
    newSplitOpOperands[0] = newReshape;
    newSplitOpOperands[1] = rewriter.create<ConstantOp>(loc, newSplitSizes);
    newSplitOpOperands[2] =
        rewriter.create<ConstantOp>(loc, int64_t(newSplitAxis)).getResult();
    OperationState splitOpState(loc, op->getName(), newSplitOpOperands,
                                newSplitTypes, op->getAttrs(),
                                op->getSuccessors());
    auto newSplit = rewriter.create(splitOpState);

    SmallVector<Operation *> reshapeOps(users);
    SmallVector<Value> splitResults(newSplit->getResults());
    for (uint64_t i = 0; i < reshapeOps.size(); ++i) {
      rewriter.replaceAllUsesWith(reshapeOps[i]->getResult(0), splitResults[i]);
      rewriter.eraseOp(reshapeOps[i]);
    }

    rewriter.eraseOp(op);

    return success();
  }
};

/// Attempts to sink broadcasts through reduction style operators
///
/// E.g.
///
///   Input                     Input      Axis
///     │                         │         │
///     ▼                         │         │
/// ┌─────────┐                   ▼         ▼
/// │Broadcast│   Axis           ┌───────────┐
/// └────┬────┘    │             │ReductionOp│
///      │         │             └─────┬─────┘
///      ▼         ▼    ─────►         │
///     ┌───────────┐                  ▼
///     │ReductionOp│             ┌─────────┐
///     └─────┬─────┘             │Broadcast│
///           │                   └────┬────┘
///           ▼                        │
///                                    ▼
///
/// This general transformation can only be applied as long as the axis of
/// reduction is done on a dimension which is not modified in the broadcast.
///
/// Why is this possible? One easy way to get an intuition is seeing broadcast
/// as a "view"/"reindexing" operation of an original buffer.
///
/// For example if:
///    %T1 : !mo.tensor<[A, B, C, 1], f32>
///    %T2 : mo.static.broadcast_to(%A):
///         !mo.tensor<[A, B, C, 1], f32> -> !mo.tensor<[P1, A, B, C, P2], f32>
///
/// Then T2[p1, a, b, c, p2] == T1[a, b, c, 0] for all valid variables!
///
/// Therefore if we are reduce-add across dimension 'A':
///
/// T2[p1, 0, b, c, p2] = \sum_i T2[p1, i, b, c, p2]
///                     = \sum_i T1[i, b, c, 0]
///
/// Which is a reduction across dimension 'A' in the original tensor, with
/// a broadcast with the same initial output shape!
///
/// When the axis of reduction is modified by the broadcast, it means the
/// reduction is applied on an axis with identical values, and the whole
/// Reduction op can be replaced by a simpler op (before applying a
/// BroadcastTo):
/// - For reduce.add (reduce.mul), the input must be multiplied (use pow) by the
/// size of the axis.
/// - For min / max / mean, the input isn't modfied (identity).
/// - For argmin / argmax, the input is transformed into a tensor of 0s.
struct SinkBroadcastsThroughReduction
    : public DeviceAwareRewritePattern<OpInterfaceRewritePattern, Reduction> {
  SinkBroadcastsThroughReduction(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(Reduction op,
                             DeviceAwareRewriter &rewriter) const override {
    TensorValue input = op.getInput();
    Location loc = op->getLoc();
    MLIRContext *ctx = op->getContext();

    auto bcastOp = input.getDefiningOp<Static::BroadcastToOp>();
    if (!bcastOp)
      return rewriter.notifyMatchFailure(loc, "Not preceded by broadcast.");

    auto bcastOpInput = bcastOp.getInput();
    if (!bcastOpInput.getType().hasRank() || !bcastOp.getType().hasRank())
      return rewriter.notifyMatchFailure(
          loc, "BCast to be statically ranked to work.");

    std::optional<int64_t> maybeAxis = tryGetConstantInt(op.getAxis());
    if (!maybeAxis.has_value())
      return rewriter.notifyMatchFailure(
          loc, "Need to know axis to perform optimization.");
    int64_t reductionAxis = maybeAxis.value();

    // respect negative axis rules
    int64_t initialRank = input.getType().getRank();
    if (reductionAxis < -initialRank || reductionAxis >= initialRank)
      return rewriter.notifyMatchFailure(
          loc, "Axis is not within [-rank, rank - 1]");

    reductionAxis = normalizeAxis(reductionAxis, input.getType().getRank());

    // Here we want to map the axes we are reducing on in the broadcasted tensor
    // to on the non-broadcasted input.
    //
    // E.g. if we broadcasted [A, B, 1] --> [1, D, A, B, 1]
    //
    // And we are reducing along the dim with 'A', then we just need to count
    // the number of additional broadcasted dimensions before the given axis.
    // In this case it is 2, since we added axis `1` and `D`.
    auto bcastShapeAttr = bcastOp.getType().getTypedShapeAttr();
    ErrorOr<SmallVector<int64_t>> maybeBcastedAxis =
        ShapeAttr::getBroadcastedAxes(
            bcastOp.getInput().getType().getTypedShapeAttr(), bcastShapeAttr);
    if (maybeBcastedAxis.isError())
      return rewriter.notifyMatchFailure(loc, maybeBcastedAxis.getError());

    SmallVector<int64_t> &bcastedAxis = maybeBcastedAxis.get();
    size_t numBeforeAxis = 0;
    for (int64_t bcastedDimIndex : bcastedAxis) {
      if (bcastedDimIndex == reductionAxis) {
        // When we reduce along the broadcasted axis the reduction op can be
        // simplified.
        auto maybeNewInput =
            TypeSwitch<Operation *, FailureOr<TensorValue>>(op)
                .Case<ReduceMinOp, ReduceMaxOp, MeanOp>([&](auto op) {
                  // For min / max / mean, the reduction is removed.
                  return bcastOpInput;
                })
                .Case<ReduceAddOp, ReduceMulOp>([&](auto op) {
                  // For reduce.add, the input is multiplied by the size of the
                  // broadcast axis. (For reduce.mul, we use mo.pow instead).
                  TypedAttr dimAttr = bcastShapeAttr[reductionAxis];
                  auto dimIndex =
                      rewriter.create<MOSH::ParamToValueOp>(loc, dimAttr);
                  auto dimT = rewriter.create<IndexToTensorOp>(
                      loc,
                      TensorType::get(
                          ArrayRef<int64_t>({}),
                          IntegerType::get(ctx, 64, IntegerType::Unsigned)),
                      dimIndex);
                  auto dimCast = rewriter.create<CastOp>(
                      loc, dimT, bcastOpInput.getType().getDtype());
                  auto dimBroadcast = rewriter.create<Static::BroadcastToOp>(
                      loc,
                      dimCast.getType().cloneWith(
                          bcastOpInput.getType().getShapeAttr()),
                      dimCast);
                  TensorValue scaledInput =
                      isa<ReduceAddOp>(op)
                          ? rewriter
                                .create<MulOp>(loc, bcastOpInput, dimBroadcast)
                                .getResult()
                          : rewriter
                                .create<PowOp>(loc, bcastOpInput.getType(),
                                               bcastOpInput, dimBroadcast)
                                .getResult();
                  return scaledInput;
                })
                .Case<ArgMinOp, ArgMaxOp>([&](auto op) {
                  // For arg_min / arg_max, the output becomes a tensor of 0s.
                  auto v0 = rewriter.create<MO::ConstantOp>(loc, 0);
                  TensorValue v0Cast = rewriter.create<CastOp>(
                      loc, v0, op.getResult().getType().getDtype());
                  return v0Cast;
                })
                .Default([&](auto op) {
                  return rewriter.notifyMatchFailure(
                      loc, "Cannot optimize if reducing along axes modified by "
                           "broadcast: Unsupported reduce operation");
                });
        if (failed(maybeNewInput))
          return failure();

        rewriter.replaceOpWithNewOp<Static::BroadcastToOp>(
            op, op.getResult().getType(), *maybeNewInput);
        return success();
      }
      if (bcastedDimIndex > reductionAxis)
        break;
      numBeforeAxis++;
    }

    // Now we know we are reducing along a non-broadcasted axis, we can elide
    // the broadcast for later.
    int64_t newAxis = reductionAxis - numBeforeAxis;
    TensorValue newAxisTensor =
        rewriter.create<ConstantOp>(op->getLoc(), newAxis);
    auto clonedReduction = cast<MO::Reduction>(*rewriter.clone(*op));
    clonedReduction.getInputMutable().assign(bcastOpInput);
    clonedReduction.getAxisMutable().assign(newAxisTensor);

    TensorType newTy = bcastOpInput.getType()
                           .cloneWithNewDim(newAxis, 1)
                           .cloneWith(op.getResult().getType().getDtype());
    clonedReduction.getResult().setType(newTy);

    rewriter.replaceOpWithNewOp<Static::BroadcastToOp>(
        op, op.getResult().getType(), clonedReduction.getResult());

    return success();
  }
};

template <typename ElemwiseInterfaceT>
struct SinkBroadcastsThroughElemwise
    : public DeviceAwareRewritePattern<OpInterfaceRewritePattern,
                                       ElemwiseInterfaceT> {
  SinkBroadcastsThroughElemwise(MLIRContext *ctx)
      : DeviceAwareRewritePattern<OpInterfaceRewritePattern,
                                  ElemwiseInterfaceT>(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(ElemwiseInterfaceT op,
                             DeviceAwareRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // We first ensure the elementwise op has only broadcast inputs, and collect
    // the input shapes of the broadcasts.
    SmallVector<TypedAttr> shapeParams;
    for (Value arg : op->getOperands()) {
      auto bcastOp = arg.getDefiningOp<Static::BroadcastToOp>();
      if (!bcastOp)
        return rewriter.notifyMatchFailure(loc, "not preceded by broadcast");
      shapeParams.emplace_back(bcastOp.getInput().getType().getShapeAttr());
    }

    // We calculate the smallest common shapes of the broadcast inputs.
    TypedAttr commonBcastShape =
        ShapeAttr::getCommonBroadcastShape(shapeParams);

    // Here we cannot prove a better shape, so just use the existing shape
    // annotation which we assume works well.
    if (!commonBcastShape)
      return rewriter.notifyMatchFailure(
          loc, "No common shape can be proven for broadcasts.");

    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          loc, "Assume only one result from elemwise op.");

    TensorType oldTy = cast<TensorType>(op->getResult(0).getType());
    if (commonBcastShape == oldTy.getShapeAttr())
      return rewriter.notifyMatchFailure(
          loc, "Already at the mode reduced broadcast form.");

    // Clone the elemwise operator
    TensorValue resVal = cast<TensorValue>(op->getResult(0));
    Operation *clonedOp = rewriter.clone(*op);
    TensorValue clonedResVal = cast<TensorValue>(clonedOp->getResult(0));

    // Set the operands to inputs broadcast to the common shape
    rewriter.setInsertionPoint(clonedOp);
    for (size_t i = 0; i < op->getNumOperands(); i++) {
      auto oldBcastOp =
          op->getOperand(i).template getDefiningOp<Static::BroadcastToOp>();
      TensorType targetType = oldBcastOp.getType().cloneWith(commonBcastShape);

      // Only insert broadcast and replace operand if it changes shapes
      if (oldBcastOp.getType() != targetType) {
        TensorValue newOperand = rewriter.create<Static::BroadcastToOp>(
            oldBcastOp->getLoc(), targetType, oldBcastOp.getInput());
        clonedOp->setOperand(i, newOperand);
      }
    }
    clonedResVal.setType(resVal.getType().cloneWith(commonBcastShape));

    // To type check, insert a broadcast back to the original shape
    rewriter.setInsertionPointAfter(clonedOp);
    auto newResVal = rewriter.create<Static::BroadcastToOp>(
        op->getLoc(), resVal.getType(), clonedResVal);
    rewriter.replaceAllUsesWith(resVal, newResVal);

    return success();
  }
};

struct SinkBroadcastsThroughLinalgBandPartOp
    : public DeviceAwareRewritePattern<OpRewritePattern, LinAlg::BandPartOp> {
  SinkBroadcastsThroughLinalgBandPartOp(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(LinAlg::BandPartOp op,
                             DeviceAwareRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MO::TensorType resTy = op.getType();
    if (!resTy.hasRank())
      return rewriter.notifyMatchFailure(loc, "not statically ranked");

    MO::TensorValue inputVal = op.getInput();
    auto bcastOp = inputVal.getDefiningOp<MO::Static::BroadcastToOp>();
    if (!bcastOp)
      return rewriter.notifyMatchFailure(loc, "not preceded by broadcast");

    MO::TensorValue bcastInput = bcastOp.getInput();
    MO::TensorType bcastInputTy = bcastInput.getType();
    if (!bcastInputTy.hasRank())
      return rewriter.notifyMatchFailure(
          loc, "preceding broadcast has unranked input");

    MO::TensorType inputTy = inputVal.getType();
    SmallVector<TypedAttr> newBcastResDims;
    if (bcastInputTy.getRank() > 2) {
      // If broadcasting in the batch dimensions, we check their equality.
      ArrayRef<TypedAttr> inBatchDims = bcastInputTy.getDims().drop_back(2);
      ArrayRef<TypedAttr> outBatchDims = inputTy.getDims().drop_back(2);
      if (inBatchDims.equals(outBatchDims))
        return rewriter.notifyMatchFailure(
            loc, "preceding broadcast with batch dims canonical");
      newBcastResDims.append(inBatchDims.begin(), inBatchDims.end());
    } else {
      // We are broadcasting from something less than rank-2, so it's enough to
      // ensure that we are broadcasting to rank-2.
      if (inputTy.getRank() == 2)
        return rewriter.notifyMatchFailure(
            loc, "preceding broadcast without batch dims canonical");
    }

    // We will inject a new broadcast before the band part op that does not
    // broadcast in the batch dimensions. If there is no broadcasting in the
    // trailing two dimensions (e.g. because they are static and equal) the
    // broadcast canonicalizer will erase the op.
    ArrayRef<TypedAttr> matDims = inputTy.getDims().take_back(2);
    newBcastResDims.append(matDims.begin(), matDims.end());

    auto shapeTy = rewriter.getType<ShapeType>();
    MO::TensorType newBcastResTy =
        inputTy.cloneWith(ShapeAttr::get(newBcastResDims, shapeTy));
    auto newBcastOp = rewriter.create<MO::Static::BroadcastToOp>(
        loc, newBcastResTy, bcastInput);

    // We replace the bandpart op using a new input and result type.
    SmallVector<Value> newOperands(op.getOperands());
    newOperands[0] = newBcastOp;
    auto newBandPartOp = rewriter.create<MO::LinAlg::BandPartOp>(
        loc, newBcastResTy, newOperands,
        op.getOperation()->getAttrDictionary().getValue());

    // Note that the original result type is used.
    rewriter.replaceOpWithNewOp<MO::Static::BroadcastToOp>(op, resTy,
                                                           newBandPartOp);
    return success();
  }
};

template <typename T>
static inline ErrorOr<MO::ConstantOp>
createScalarConstantOpOfType(M::DType dtype, T value,
                             DeviceAwareRewriter &rewriter, Location loc) {
  return dtype.dispatch<ErrorOr<MO::ConstantOp>>()
      .template when<M::DType::si32>([&]() {
        return rewriter.create<MO::ConstantOp>(loc, int32_t(value));
      })
      .template when<M::DType::si64>([&]() {
        return rewriter.create<MO::ConstantOp>(loc, int64_t(value));
      })
      .template when<M::DType::f32>(
          [&]() { return rewriter.create<MO::ConstantOp>(loc, float(value)); })
      .template when<M::DType::f64>(
          [&]() { return rewriter.create<MO::ConstantOp>(loc, double(value)); })
      .template when<M::DType::kBool>(
          [&]() { return rewriter.create<MO::ConstantOp>(loc, bool(value)); })
      .otherwise([&]() { return Error("Unsupported dtype"); });
}

template <typename T>
static inline ErrorOr<Value>
createBroadcastedScalarConstantOpOfType(TensorType bcastType, T value,
                                        DeviceAwareRewriter &rewriter,
                                        Location loc) {
  auto cstErrOr =
      createScalarConstantOpOfType(bcastType.getDtype(), value, rewriter, loc);
  if (cstErrOr.isError())
    return Error(cstErrOr.getError());

  return rewriter.create<MO::Static::BroadcastToOp>(loc, bcastType, *cstErrOr);
}

template <typename OpType, typename T, typename Func>
static LogicalResult
replaceOpResultWithBcastOfScalarIf(OpType op, T value,
                                   DeviceAwareRewriter &rewriter, Func &&func) {
  if (!func())
    return failure();

  auto bcastErrOr = createBroadcastedScalarConstantOpOfType(
      op.getType(), value, rewriter, op.getLoc());
  if (bcastErrOr.isError())
    return rewriter.notifyMatchFailure(op, bcastErrOr.getError());

  rewriter.replaceAllUsesWith(op, *bcastErrOr);

  return success();
}

// add(x, 0) -> x
struct ArithmeticFoldingAddWithZero
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::AddOp> {
  ArithmeticFoldingAddWithZero(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::AddOp addOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_BinaryCommutativeOp<MO::AddOp>(
             m_Any(&x),
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 0))
             .match(addOp)) {
      return failure();
    }
    rewriter.replaceAllUsesWith(addOp, x);

    return success();
  }
};

// sub(x, 0) -> x
struct ArithmeticFoldingSubWithZero
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::SubOp> {
  ArithmeticFoldingSubWithZero(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::SubOp subOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_Op<MO::SubOp>(
             m_Any(&x),
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 0))
             .match(subOp)) {
      return failure();
    }
    rewriter.replaceAllUsesWith(subOp, x);

    return success();
  }
};

// sub(x, x) -> bcast(0)
struct ArithmeticFoldingSubWithSame
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::SubOp> {
  ArithmeticFoldingSubWithSame(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::SubOp subOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(subOp, 0, rewriter, [&]() {
      return subOp.getInputX() == subOp.getInputY();
    });

    return success();
  }
};

// mul(x, 1) -> x
struct ArithmeticFoldingMulWithOne
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MulOp> {
  ArithmeticFoldingMulWithOne(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MulOp mulOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_BinaryCommutativeOp<MO::MulOp>(
             m_Any(&x),
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 1))
             .match(mulOp)) {
      return failure();
    }
    rewriter.replaceAllUsesWith(mulOp, x);

    return success();
  }
};

// mul(x, 0) -> bcast(0)
struct ArithmeticFoldingMulWithZero
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MulOp> {
  ArithmeticFoldingMulWithZero(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MulOp mulOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(mulOp, 0, rewriter, [&]() {
      return m_BinaryCommutativeOp<MO::MulOp>(
                 m_Any(), m_SplatOrBroadcastedConstantOfAnyType<
                              MO::Static::BroadcastToOp>(0))
          .match(mulOp);
    });

    return success();
  }
};

// div(0, x) -> bcast(0)
struct ArithmeticFoldingZeroDiv
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::DivOp> {
  ArithmeticFoldingZeroDiv(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::DivOp divOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(divOp, 0, rewriter, [&]() {
      return m_Op<MO::DivOp>(m_SplatOrBroadcastedConstantOfAnyType<
                                 MO::Static::BroadcastToOp>(0),
                             m_Any())
          .match(divOp);
    });
  }
};

// div(x, 1) -> x
struct ArithmeticFoldingDivByOne
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::DivOp> {
  ArithmeticFoldingDivByOne(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::DivOp divOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_Op<MO::DivOp>(
             m_Any(&x),
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 1))
             .match(divOp)) {
      return failure();
    }
    rewriter.replaceAllUsesWith(divOp, x);

    return success();
  }
};

// pow(x, 1) -> x
struct ArithmeticFoldingPowerByOne
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::PowOp> {
  ArithmeticFoldingPowerByOne(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::PowOp powOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_Op<MO::PowOp>(m_Any(&x), m_SplatConstantOfAnyType(1)).match(powOp))
      return failure();

    rewriter.replaceAllUsesWith(powOp, x);

    return success();
  }
};

// pow(x, 0) -> bcast(1)
struct ArithmeticFoldingPowByZero
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::PowOp> {
  ArithmeticFoldingPowByZero(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::PowOp powOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(powOp, 1, rewriter, [&]() {
      return m_Op<MO::PowOp>(m_Any(), m_SplatOrBroadcastedConstantOfAnyType<
                                          MO::Static::BroadcastToOp>(0))
          .match(powOp);
    });
  }
};

// pow(0, y) -> bcast(0)
struct ArithmeticFoldingPowerOfZero
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::PowOp> {
  ArithmeticFoldingPowerOfZero(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::PowOp powOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(powOp, 0, rewriter, [&]() {
      return m_Op<MO::PowOp>(m_SplatOrBroadcastedConstantOfAnyType<
                                 MO::Static::BroadcastToOp>(0),
                             m_Any())
          .match(powOp);
    });
  }
};

// pow(1, y) -> bcast(1)
struct ArithmeticFoldingPowerOfOne
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::PowOp> {
  ArithmeticFoldingPowerOfOne(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::PowOp powOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(powOp, 1, rewriter, [&]() {
      return m_Op<MO::PowOp>(m_SplatOrBroadcastedConstantOfAnyType<
                                 MO::Static::BroadcastToOp>(1),
                             m_Any())
          .match(powOp);
    });

    return success();
  }
};

// max(x, x) -> x
struct ArithmeticFoldingMaxSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MaxOp> {
  ArithmeticFoldingMaxSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MaxOp maxOp,
                             DeviceAwareRewriter &rewriter) const override {
    if (maxOp.getInputX() != maxOp.getInputY())
      return failure();

    rewriter.replaceAllUsesWith(maxOp, maxOp.getInputX());

    return success();
  }
};

// min(x, x) -> x
struct ArithmeticFoldingMinSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MinOp> {
  ArithmeticFoldingMinSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MinOp minOp,
                             DeviceAwareRewriter &rewriter) const override {
    if (minOp.getInputX() != minOp.getInputY())
      return failure();

    rewriter.replaceAllUsesWith(minOp, minOp.getInputX());

    return success();
  }
};

// min(x, max(x, y)) -> x
struct ArithmeticFoldingMinOfMax
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MinOp> {
  ArithmeticFoldingMinOfMax(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MinOp minOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x, y, z;
    if (!m_BinaryCommutativeOp<MO::MinOp>(
             m_Any(&x), m_BinaryCommutativeOp<MO::MaxOp>(m_Any(&y), m_Any(&z)))
             .match(minOp)) {
      return failure();
    }

    if (x != z && x != y)
      return failure();

    rewriter.replaceAllUsesWith(minOp, x);

    return success();
  }
};
// max(x, min(x, y)) -> x
struct ArithmeticFoldingMaxOfMin
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MaxOp> {
  ArithmeticFoldingMaxOfMin(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MaxOp maxOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x, y, z;
    if (!m_BinaryCommutativeOp<MO::MaxOp>(
             m_Any(&x), m_BinaryCommutativeOp<MO::MinOp>(m_Any(&y), m_Any(&z)))
             .match(maxOp)) {
      return failure();
    }

    if (x != z && x != y)
      return failure();

    rewriter.replaceAllUsesWith(maxOp, x);

    return success();
  }
};

// mod(x, x) -> bcast(0)
struct ArithmeticFoldingModSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::ModOp> {
  ArithmeticFoldingModSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::ModOp modOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(modOp, 0, rewriter, [&]() {
      return modOp.getInputX() == modOp.getInputY() &&
             !m_Optional(
                  m_SplatConstantOfAnyType(0),
                  m_Op<MO::Static::BroadcastToOp>(m_SplatConstantOfAnyType(0)))
                  .match(modOp.getInputX());
    });
  }
};

// mod(x, 1) -> bcast(0)
struct ArithmeticFoldingModByOne
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::ModOp> {
  ArithmeticFoldingModByOne(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::ModOp modOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(modOp, 0, rewriter, [&]() {
      return m_Op<MO::ModOp>(
                 m_Any(), m_Optional(m_Optional(m_SplatConstant(int64_t(1)),
                                                m_SplatConstant(int32_t(1))),
                                     m_Op<MO::Static::BroadcastToOp>(m_Optional(
                                         m_SplatConstant(int64_t(1)),
                                         m_SplatConstant(int32_t(1))))))
          .match(modOp);
    });
  }
};

// >=(x, x) -> bcast(true)
struct ArithmeticFoldingGreatherOrEqualThanSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::GreaterEqualOp> {
  ArithmeticFoldingGreatherOrEqualThanSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::GreaterEqualOp geOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(geOp, true, rewriter, [&]() {
      return geOp.getInputX() == geOp.getInputY();
    });

    return success();
  }
};

// >(x, x) -> bcast(false)
struct ArithmeticFoldingGreatherThanSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::GreaterOp> {
  ArithmeticFoldingGreatherThanSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::GreaterOp gtOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(gtOp, false, rewriter, [&]() {
      return gtOp.getInputX() == gtOp.getInputY();
    });
  }
};

// !=(x, x) -> bcast(false)
struct ArithmeticFoldingNotEqualSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::NotEqualOp> {
  ArithmeticFoldingNotEqualSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::NotEqualOp neOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(neOp, false, rewriter, [&]() {
      return neOp.getInputX() == neOp.getInputY();
    });

    return success();
  }
};

// ==(x, x) -> bcast(true)
struct ArithmeticFoldingEqualSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::EqualOp> {
  ArithmeticFoldingEqualSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::EqualOp eqOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(eqOp, true, rewriter, [&]() {
      return eqOp.getInputX() == eqOp.getInputY();
    });
  }
};

// and(x, x) -> x
struct ArithmeticFoldingAndSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::AndOp> {
  ArithmeticFoldingAndSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::AndOp andOp,
                             DeviceAwareRewriter &rewriter) const override {
    if (andOp.getInputX() != andOp.getInputY())
      return failure();

    rewriter.replaceAllUsesWith(andOp, andOp.getInputX());

    return success();
  }
};

// or(x, x) -> x
struct ArithmeticFoldingOrSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::OrOp> {
  ArithmeticFoldingOrSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::OrOp orOp,
                             DeviceAwareRewriter &rewriter) const override {
    if (orOp.getInputX() != orOp.getInputY())
      return failure();

    rewriter.replaceAllUsesWith(orOp, orOp.getInputX());

    return success();
  }
};

// xor(x, x) -> bcast(false)
struct ArithmeticFoldingXorSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::XorOp> {
  ArithmeticFoldingXorSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::XorOp xorOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(xorOp, false, rewriter, [&]() {
      return xorOp.getInputX() == xorOp.getInputY();
    });
  }
};

// pow(pow(x, y), z) = pow(x, y * z)
// This pattern can be repeatedly applied for arbitrarily long chains.
struct MathematicalCanonicalizationChainsOfPowToSinglePow
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::PowOp> {
  MathematicalCanonicalizationChainsOfPowToSinglePow(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::PowOp powOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x, y, z;
    if (!m_Op<MO::PowOp>(m_Op<MO::PowOp>(m_Any(&x), m_Any(&y)), m_Any(&z))
             .match(powOp)) {
      return failure();
    }

    if (y.getType() != z.getType())
      return failure();

    Location loc = powOp->getLoc();
    auto add = rewriter.create<MO::MulOp>(loc, y.getType(), y, z);
    auto newPow = rewriter.create<MO::PowOp>(loc, powOp.getType(), x, add);
    rewriter.replaceAllUsesWith(powOp, newPow);

    return success();
  }
};

// rsqrt(x) -> div(bcast(1), sqrt(x))
struct ForwardMathematicalCanonicalizationRsqrtToDivSqrt
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::RsqrtOp> {
  ForwardMathematicalCanonicalizationRsqrtToDivSqrt(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::RsqrtOp rsqrtOp,
                             DeviceAwareRewriter &rewriter) const override {
    Location loc = rsqrtOp->getLoc();
    TensorType tensorType = rsqrtOp.getType();
    auto bcast1ErrOr =
        createBroadcastedScalarConstantOpOfType(tensorType, 1, rewriter, loc);
    if (bcast1ErrOr.isError())
      return rewriter.notifyMatchFailure(rsqrtOp, bcast1ErrOr.getError());
    auto sqrt =
        rewriter.create<MO::SqrtOp>(loc, tensorType, rsqrtOp.getInput());
    auto div = rewriter.create<MO::DivOp>(loc, tensorType, *bcast1ErrOr, sqrt);
    rewriter.replaceAllUsesWith(rsqrtOp, div);

    return success();
  }
};

// div(bcast(1), sqrt(x)) -> rsqrt(x)
struct BackwardMathematicalCanonicalizationDivSqrtToRsqrt
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::DivOp> {
  BackwardMathematicalCanonicalizationDivSqrtToRsqrt(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::DivOp divOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_Op<MO::DivOp>(
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 1),
             m_Op<MO::SqrtOp>(m_Any(&x)))
             .match(divOp)) {
      return failure();
    }
    auto rsqrt =
        rewriter.create<MO::RsqrtOp>(divOp->getLoc(), divOp.getType(), x);
    rewriter.replaceAllUsesWith(divOp, rsqrt);

    return success();
  }
};

// relu(x) -> max(x, bcast(0))
struct ForwardMathematicalCanonicalizationReluToMax
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::ReluOp> {
  ForwardMathematicalCanonicalizationReluToMax(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::ReluOp reluOp,
                             DeviceAwareRewriter &rewriter) const override {
    Location loc = reluOp->getLoc();
    TensorType tensorType = reluOp.getType();

    auto bcast0ErrOr =
        createBroadcastedScalarConstantOpOfType(tensorType, 0, rewriter, loc);
    if (bcast0ErrOr.isError())
      return rewriter.notifyMatchFailure(reluOp, bcast0ErrOr.getError());
    auto maxOp = rewriter.create<MO::MaxOp>(loc, tensorType, reluOp.getInput(),
                                            *bcast0ErrOr);
    rewriter.replaceAllUsesWith(reluOp, maxOp);

    return success();
  }
};

// max(x, bcast(0)) -> relu(x)
struct BackwardMathematicalCanonicalizationMaxToRelu
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MaxOp> {
  BackwardMathematicalCanonicalizationMaxToRelu(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MaxOp maxOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_BinaryCommutativeOp<MO::MaxOp>(
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 0),
             m_Any(&x))
             .match(maxOp)) {
      return failure();
    }
    auto relu =
        rewriter.create<MO::ReluOp>(maxOp->getLoc(), maxOp.getType(), x);
    rewriter.replaceAllUsesWith(maxOp, relu);

    return success();
  }
};

// mul(x, bcast(2)) = add(x, x)
struct ForwardMathematicalCanonicalizationMulByTwoToAdd
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MulOp> {
  ForwardMathematicalCanonicalizationMulByTwoToAdd(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MulOp mulOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_BinaryCommutativeOp<MO::MulOp>(
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 2),
             m_Any(&x))
             .match(mulOp)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<MO::AddOp>(mulOp, mulOp.getType(), x, x);

    return success();
  }
};

// add(x, x) = mul(x, bcast(2))
struct BackwardMathematicalCanonicalizationMulByTwoToAdd
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::AddOp> {
  BackwardMathematicalCanonicalizationMulByTwoToAdd(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::AddOp addOp,
                             DeviceAwareRewriter &rewriter) const override {
    if (addOp.getInputX() != addOp.getInputY()) {
      return failure();
    }

    auto tensorType = addOp.getType();
    auto loc = addOp->getLoc();
    auto bcast2ErrOr =
        createBroadcastedScalarConstantOpOfType(tensorType, 2, rewriter, loc);
    if (bcast2ErrOr.isError())
      return rewriter.notifyMatchFailure(addOp, bcast2ErrOr.getError());

    rewriter.replaceOpWithNewOp<MO::MulOp>(addOp, tensorType, addOp.getInputX(),
                                           *bcast2ErrOr);

    return success();
  }
};

// pow(x, 2) = mul(x, x)
struct ForwardMathematicalCanonicalizationPowByTwoToMul
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::PowOp> {
  ForwardMathematicalCanonicalizationPowByTwoToMul(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::PowOp powOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_Op<MO::PowOp>(
             m_Any(&x),
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 2))
             .match(powOp)) {
      return failure();
    }
    auto mul =
        rewriter.create<MO::MulOp>(powOp->getLoc(), powOp.getType(), x, x);
    rewriter.replaceAllUsesWith(powOp, mul);

    return success();
  }
};

// pow(x, -{1, 2}) = div(1, pow(x, {1, 2})
struct ForwardMathematicalCanonicalizationPowByNegativeOneOrTwoToDivAndPow
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::PowOp> {
  ForwardMathematicalCanonicalizationPowByNegativeOneOrTwoToDivAndPow(
      MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::PowOp powOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    Value one;
    Value two;
    if (!m_Op<MO::PowOp>(m_Any(&x),
                         m_Optional(m_SplatOrBroadcastedConstantOfAnyType<
                                        MO::Static::BroadcastToOp>(&one, -1),
                                    m_SplatOrBroadcastedConstantOfAnyType<
                                        MO::Static::BroadcastToOp>(&two, -2)))
             .match(powOp)) {
      return failure();
    }

    auto tensorType = powOp.getType();
    auto loc = powOp->getLoc();

    auto bcastErrOr = createBroadcastedScalarConstantOpOfType(
        tensorType, one ? 1 : 2, rewriter, loc);
    if (bcastErrOr.isError())
      return rewriter.notifyMatchFailure(powOp, bcastErrOr.getError());

    auto bcast1ErrOr =
        createBroadcastedScalarConstantOpOfType(tensorType, 1, rewriter, loc);
    if (bcast1ErrOr.isError())
      return rewriter.notifyMatchFailure(powOp, bcast1ErrOr.getError());

    rewriter.modifyOpInPlace(powOp, [&]() { powOp.setRhsInput(*bcastErrOr); });
    rewriter.setInsertionPointAfter(powOp);
    auto div = rewriter.create<MO::DivOp>(loc, tensorType, *bcast1ErrOr, powOp);
    rewriter.replaceAllUsesExcept(powOp, div, div);

    return success();
  }
};

// div(1, pow(x, {1, 2}) = pow(x, -{1, 2})
struct BackwardMathematicalCanonicalizationDivAndPowToPowByNegativeOneOrTwo
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::DivOp> {
  BackwardMathematicalCanonicalizationDivAndPowToPowByNegativeOneOrTwo(
      MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::DivOp divOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    Value one;
    Value two;
    if (!m_Op<MO::DivOp>(
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 1),
             m_Op<MO::PowOp>(
                 m_Any(&x),
                 m_Optional(m_SplatOrBroadcastedConstantOfAnyType<
                                MO::Static::BroadcastToOp>(&one, 1),
                            m_SplatOrBroadcastedConstantOfAnyType<
                                MO::Static::BroadcastToOp>(&two, 2))))
             .match(divOp)) {
      return failure();
    }

    assert(one || two);

    auto tensorType = divOp.getType();
    auto loc = divOp->getLoc();

    auto bcastErrOr = createBroadcastedScalarConstantOpOfType(
        tensorType, one ? -1 : -2, rewriter, loc);
    if (bcastErrOr.isError())
      return rewriter.notifyMatchFailure(divOp, bcastErrOr.getError());

    auto pow = rewriter.create<MO::PowOp>(loc, tensorType, x, *bcastErrOr);
    rewriter.replaceAllUsesWith(divOp, pow);

    return success();
  }
};

// sub(0, x) = negative(x)
struct MathematicalCanonicalizationZeroSubToNegative
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::SubOp> {
  MathematicalCanonicalizationZeroSubToNegative(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::SubOp subOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_Op<MO::SubOp>(
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 0),
             m_Any(&x))
             .match(subOp)) {
      return failure();
    }
    auto neg =
        rewriter.create<MO::NegativeOp>(subOp->getLoc(), subOp.getType(), x);
    rewriter.replaceAllUsesWith(subOp, neg);

    return success();
  }
};

// add(x, negative(y)) = sub(x, y)
struct MathematicalSimplificationAddWithNegative
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::AddOp> {
  MathematicalSimplificationAddWithNegative(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::AddOp addOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x, y;
    if (!m_BinaryCommutativeOp<MO::AddOp>(m_Any(&x),
                                          m_Op<MO::NegativeOp>(m_Any(&y)))
             .match(addOp)) {
      return failure();
    }

    auto sub =
        rewriter.create<MO::SubOp>(addOp->getLoc(), addOp.getType(), x, y);
    rewriter.replaceAllUsesWith(addOp, sub);

    return success();
  }
};

// sub(x, negative(y)) = add(x, y)
struct MathematicalSimplificationSubWithNegative
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::SubOp> {
  MathematicalSimplificationSubWithNegative(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::SubOp subOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x, y;
    if (!m_Op<MO::SubOp>(m_Any(&x), m_Op<MO::NegativeOp>(m_Any(&y)))
             .match(subOp)) {
      return failure();
    }

    auto add =
        rewriter.create<MO::AddOp>(subOp->getLoc(), subOp.getType(), x, y);
    rewriter.replaceAllUsesWith(subOp, add);

    return success();
  }
};

// div(x, div(y, z)) = div(mul(x, z), y)
struct MathematicalSimplificationDivByDivToDivOfMul
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::DivOp> {
  MathematicalSimplificationDivByDivToDivOfMul(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::DivOp divOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x, y, z;
    if (!m_Op<MO::DivOp>(m_Any(&x), m_Op<MO::DivOp>(m_Any(&y), m_Any(&z)))
             .match(divOp)) {
      return failure();
    }

    auto mul =
        rewriter.create<MO::MulOp>(divOp->getLoc(), divOp.getType(), x, z);
    rewriter.modifyOpInPlace(divOp, [&]() {
      divOp.setLhsInput(mul);
      divOp.setRhsInput(y);
    });

    return success();
  }
};

// div(div(x, y), z) = div(x, mul(y, z))
struct MathematicalSimplificationDivOfDivToDivByMul
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::DivOp> {
  MathematicalSimplificationDivOfDivToDivByMul(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::DivOp divOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x, y, z;
    if (!m_Op<MO::DivOp>(m_Op<MO::DivOp>(m_Any(&x), m_Any(&y)), m_Any(&z))
             .match(divOp)) {
      return failure();
    }

    auto mul =
        rewriter.create<MO::MulOp>(divOp->getLoc(), divOp.getType(), y, z);
    rewriter.modifyOpInPlace(divOp, [&]() {
      divOp.setLhsInput(x);
      divOp.setRhsInput(mul);
    });

    return success();
  }
};

// div(x, bcast(CST_or_small)) = mul(x, bcast(div(bcast(1), CST_or_small)))
struct MathematicalSimplificationDivBySmallTensorToMul
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::DivOp> {
  MathematicalSimplificationDivBySmallTensorToMul(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::DivOp divOp,
                             DeviceAwareRewriter &rewriter) const override {
    auto tensorType = divOp.getType();
    if (!tensorType.hasStaticShape())
      return failure();

    Location loc = divOp->getLoc();
    if (!tensorType.getDtype().isFloat())
      return rewriter.notifyMatchFailure(loc, "input type isn't float");

    auto x = divOp.getInputX();
    auto y = divOp.getInputY();

    // This is to avoid looping.
    bool xConstantFoldable =
        dependsOnConstantsOnly<MO::ConstantOp>(x.getDefiningOp());
    if (xConstantFoldable)
      return failure();

    Value origY = y;
    if (auto bcast =
            dyn_cast_or_null<MO::Static::BroadcastToOp>(origY.getDefiningOp()))
      origY = bcast.getInput();

    auto origYType = llvm::cast<TensorType>(origY.getType());
    if (!origYType.hasStaticShape())
      return failure();

    static constexpr int64_t kMulByDivReciprocalThroughputEstimate = 8;
    // Add 1 because it still takes time to do a multiplication.
    static constexpr int64_t kThreshold =
        kMulByDivReciprocalThroughputEstimate + 1;
    // This is checking if the broadcasted value is a constant or if it was
    // broadcasted by a factor of at least `kThreshold`.
    bool yConstantFoldable =
        dependsOnConstantsOnly<MO::ConstantOp>(y.getDefiningOp());
    // Avoid division by 0.
    if (!yConstantFoldable && (*origYType.getNumElements() == 0 ||
                               (*tensorType.getNumElements() /
                                *origYType.getNumElements()) < kThreshold)) {
      return failure();
    }

    auto bcast1ErrOr =
        createBroadcastedScalarConstantOpOfType(origYType, 1, rewriter, loc);
    if (bcast1ErrOr.isError())
      return rewriter.notifyMatchFailure(divOp, bcast1ErrOr.getError());

    auto div = rewriter.create<MO::DivOp>(loc, origYType, *bcast1ErrOr, origY);
    auto bcastDiv =
        rewriter.create<MO::Static::BroadcastToOp>(loc, tensorType, div);
    auto mul = rewriter.create<MO::MulOp>(loc, tensorType, x, bcastDiv);

    rewriter.replaceAllUsesWith(divOp, mul);

    return success();
  }
};

// mul(x, div(1, y)) = div(x, y)
// where y is not a constant or a broadcasted value.
// This pattern is kind of the opposite of the previous one except it will not
// affect constants or broadcasted values.
struct MathematicalSimplificationMulOfDivToDiv
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MulOp> {
  MathematicalSimplificationMulOfDivToDiv(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MulOp mulOp,
                             DeviceAwareRewriter &rewriter) const override {
    if (!mulOp.getInputY().getDefiningOp())
      return failure();

    Value x, y;
    if (!m_Op<MulOp>(m_Any(&x),
                     m_Op<DivOp>(m_SplatOrBroadcastedConstantOfAnyType<
                                     MO::Static::BroadcastToOp>(1),
                                 m_Any(&y)))
             .match(mulOp)) {
      return failure();
    }

    if (isa_and_nonnull<MO::Static::BroadcastToOp>(y.getDefiningOp()) ||
        dependsOnConstantsOnly<MO::ConstantOp>(y.getDefiningOp())) {
      return failure();
    }

    auto div =
        rewriter.create<MO::DivOp>(mulOp->getLoc(), mulOp.getType(), x, y);
    rewriter.replaceAllUsesWith(mulOp, div);

    return success();
  }
};

// pow(x, bcast(CST_float)) = pow(x, bcast(int(CST_float)))
// where CST_float == float(int(CST_float))
struct MathematicalSimplificationPowByFloatToPowByInt
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::PowOp> {
  MathematicalSimplificationPowByFloatToPowByInt(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::PowOp powOp,
                             DeviceAwareRewriter &rewriter) const override {

    auto bcast = dyn_cast_or_null<MO::Static::BroadcastToOp>(
        powOp.getInputY().getDefiningOp());
    if (!bcast)
      return failure();

    auto cst =
        dyn_cast_or_null<MO::ConstantOp>(bcast.getInput().getDefiningOp());
    if (!cst)
      return failure();

    auto dtype = cst.getType().getDtype();
    if (!dtype.isFloat())
      return failure();

    std::optional<double> value = [&]() -> std::optional<double> {
      if (auto val = tryGetConstantFloat(cst))
        return *val;
      if (auto val = tryGetConstantDouble(cst))
        return *val;

      return {};
    }();

    if (!value)
      return rewriter.notifyMatchFailure(
          powOp, "unable to get the floating point exponent");

    double exponent;
    if (std::modf(*value, &exponent))
      return rewriter.notifyMatchFailure(
          powOp, "the exponent contains a fractional part");

    // Get the bitwidth of the exponent. If it fits in 32 bits then use that.
    int bitWidth =
        std::abs(exponent) < std::numeric_limits<int32_t>::max() ? 32 : 64;

    auto bcastIntType =
        TensorType::get(powOp.getType().getShapeAttr(),
                        rewriter.getIntegerType(bitWidth, true));
    auto newCst = createBroadcastedScalarConstantOpOfType(
        bcastIntType, int64_t(exponent), rewriter, powOp.getLoc());
    if (newCst.isError())
      return rewriter.notifyMatchFailure(powOp, newCst.getError());

    rewriter.modifyOpInPlace(powOp, [&]() { powOp.setRhsInput(*newCst); });

    return success();
  }
};

struct SymbolicOptimizationPass
    : public MO::impl::SymbolicOptimizationBase<SymbolicOptimizationPass> {
  using SymbolicOptimizationBase::SymbolicOptimizationBase;

  // Runs all default patterns plus the ones provided in the template
  // parameters.
  template <typename... PatternTypes>
  LogicalResult applyCorePatternsPlusExtraPatterns() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    if constexpr (sizeof...(PatternTypes))
      // Pattern that should execute now.
      patterns.add<PatternTypes...>(context);

    // Transpose swimming.
    patterns.add<SinkTransposeThroughUnaryOpPattern,
                 SinkTransposeThroughBinaryOpPattern,
                 SinkTransposeThroughStaticBroadcastToOpPattern>(context);

    // Reshape/Rebind insertion/swimming.
    patterns.add<
        ElementWiseUnaryVectorPattern, ElementWiseBinaryVectorPattern,
        BroadcastReshapePattern, SinkStaticReshapeOpThroughSoftmaxOp,
        SinkStaticReshapeLikeOpThroughBroadcastedElementwiseUnaryOp<
            Static::ReshapeOp>,
        SinkStaticReshapeLikeOpThroughBroadcastedElementwiseUnaryOp<RebindOp>,
        SinkStaticReshapeLikeOpThroughBroadcastedElementwiseBinaryOp<
            Static::ReshapeOp>,
        SinkStaticReshapeLikeOpThroughBroadcastedElementwiseBinaryOp<RebindOp>,
        SinkSameStaticReshapeLikeOpsThroughElementwiseBinaryOp<
            Static::ReshapeOp>,
        SinkSameStaticReshapeLikeOpsThroughElementwiseBinaryOp<RebindOp>,
        SinkSqueezeLikeReshapeThroughTranspose,
        SwimReshapeThroughBinaryOpWithConstantSubtree,
        SwimTransposeThroughBinaryOpWithConstantSubtree>(context);

    // Broadcast swimming.
    patterns.add<SinkBroadcastsThroughReduction,
                 CommuteBroadcastWithFlatennedArgNonzero,
                 SinkBroadcastsThroughElemwise<ElementWiseBinary>,
                 SinkBroadcastsThroughElemwise<ElementWiseUnary>,
                 SinkBroadcastsThroughLinalgBandPartOp>(context);

    // TODO(24646): Ensure these use device aware rewriter.
    TransposeOp::getCanonicalizationPatterns(patterns, context);
    Static::BroadcastToOp::getCanonicalizationPatterns(patterns, context);

    // Matmul rewrites.
    patterns.add<ConvertBatchMatmulToMatmulPattern,
                 ConvertBatchMatmulBatchSizeOneToMatmulPattern,
                 ConvToMatmulPattern, MergeParallelMatmulPattern<MatmulOp>,
                 MergeParallelMatmulPattern<BatchMatmulOp>>(context);

    // Split swimming.
    patterns.add<SinkSplitThroughStaticReshapeLikeOpsPattern<Static::ReshapeOp>,
                 SinkSplitThroughStaticReshapeLikeOpsPattern<RebindOp>,
                 SinkSplitThroughElementwisePattern>(context);

    // Concat rewriters.
    patterns.add<FlattenNestedConcatPattern,
                 SwimConcatThroughReshapeLikeOpsPattern<Static::ReshapeOp>,
                 SwimConcatThroughReshapeLikeOpsPattern<RebindOp>>(context);

    // Miscellaneous rewrites.
    patterns.add<GathersToSplitPattern, GeluPattern, SplitAndSqueezesPattern,
                 PadConvPattern>(context);

    // Elementwise optimizations.
    patterns
        .add<ArithmeticFoldingAddWithZero, ArithmeticFoldingSubWithZero,
             ArithmeticFoldingMulWithOne, ArithmeticFoldingDivByOne,
             ArithmeticFoldingPowerByOne, ArithmeticFoldingPowerOfOne,
             ArithmeticFoldingMaxSameOperand, ArithmeticFoldingMinSameOperand,
             ArithmeticFoldingEqualSameOperand, ArithmeticFoldingAndSameOperand,
             ArithmeticFoldingOrSameOperand, ArithmeticFoldingXorSameOperand,
             MathematicalCanonicalizationChainsOfPowToSinglePow,
             MathematicalCanonicalizationZeroSubToNegative,
             MathematicalSimplificationAddWithNegative,
             MathematicalSimplificationSubWithNegative>(context);
    if (symbolicFfastMath) {
      patterns.add<ArithmeticFoldingSubWithSame, ArithmeticFoldingMulWithZero,
                   ArithmeticFoldingPowByZero, ArithmeticFoldingMinOfMax,
                   ArithmeticFoldingMaxOfMin,
                   ArithmeticFoldingGreatherOrEqualThanSameOperand,
                   ArithmeticFoldingGreatherThanSameOperand,
                   ArithmeticFoldingNotEqualSameOperand,
                   MathematicalSimplificationDivByDivToDivOfMul,
                   MathematicalSimplificationDivOfDivToDivByMul,
                   MathematicalSimplificationDivBySmallTensorToMul,
                   MathematicalSimplificationMulOfDivToDiv,
                   MathematicalSimplificationPowByFloatToPowByInt>(context);
    }
    if (symbolicNoUndefinedMath) {
      patterns.add<ArithmeticFoldingZeroDiv, ArithmeticFoldingPowerOfZero,
                   ArithmeticFoldingModByOne, ArithmeticFoldingModSameOperand>(
          context);
    }

    // TODO(GRA-540): This is done to avoid potential subtle high-iteration
    // count loops in the pattern.
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    return applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                        config);
  }

  LogicalResult applyForwardMathematicalCanonicalizations() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<
        ForwardMathematicalCanonicalizationRsqrtToDivSqrt,
        ForwardMathematicalCanonicalizationReluToMax,
        ForwardMathematicalCanonicalizationMulByTwoToAdd,
        ForwardMathematicalCanonicalizationPowByTwoToMul,
        ForwardMathematicalCanonicalizationPowByNegativeOneOrTwoToDivAndPow>(
        context);

    // TODO(GRA-540): This is done to avoid potential subtle high-iteration
    // count loops in the pattern.
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    return applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                        config);
  }

  LogicalResult applyBackwardMathematicalCanonicalizations() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<
        BackwardMathematicalCanonicalizationDivSqrtToRsqrt,
        BackwardMathematicalCanonicalizationMaxToRelu,
        BackwardMathematicalCanonicalizationMulByTwoToAdd,
        BackwardMathematicalCanonicalizationDivAndPowToPowByNegativeOneOrTwo>(
        context);

    // TODO(GRA-540): This is done to avoid potential subtle high-iteration
    // count loops in the pattern.
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    return applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                        config);
  }

  LogicalResult applyCorePatterns() {
    // First run all the optimizations with slices turned into split when
    // possible.
    if (applyCorePatternsPlusExtraPatterns<SliceToSplitPattern>().failed())
      return failure();

    // Once all split operation patterns have run, convert them to slice and run
    // all of the other patterns too. Without it, split patterns would never
    // apply because they would be turned into slices too early.
    return applyCorePatternsPlusExtraPatterns<SplitToSlicePattern>();
  }

  void runOnOperation() override {
    if (applyCorePatterns().failed())
      return signalPassFailure();
    if (applyForwardMathematicalCanonicalizations().failed())
      return signalPassFailure();
    if (applyCorePatterns().failed())
      return signalPassFailure();
    if (applyBackwardMathematicalCanonicalizations().failed())
      return signalPassFailure();
    if (applyCorePatterns().failed())
      return signalPassFailure();
  }
}; // namespace

} // namespace
