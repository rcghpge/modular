//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "GenericML/GraphCompiler/MODialect/MODevices.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/GraphCompiler/Matchers.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "GenericML/Support/DataflowUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/ML/Permutation.h"

#include "./Utils.h"

#include <numeric>

#define DEBUG_TYPE "symbolic-optimization"

using namespace M;
using namespace MO;
using namespace mlir::matchers;
using mlir::m_Op;

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
    : OpRewritePattern<ArgNonzeroOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(MO::ArgNonzeroOp argNonzeroOp,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(argNonzeroOp, commonRewriter);
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
    auto lim = rewriter.create<KGEN::ParamConstantOp>(loc, limAttr);
    auto limT = rewriter.create<IndexToTensorOp>(loc, start.getType(), lim);

    // Finally, we use the flat size as the step size.
    auto step = rewriter.create<KGEN::ParamConstantOp>(loc, fsAttr);
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
    assert(oldDecls.size() == 1 &&
           "mo.argNonzero declares more than one new param");
    argNonzeroOp.setOutputParamDecls({});
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
    : public OpRewritePattern<BatchMatmulOp> {
  ConvertBatchMatmulToMatmulPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx, 4) {}

  LogicalResult
  matchAndRewrite(MO::BatchMatmulOp op,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);

    Value input, weight;
    Operation *broadcast = nullptr, *optRebindOp = nullptr;
    auto weightBcastP =
        m_Op<Static::BroadcastToOp>(&broadcast, m_MOMatrix(m_Any(&weight)));
    if (!m_Op<BatchMatmulOp>(
             m_Any(&input),
             m_Optional(m_Op<RebindOp>(&optRebindOp, weightBcastP),
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

    if (!rewriter.addOps({broadcast, optRebindOp})) {
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
    : public OpRewritePattern<BatchMatmulOp> {
  ConvertBatchMatmulBatchSizeOneToMatmulPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx, 5) {}

  LogicalResult
  matchAndRewrite(MO::BatchMatmulOp op,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);

    TensorType aType = op.getTensorA().getType();
    TensorType bType = op.getTensorB().getType();

    if (!aType.hasRank())
      return rewriter.notifyMatchFailure(
          op, "rank must be known in order to match");

    auto rank = aType.getRank();
    // Verifier ensures rank >= 3, so requiredOneDims >= 1
    auto requiredOneDims = rank - 2;

    for (size_t i = 0, e = (requiredOneDims); i < e; i++)
      if ((aType.getIntDim(i) != 1) || (aType.getIntDim(i) != 1))
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
    auto eltTy = aType.getElementType(); // elt types are all the same

    auto newAType =
        TensorType::get(MOSH::ShapeAttr::get(aShapeDims, shapeTy), eltTy);
    auto newBType =
        TensorType::get(MOSH::ShapeAttr::get(bShapeDims, shapeTy), eltTy);
    auto newResType =
        TensorType::get(MOSH::ShapeAttr::get(resShapeDims, shapeTy), eltTy);

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
struct ConvToMatmulPattern : public OpRewritePattern<ConvOp> {
  ConvToMatmulPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult
  matchAndRewrite(ConvOp op, PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);

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
struct FlattenNestedConcatPattern : public OpRewritePattern<ConcatOp> {
  FlattenNestedConcatPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult
  matchAndRewrite(ConcatOp op, PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);

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
        if (!rewriter.addOp(nestedConcatOp)) {
          return rewriter.notifyMatchFailure(
              op, "matched ops have inconsistent devices");
        }
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

    // Replace the old concat with a new concat that operate on the flattened
    // input values. Note that the types are guaranteed to match (shape
    // inference either verified it statically, or dropped runtime asserts).
    rewriter.replaceOpWithNewOp<ConcatOp>(op, op.getType(), op.getAxis(),
                                          flattenedInputs);
    return mlir::success();
  }
};

/// A pattern that detects a power op whose exponent is a constant floating
/// point 2.0, and rewrites it into a single multiplication.
///
/// Example:
///
/// input : tensor<[...]>
///
///   res = pow(input, 2.0)
/// ===>
///   res = input * input
struct Pow2Pattern : public OpRewritePattern<PowOp> {
  Pow2Pattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult
  matchAndRewrite(MO::PowOp op,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);
    Value input;
    Operation *broadcastOp = nullptr;
    if (!m_Op<PowOp>(m_Any(&input), m_Op<Static::BroadcastToOp>(
                                        &broadcastOp, m_ScalarConstant(2.0)))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input)
      return rewriter.notifyMatchFailure(op, "null input value");

    if (!rewriter.addOp(broadcastOp)) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    rewriter.replaceOpWithNewOp<MulOp>(op, input, input);

    return mlir::success();
  }
};

/// A pattern that detects a power op whose exponent is a constant floating
/// point 3.0, and rewrites it into two multiplications.
///
/// Example:
///
/// input : tensor<[...]>
///
///   res = pow(input, 3.0)
/// ===>
///   res = input * input * input
struct Pow3Pattern : public OpRewritePattern<PowOp> {
  Pow3Pattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult
  matchAndRewrite(MO::PowOp op,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);
    Value input;
    Operation *broadcastOp = nullptr;
    if (!m_Op<PowOp>(m_Any(&input), m_Op<Static::BroadcastToOp>(
                                        &broadcastOp, m_ScalarConstant(3.0)))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input)
      return rewriter.notifyMatchFailure(op, "null input value");

    if (!rewriter.addOp(broadcastOp)) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    Value sq = rewriter.create<MulOp>(op->getLoc(), input, input).getResult();
    rewriter.replaceOpWithNewOp<MulOp>(op, sq, input);

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
struct GathersToSplitPattern : public OpRewritePattern<GatherOp> {
  GathersToSplitPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult
  matchAndRewrite(GatherOp op, PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);

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

      // Device annotations must be consistent.
      if (!rewriter.addOp(userGatherOp))
        continue;

      // Update dominant gather and make sure all gathers are in the same block
      if (!dominantGatherOp)
        dominantGatherOp = userGatherOp;

      // TODO account for control-flow
      if (userGatherOp->getBlock() != dominantGatherOp->getBlock())
        continue;

      if (userGatherOp->isBeforeInBlock(dominantGatherOp))
        dominantGatherOp = userGatherOp;

      idxToGather.try_emplace(idx, userGatherOp);
    }

    // We must have input gather users that cover the entire 0-axis
    if (idxToGather.size() != inputAxis0Dim) {
      return rewriter.notifyMatchFailure(
          op, "not enough qualified gather users for input");
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
struct GeluPattern : public OpRewritePattern<MulOp> {
  GeluPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 4) {}

  LogicalResult
  matchAndRewrite(MO::MulOp op,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);

    Value input, input_erf;
    Operation *sqrt2Broadcast = nullptr, *invSqrt2Broadcast = nullptr,
              *erfOp = nullptr, *divOp = nullptr, *mulOp = nullptr,
              *oneBroadcastOp = nullptr, *addOp = nullptr,
              *halfBroadcastOp = nullptr, *mul1Op = nullptr, *mul2Op = nullptr;
    // gelu = x * 1/2 * (1 + erf(x/sqrt(2)))
    auto sq2 = m_Op<Static::BroadcastToOp>(
        &sqrt2Broadcast, m_ScalarConstant(llvm::numbers::sqrt2f));
    auto divsq2 = m_Op<Static::BroadcastToOp>(
        &invSqrt2Broadcast, m_ScalarConstant(llvm::numbers::inv_sqrt2f));

    auto x_erf = m_Any(&input_erf);
    auto erf = m_Op<ErfOp>(&erfOp, m_Optional(m_Op<DivOp>(&divOp, x_erf, sq2),
                                              m_BinaryCommutativeOp<MulOp>(
                                                  &mulOp, x_erf, divsq2)));
    auto one =
        m_Op<Static::BroadcastToOp>(&oneBroadcastOp, m_ScalarConstant(1.0f));
    auto add = m_BinaryCommutativeOp<AddOp>(&addOp, one, erf);

    auto half =
        m_Op<Static::BroadcastToOp>(&halfBroadcastOp, m_ScalarConstant(0.5f));
    auto x = m_Any(&input);
    // TODO: write an associative commutative matcher?
    auto gelu = m_Optional(
        m_BinaryCommutativeOp<MulOp>(
            &mul1Op, m_BinaryCommutativeOp<MulOp>(&mul2Op, x, half), add),
        m_BinaryCommutativeOp<MulOp>(
            &mul1Op, half, m_BinaryCommutativeOp<MulOp>(&mul2Op, x, add)));

    if (!gelu.match(op->getResult(0)))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !input_erf)
      return rewriter.notifyMatchFailure(op, "null input value");
    if (!(input == input_erf))
      return rewriter.notifyMatchFailure(op, "inputs don't match");

    if (!rewriter.addOps({sqrt2Broadcast, invSqrt2Broadcast, erfOp, divOp,
                          mulOp, oneBroadcastOp, addOp, halfBroadcastOp, mul1Op,
                          mul2Op})) {
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
    : public mlir::OpInterfaceRewritePattern<ElementWiseUnary> {
  SinkTransposeThroughUnaryOpPattern(MLIRContext *ctx)
      : OpInterfaceRewritePattern(ctx, 2) {}

  LogicalResult
  matchAndRewrite(ElementWiseUnary op,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);
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

    if (!rewriter.addOp(opaqueTransposeOp)) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    TensorType finalOutputType = op.getResult().getType();
    auto transposeOp = cast<TransposeOp>(opaqueTransposeOp);
    TensorValue input = transposeOp.getInput();
    TensorValue opRes = op.getResult();
    rewriter.replaceAllUsesWith(op->getResults(), transposeOp.getResult());
    op.setInput(input);

    DType unaryResultDType = opRes.getType().getDtype();
    opRes.setType(input.getType().cloneWith(unaryResultDType));
    transposeOp.getInputMutable().assign(opRes);
    transposeOp->moveAfter(op);
    transposeOp.getResult().setType(finalOutputType);

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
    : public mlir::OpInterfaceRewritePattern<ElementWiseBinary> {
  SinkTransposeThroughBinaryOpPattern(MLIRContext *ctx)
      : OpInterfaceRewritePattern(ctx, 2) {}

  LogicalResult
  matchAndRewrite(ElementWiseBinary op,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);
    TensorValue input1, input2, perm1, perm2;
    Operation *leftTransposeOp = nullptr, *rightTransposeOp = nullptr;
    auto transpose1P =
        m_Op<TransposeOp>(&leftTransposeOp, m_Any(&input1), m_Any(&perm1));
    auto transpose2P =
        m_Op<TransposeOp>(&rightTransposeOp, m_Any(&input2), m_Any(&perm2));
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

    if (!rewriter.addOps({leftTransposeOp, rightTransposeOp})) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    TensorValue opRes = op.getResult();
    op.setLhsInput(input1);
    op.setRhsInput(input2);
    TensorType oldOpResTy = opRes.getType();

    opRes.setType(input1Ty.cloneWith(oldOpResTy.getDtype()));
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
    : public OpRewritePattern<Static::BroadcastToOp> {
  SinkTransposeThroughStaticBroadcastToOpPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx, 2) {}

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
  matchAndRewrite(Static::BroadcastToOp op,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);
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

    if (!rewriter.addOp(opaqueTransposeOp)) {
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
      transposeOp.getPermMutable().assign(newPerm);
    }
    SmallVector<TypedAttr> newBroadcastDims =
        unpermute(oldBroadcastResTy.getDims(), permVec);

    auto shapeTy = rewriter.getType<ShapeType>();
    MO::TensorType newBroadcastResTy =
        oldBroadcastResTy.cloneWith(ShapeAttr::get(newBroadcastDims, shapeTy));

    rewriter.replaceAllUsesWith(op, transposeOp.getResult());
    op.getInputMutable().assign(input);
    op.getResult().setType(newBroadcastResTy);
    transposeOp.getInputMutable().assign(op.getResult());
    transposeOp->moveAfter(op);
    transposeOp.getResult().setType(oldBroadcastResTy);

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
    : public mlir::OpInterfaceRewritePattern<ElementWiseBinary> {
  SinkStaticReshapeLikeOpThroughBroadcastedElementwiseBinaryOp(MLIRContext *ctx)
      : OpInterfaceRewritePattern(ctx, 2) {}

  LogicalResult
  matchAndRewrite(ElementWiseBinary op,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);

    // Match pattern and get results.
    TensorValue inputTensor, elemwiseTensor;
    Operation *reshapeOp = nullptr, *broadcastOp = nullptr,
              *elementwiseOp1 = nullptr, *elementwiseOp2 = nullptr;
    auto reshapeP = m_Op<ReshapeOpType>(&reshapeOp, m_Any(&inputTensor));
    auto broadcastP =
        m_Op<Static::BroadcastToOp>(&broadcastOp, m_Any(&elemwiseTensor));
    auto elemwiseOpP1 =
        m_Op<ElementWiseBinary>(&elementwiseOp1, reshapeP, broadcastP);
    auto elemwiseOpP2 =
        m_Op<ElementWiseBinary>(&elementwiseOp2, broadcastP, reshapeP);

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

    if (!rewriter.addOps(
            {reshapeOp, broadcastOp, elementwiseOp1, elementwiseOp2})) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
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
    : public mlir::OpInterfaceRewritePattern<ElementWiseUnary> {
  SinkStaticReshapeLikeOpThroughBroadcastedElementwiseUnaryOp(MLIRContext *ctx)
      : OpInterfaceRewritePattern(ctx, 2) {}

  LogicalResult
  matchAndRewrite(ElementWiseUnary op,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);
    TensorValue inputTensor;
    Operation *reshapeOp = nullptr;
    auto reshapeP = m_Op<ReshapeOpType>(&reshapeOp, m_Any(&inputTensor));
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

    auto inputType = inputTensor.getType().cast<TensorType>();
    // This avoids creating a unary op with unknown shapes.
    if (inputType.hasUnknownShape())
      return rewriter.notifyMatchFailure(op, "Shape must be known");

    if (!rewriter.addOp(reshapeOp)) {
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
    : public mlir::OpInterfaceRewritePattern<ElementWiseBinary> {
  SinkSameStaticReshapeLikeOpsThroughElementwiseBinaryOp(MLIRContext *ctx)
      : OpInterfaceRewritePattern(ctx, 2) {}

  LogicalResult
  matchAndRewrite(ElementWiseBinary op,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);
    TensorValue input1Tensor, input2Tensor;
    Operation *reshapeOp1 = nullptr, *reshapeOp2 = nullptr;
    auto reshapeP1 = m_Op<ReshapeOpType>(&reshapeOp1, m_Any(&input1Tensor));
    auto reshapeP2 = m_Op<ReshapeOpType>(&reshapeOp2, m_Any(&input2Tensor));
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

    if (!rewriter.addOps({reshapeOp1, reshapeOp2})) {
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
struct SplitAndSqueezesPattern : public OpRewritePattern<SplitOp> {
  // TODO(#13349) Generalize or remove this pattern
  SplitAndSqueezesPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  // Check whether `op` is a `mo.static.reshape` (or rebind followed by a
  // reshape) that effectively squeezes the given axis `squeezeAxis`.
  static bool isSqueezingReshape(DeviceAwareRewriter &rewriter, Operation *op,
                                 ArrayRef<TypedAttr> inferredInDims,
                                 size_t squeezeAxis) {
    if (!rewriter.addOp(op))
      return false;

    // TODO(#13348) remove this check once we have shape inference for SplitOp
    if (auto rebindOp = dyn_cast<RebindOp>(op)) {
      if (!rebindOp->hasOneUse())
        return false;
      op = *rebindOp->user_begin();
      if (!rewriter.addOp(op))
        return false;
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

  /// Construct a new TensorType based on `ty` with the shape dimensions
  /// permuted by `perm`.
  static TensorType permuteDims(TensorType ty, ArrayRef<int64_t> perm) {
    SmallVector<TypedAttr> permutedDims =
        permute(ty.getTypedShapeAttr().getValues(), perm);
    auto shapeTy = ShapeType::get(ty.getContext());
    return ty.cloneWith(ShapeAttr::get(permutedDims, shapeTy));
  }

  LogicalResult
  matchAndRewrite(SplitOp op, PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);
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

    // Construct transpose permutation and zero axis value (before split op)
    Location loc = op.getLoc();
    SmallVector<int64_t> permVec(inputRank);
    std::iota(permVec.begin(), permVec.end(), 0);
    std::swap(permVec[0], permVec[splitAxis]);
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
      assert(res.hasOneUse() && "guards should've enforced this");
      if (auto rebindOp = dyn_cast<RebindOp>(*res.user_begin())) {
        TensorType oldResTy = rebindOp.getResult().getType();
        TensorType newResTy = permuteDims(oldResTy, permVec);
        rebindOp.getResult().setType(newResTy);
      }
    }

    return success();
  }
};

struct ElementWiseUnaryVectorPattern
    : public mlir::OpInterfaceRewritePattern<ElementWiseUnary> {
  ElementWiseUnaryVectorPattern(MLIRContext *ctx)
      : OpInterfaceRewritePattern(ctx, 2) {}

  LogicalResult
  matchAndRewrite(MO::ElementWiseUnary op,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);
    Location loc = op.getLoc();
    TensorValue input = op.getInput();

    auto inType = input.getType().cast<TensorType>();

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
    : public mlir::OpInterfaceRewritePattern<ElementWiseBinary> {
  ElementWiseBinaryVectorPattern(MLIRContext *ctx)
      : OpInterfaceRewritePattern(ctx, 2) {}

  LogicalResult
  matchAndRewrite(MO::ElementWiseBinary op,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);
    Location loc = op.getLoc();
    TensorValue a = op.getLhsInput();
    TensorValue b = op.getRhsInput();

    auto inType = a.getType().cast<TensorType>();

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
    op.setLhsInput(reshapeA.getResult());
    op.setRhsInput(reshapeB.getResult());
    op.getResult().setType(newTypeShape.cloneWith(oldTensorType.getDtype()));
    rewriter.setInsertionPointAfter(op);
    auto out =
        rewriter.create<Static::ReshapeOp>(loc, oldTensorType, op.getResult());
    rewriter.replaceAllUsesExcept(opRes, out.getResult(), out);
    return mlir::success();
  }
};

struct BroadcastReshapePattern : public OpRewritePattern<Static::ReshapeOp> {
  BroadcastReshapePattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult
  matchAndRewrite(Static::ReshapeOp op,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);
    Value input;
    Operation *broadcastOp = nullptr;
    if (!m_Op<Static::ReshapeOp>(
             m_Op<Static::BroadcastToOp>(&broadcastOp, m_Any(&input)))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input)
      return rewriter.notifyMatchFailure(op, "null input value");

    TensorType inType = input.getType().cast<TensorType>();

    // Check if the input is a scalar
    if (!inType.hasRank())
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    for (size_t i = 0, e = inType.getRank(); i < e; ++i)
      if (inType.getIntDim(i) != 1)
        return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!rewriter.addOp(broadcastOp)) {
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

/// A pattern that detects a matmul with a vector weight and a constant
/// input, and transposes it.
struct GemvPattern : public OpRewritePattern<MatmulOp> {
  GemvPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult
  matchAndRewrite(MO::MatmulOp op,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);
    Location loc = op.getLoc();
    Value a = op.getTensorA();
    Value b = op.getTensorB();
    auto aType = b.getType().cast<TensorType>();
    auto bType = b.getType().cast<TensorType>();

    if (!(a.getDefiningOp<MO::ConstantOp>() && bType.hasStaticShape() &&
          bType.getIntDim(1) == 1 && op.getType().hasStaticShape() &&
          aType.hasStaticShape()))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    TensorValue perm =
        rewriter.create<ConstantOp>(loc, SmallVector<int64_t>{1, 0});

    auto transposeType = [&](Value x) {
      auto xType = x.getType().cast<TensorType>();
      return TensorType::get(
          SmallVector<int64_t>{xType.getIntDim(1), xType.getIntDim(0)},
          xType.getElementType());
    };
    auto transpose = [&](Value x) {
      auto xTransposeTy = transposeType(x);
      return rewriter.create<TransposeOp>(loc, xTransposeTy, x, perm);
    };

    auto matmul = rewriter.create<MatmulOp>(loc, transposeType(op),
                                            transpose(b), transpose(a));
    rewriter.replaceOp(op, transpose(matmul));
    return mlir::success();
  }
};

/// A pattern that replaces a split with several slices to enable fusion. This
/// is the converse of SliceToSplitPattern.
struct SplitToSlicePattern : public OpRewritePattern<SplitOp> {
  SplitToSlicePattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult
  matchAndRewrite(MO::SplitOp op,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);
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
    if (axis < 0)
      axis = inTy.getRank() + axis;

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
struct SliceToSplitPattern : public OpRewritePattern<SliceOp> {
  SliceToSplitPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::SliceOp op,
                                PatternRewriter &rewriter) const override {
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

    auto inputType = input.getType().cast<TensorType>();
    int64_t sliceDimension;
    if (!llvm::all_of(sliceOps, [&](SliceOp sliceOp) {
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
        })) {
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
      return SmallVector<APInt>(value.getDefiningOp<ConstantOp>()
                                    .getValue()
                                    .cast<IntArrayElementsAttr>()
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
struct MergeParallelMatmulPattern : public OpRewritePattern<MatmulOpType> {
  MergeParallelMatmulPattern(MLIRContext *ctx)
      : OpRewritePattern<MatmulOpType>(ctx, 2) {}

  LogicalResult
  matchAndRewrite(MatmulOpType op,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);
    // Gather all matmul operations that use the same activations.
    Value activations = op.getInputA();
    SmallVector<MatmulOpType> matmulUsers;
    for (Operation *user : activations.getUsers()) {
      if (auto matmulOp = dyn_cast<MatmulOpType>(user)) {
        if (!rewriter.addOp(matmulOp)) {
          return rewriter.notifyMatchFailure(
              op, "matched ops have inconsistent devices");
        }
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
    auto oldMatmulType = op.getType().template cast<TensorType>();
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

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);

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
        op.getInputB().getType().template cast<TensorType>().cloneWithNewDim(
            oldMatmulType.getRank() - 1, totalColumns);
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

static bool splitWithStaticRankLastDimensionFullUse(SplitOp op) {
  auto inputType = op.getInput().getType().cast<TensorType>();
  if (!inputType.hasRank())
    return false;

  auto axisOr = tryGetConstantInt(op.getAxis());
  if (!axisOr)
    return false;

  uint64_t lastAxis = inputType.getRank() - 1;
  uint64_t axis = *axisOr;
  if (axis != lastAxis)
    return false;

  if (op.getNumResults() != llvm::range_size(op->getUsers()) ||
      op->getNumResults() != llvm::range_size(op->getUses()))
    return false;

  return true;
}

/// A pattern that will move a split/ operation after identical elementwise
/// operations if certain conditions are met
/// CONDITIONS:
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
struct SinkSplitThroughElementwisePattern : public OpRewritePattern<SplitOp> {
  SinkSplitThroughElementwisePattern(MLIRContext *ctx)
      : OpRewritePattern(ctx, 2) {}

  LogicalResult
  matchAndRewrite(SplitOp op, PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);
    if (!splitWithStaticRankLastDimensionFullUse(op)) {
      return rewriter.notifyMatchFailure(
          op, "Split op must be statically ranked, operate on last axis and "
              "have one use and one user per result");
    }

    auto firstResultType = op.getResults().front().getType().cast<TensorType>();
    SmallVector<Value> elementwiseValues;
    for (auto result : op.getResults()) {
      if (result.getType() != firstResultType) {
        return rewriter.notifyMatchFailure(
            op, "Split results don't have the same size");
      }

      Operation *user = result.getUses().begin()->getOwner();
      if (!isa<ElementWiseUnary, ElementWiseBinary>(user)) {
        return rewriter.notifyMatchFailure(
            op, "Split result use is not an elementwise operation");
      }
      if (!rewriter.addOp(user)) {
        return rewriter.notifyMatchFailure(
            op, "matched ops have inconsistent devices");
      }

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
        auto constantOpType = constantOp.getType().cast<TensorType>();
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

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    auto inputType = op.getInput().getType().cast<TensorType>();
    uint64_t axis = *tryGetConstantInt(op.getAxis());
    uint64_t lastDimSize =
        inputType.getDims().back().cast<IntegerAttr>().getInt();

    SmallVector<Value, 2> elementwiseOperands;
    if (!constantsToConcatenate.empty()) {
      TensorType constantType =
          constantsToConcatenate.front().getType().cast<TensorType>();
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
            broadcastOp.getType().cloneWithNewDim(axis, lastDimSize);
        broadcast = rewriter
                        .create<Static::BroadcastToOp>(loc, broadcastType,
                                                       concatenatedConstants)
                        .getResult();
      }

      elementwiseOperands.resize(2);
      elementwiseOperands[firstUseIndex] = broadcast;
      elementwiseOperands[1 - firstUseIndex] = op.getInput();
    } else {
      elementwiseOperands.push_back(op.getInput());
    }

    auto elementwiseType =
        firstOwner->getResultTypes().front().cast<TensorType>().cloneWithNewDim(
            axis, lastDimSize);
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

/// A pattern that will sink a split operation through a reshape-like operation
/// such as Static::ReshapeOp and RebindOp but that do not temper with the split
/// dimension.
template <typename ReshapeOpType>
struct SinkSplitThroughStaticReshapeLikeOpsPattern
    : public OpRewritePattern<SplitOp> {
  SinkSplitThroughStaticReshapeLikeOpsPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx, 2) {}

  LogicalResult
  matchAndRewrite(SplitOp op, PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);
    if (!splitWithStaticRankLastDimensionFullUse(op)) {
      return rewriter.notifyMatchFailure(
          op, "Split op must be statically ranked, operate on last axis and "
              "have one use and one user per result");
    }

    auto users = op->getUsers();
    if (!llvm::all_of(users, [](Operation *user) {
          return llvm::isa<ReshapeOpType>(user);
        })) {
      return rewriter.notifyMatchFailure(op, "Users are not reshape-like");
    }

    auto firstReshapeOutputType =
        users.begin()->getResultTypes().front().cast<TensorType>();
    auto firstReshapeOutputShape = firstReshapeOutputType.getDims();
    if (!llvm::all_of(users, [&](Operation *reshape) {
          auto otherReshapeOutputShape =
              reshape->getResultTypes().front().cast<TensorType>().getDims();
          if (otherReshapeOutputShape.empty()) {
            return false;
          }

          return firstReshapeOutputShape.drop_back() ==
                 otherReshapeOutputShape.drop_back();
        })) {
      return rewriter.notifyMatchFailure(
          op, "Reshapes are not identical on other dimensions or they have "
              "scalar inputs or outputs");
    }

    if (llvm::any_of(llvm::zip(op.getResults(), users), [&](auto result_user) {
          auto [result, user] = result_user;
          return user->getResult(0)
                     .getType()
                     .template cast<TensorType>()
                     .getDims()
                     .back() !=
                 result.getType().template cast<TensorType>().getDims().back();
        })) {
      return rewriter.notifyMatchFailure(
          op, "Reshape is tempering with the split dimension");
    }

    for (Operation *user : users) {
      if (!rewriter.addOp(user)) {
        return rewriter.notifyMatchFailure(
            op, "matched ops have inconsistent devices");
      }
    }

    Location loc = op->getLoc();
    rewriter.setInsertionPoint(op);

    auto inputType = op.getInput().getType().cast<TensorType>();
    auto newReshapeType = firstReshapeOutputType.cloneWithNewDim(
        firstReshapeOutputType.getRank() - 1,
        inputType.getDims().back().cast<IntegerAttr>().getInt());
    auto newReshape =
        rewriter.create<ReshapeOpType>(loc, newReshapeType, op.getInput())
            .getResult();

    SmallVector<Type> newSplitTypes;
    for (Operation *user : users)
      newSplitTypes.push_back(user->getResult(0).getType());
    SmallVector<Value> newSplitOpOperands = op.getOperands();
    newSplitOpOperands[0] = newReshape;
    newSplitOpOperands[2] =
        rewriter
            .create<ConstantOp>(loc,
                                int64_t(firstReshapeOutputType.getRank() - 1))
            .getResult();
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

/// This pattern will sink a split operation through a static reshape operation
/// that actually tempers with the split dimension. This is a specialized
/// pattern that rarely occurs but it provides performance improvement to small
/// BERT models.
struct SinkSplitThroughStaticReshapeOpPattern
    : public OpRewritePattern<SplitOp> {
  SinkSplitThroughStaticReshapeOpPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(SplitOp op,
                                PatternRewriter &rewriter) const override {
    if (!splitWithStaticRankLastDimensionFullUse(op)) {
      return rewriter.notifyMatchFailure(
          op, "Split op must be statically ranked, operate on last axis and "
              "have one use and one user per result");
    }

    // This requirement is not absolutely needed but this pattern is already so
    // specialized, it is unlikely that it will be applied somewhere else.
    auto resultTypes = op->getResultTypes();
    if (!llvm::all_of(resultTypes, [&](Type resultType) {
          return resultType == resultTypes.front();
        })) {
      return rewriter.notifyMatchFailure(op, "Split is not uniform");
    }

    auto users = op->getUsers();
    if (!llvm::all_of(users, [&](Operation *user) {
          auto reshapeOp = llvm::dyn_cast<Static::ReshapeOp>(user);
          if (!reshapeOp)
            return false;

          auto type = reshapeOp.getType();

          return type.getRank() &&
                 type == cast<Static::ReshapeOp>(*users.begin()).getType();
        })) {
      return rewriter.notifyMatchFailure(
          op, "Users must be identical non-scalar reshapes");
    }

    auto shapeBeforeReshape = op.getType(0).cast<TensorType>().getDims();
    auto shapeAfterReshape = users.begin()
                                 ->getResults()
                                 .front()
                                 .getType()
                                 .cast<TensorType>()
                                 .getDims();

    if (shapeAfterReshape.size() < shapeBeforeReshape.size()) {
      return rewriter.notifyMatchFailure(
          op, "Reshape must be expanding dimensions");
    }

    // Note: There is a massive assumption that the data is row-major. If it
    // isn't, this breaks.
    uint64_t firstDimensionThatMismatches;
    for (firstDimensionThatMismatches = 0;
         firstDimensionThatMismatches < shapeBeforeReshape.size();
         ++firstDimensionThatMismatches) {
      if (shapeBeforeReshape[firstDimensionThatMismatches] !=
          shapeAfterReshape[firstDimensionThatMismatches])
        break;
    }

    // No need to check for static dimensions in the reshaped slice of
    // dimensions because the MO::Static::ReshapeOp already enforces it.

    Location loc = op->getLoc();
    rewriter.setInsertionPoint(op);

    auto firstReshapeType = cast<Static::ReshapeOp>(*users.begin()).getType();
    SmallVector<TypedAttr> newReshapeShape(firstReshapeType.getDims());
    newReshapeShape[firstDimensionThatMismatches] =
        rewriter.getIndexAttr(newReshapeShape[firstDimensionThatMismatches]
                                  .cast<IntegerAttr>()
                                  .getInt() *
                              op->getNumResults());
    auto newReshapeType = TensorType::get(
        ShapeAttr::get(newReshapeShape, rewriter.getType<MOSH::ShapeType>()),
        firstReshapeType.getDtype());
    auto newReshape =
        rewriter.create<Static::ReshapeOp>(loc, newReshapeType, op.getInput())
            .getResult();

    SmallVector<Type> newSplitTypes(op.getNumResults(), firstReshapeType);
    SmallVector<int64_t> newSplitSizes(
        op.getNumResults(),
        firstReshapeType.getIntDim(firstDimensionThatMismatches));
    SmallVector<Value> newSplitOpOperands = op.getOperands();
    newSplitOpOperands[0] = newReshape;
    newSplitOpOperands[1] = rewriter.create<ConstantOp>(loc, newSplitSizes);
    newSplitOpOperands[2] =
        rewriter.create<ConstantOp>(loc, int64_t(firstDimensionThatMismatches))
            .getResult();
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
/// This can only be done in certain scenarios generally, most notably as
/// long as the axis of reduction is done on a dimension which is not modified
/// in the broadcast.
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
struct SinkBroadcastsThroughReduction
    : public OpInterfaceRewritePattern<Reduction> {
  SinkBroadcastsThroughReduction(MLIRContext *ctx)
      : OpInterfaceRewritePattern(ctx, 2) {}

  LogicalResult
  matchAndRewrite(Reduction op,
                  PatternRewriter &commonRewriter) const override {
    DeviceAwareRewriter rewriter(op, commonRewriter);
    TensorValue input = op.getInput();
    Location loc = op->getLoc();

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
    int64_t axis = maybeAxis.value();

    // respect negative axis rules
    int64_t initialRank = input.getType().getRank();
    if (axis < -initialRank || axis >= initialRank)
      return rewriter.notifyMatchFailure(
          loc, "Axis is not within [-rank, rank - 1]");

    if (axis < 0)
      axis += input.getType().getRank();

    // Here we want to map the axes we are reducing on in the broadcasted tensor
    // to on the non-broadcasted input.
    //
    // E.g. if we broadcasted [A, B, 1] --> [1, D, A, B, 1]
    //
    // And we are reducing along the dim with 'A', then we just need to count
    // the number of additional broadcasted dimensions before the given axis.
    // In this case it is 2, since we added axis `1` and `D`.
    ErrorOr<SmallVector<int64_t>> maybeBcastedAxis =
        ShapeAttr::getBroadcastedAxes(
            bcastOp.getInput().getType().getTypedShapeAttr(),
            bcastOp.getType().getTypedShapeAttr());
    if (maybeBcastedAxis.isError())
      return rewriter.notifyMatchFailure(loc, maybeBcastedAxis.getError());

    SmallVector<int64_t> &bcastedAxis = maybeBcastedAxis.get();
    size_t numBeforeAxis = 0;
    for (int64_t bcastedDimIndex : bcastedAxis) {
      // TODO(24589): there are fancier things we can simplify based on the type
      // of reduction here when we reduce along the broadcasted axis.
      //
      // For example, if we are doing reduce.add in this case, we just have to
      // multiply the old tensor by the size of the broadcast axis and reshape.
      //
      // For max/min, the result is invariant to these transforms.
      // For prod, it becomes pow.
      if (bcastedDimIndex == axis)
        return rewriter.notifyMatchFailure(
            loc,
            "Cannot optimize if reducing along axes modified by broadcast");
      if (bcastedDimIndex > axis)
        break;
      numBeforeAxis++;
    }

    // Now we know we are reducing along a non-broadcasted axis, we can elide
    // the broadcast for later.
    int64_t newAxis = axis - numBeforeAxis;
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

struct SymbolicOptimizationPass
    : public MO::impl::SymbolicOptimizationBase<SymbolicOptimizationPass> {
  using SymbolicOptimizationBase::SymbolicOptimizationBase;

  // Runs all default patterns plus the ones provided in the template
  // parameters.
  template <typename... PatternTypes>
  void optimizeWithPattern() {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    if constexpr (sizeof...(PatternTypes))
      // Pattern that should execute now.
      patterns.add<PatternTypes...>(context);

    // Transpose swimming.
    patterns.add<SinkTransposeThroughUnaryOpPattern,
                 SinkTransposeThroughBinaryOpPattern,
                 SinkTransposeThroughStaticBroadcastToOpPattern>(context);
    TransposeOp::getCanonicalizationPatterns(patterns, context);

    // Reshape/Rebind insertion/swimming.
    patterns.add<
        ElementWiseUnaryVectorPattern, ElementWiseBinaryVectorPattern,
        BroadcastReshapePattern,
        SinkStaticReshapeLikeOpThroughBroadcastedElementwiseUnaryOp<
            Static::ReshapeOp>,
        SinkStaticReshapeLikeOpThroughBroadcastedElementwiseUnaryOp<RebindOp>,
        SinkStaticReshapeLikeOpThroughBroadcastedElementwiseBinaryOp<
            Static::ReshapeOp>,
        SinkStaticReshapeLikeOpThroughBroadcastedElementwiseBinaryOp<RebindOp>,
        SinkSameStaticReshapeLikeOpsThroughElementwiseBinaryOp<
            Static::ReshapeOp>,
        SinkSameStaticReshapeLikeOpsThroughElementwiseBinaryOp<RebindOp>>(
        context);

    // Broadcast swimming.
    patterns.add<CommuteBroadcastWithFlatennedArgNonzero,
                 SinkBroadcastsThroughReduction>(context);
    Static::BroadcastToOp::getCanonicalizationPatterns(patterns, context);

    // Matmul rewrites.
    patterns.add<ConvertBatchMatmulToMatmulPattern,
                 ConvertBatchMatmulBatchSizeOneToMatmulPattern,
                 ConvToMatmulPattern, MergeParallelMatmulPattern<MatmulOp>,
                 MergeParallelMatmulPattern<BatchMatmulOp>>(context);

    // Split swimming.
    patterns.add<SinkSplitThroughStaticReshapeLikeOpsPattern<Static::ReshapeOp>,
                 SinkSplitThroughStaticReshapeLikeOpsPattern<RebindOp>,
                 SinkSplitThroughElementwisePattern,
                 SinkSplitThroughStaticReshapeOpPattern>(context);

    // Miscellaneous rewrites.
    patterns.add<FlattenNestedConcatPattern, Pow2Pattern, Pow3Pattern,
                 GathersToSplitPattern, GeluPattern, GemvPattern,
                 SplitAndSqueezesPattern>(context);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
  void runOnOperation() override {
    GraphOp graphOp = getOperation();

    LLVM_DEBUG({
      llvm::dbgs() << "-------- before SymbolicOptimization ----------\n";
      printGraph(llvm::dbgs(), graphOp);
      llvm::dbgs() << "--------------------------------------------\n";
    });

    // First run all the optimizations with slices turned into split when
    // possible.
    optimizeWithPattern<SliceToSplitPattern>();
    // Once all split operation patterns have run, convert them to slice and run
    // all of the other patterns too. Without it, split patterns would never
    // apply because they would be turned into slices too early.
    optimizeWithPattern<SplitToSlicePattern>();

    LLVM_DEBUG({
      llvm::dbgs() << "-------- after SymbolicOptimization ----------\n";
      printGraph(llvm::dbgs(), graphOp);
      llvm::dbgs() << "--------------------------------------------\n";
    });
  }
};

} // namespace
