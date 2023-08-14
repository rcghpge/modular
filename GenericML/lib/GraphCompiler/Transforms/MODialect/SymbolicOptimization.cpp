//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/GraphCompiler/Matchers.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "Support/ML/Permutation.h"

#include <numeric>

using namespace M;
using namespace MO;
using namespace mlir::matchers;
using mlir::m_Op;

namespace M::MO {
#define GEN_PASS_DEF_SYMBOLICOPTIMIZATION
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

} // namespace M::MO

namespace {

/// A pattern that detects `mo.where` ops preceded by a flattening reshape and a
/// rank-increasing broadcast, and rewrites them so that the broadcast happens
/// after the `mo.where` op.
///
/// This decreases the workload on `mo.where` and improves the compilers ability
/// to constant fold before and fuse after the where op.
///
/// Example:
///
/// input : tensor<[2, 3]>
///
///   bcast = broadcast(input) : ... -> tensor<[5, 2, 3]>
///   flat  = reshape(bcast)   : ... -> tensor<[30]>
///   res   = where(flat)      : ...
/// ===>
///   flat  = reshape(input)   : ... -> tensor<[6]>
///   where = where(flat)      : ...
///   res   = where + broadcast(range(...)))
struct CommuteBroadcastWithFlatennedWhere : OpRewritePattern<WhereOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MO::WhereOp whereOp,
                                PatternRewriter &rewriter) const override {
    Location loc = whereOp.getLoc();

    TensorValue inputVal = whereOp.getInput();
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

    // The pattern is now matched. We start by flattening the input of the
    // broadcast, i.e. we roughly implement this:
    // flattened = bcastInput.flatten()
    // NOTE: FS/fs stands for flat size.
    GraphOp graphOp = whereOp->getParentOfType<GraphOp>();
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
    // where_res = np.expand_dims(np.where(flattened)[0], axis=1)
    // Note we refer to the number of unique elements as W.
    KGEN::ParamDeclAttr wDecl = graphOp.getNewDimParam("W");
    auto wRef = KGEN::ParamDeclRefAttr::get(wDecl);
    IntegerType si64Ty = rewriter.getIntegerType(64, /*isSigned=*/true);
    SmallVector<TypedAttr> whereDims{wRef, rewriter.getIndexAttr(1)};
    auto whereTy = TensorType::get(whereDims, si64Ty);
    auto newWhereOp = rewriter.create<WhereOp>(loc, whereTy, flattened,
                                               ParamDeclArrayAttr::get(wDecl));

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

    // We will now add the expanded range to the transpose of the new where op's
    // output, more or less like this:
    // commonShape = np.broadcast_shapes(where_res.T.shape, aux2.shape)
    // indexes = np.broadcast_to(where_res.T, commonShape)
    //            + np.broadcast_to(aux2, commonShape)
    // Notice that the common shape is always [BS, W].

    // We start with transposing the result of where.
    auto perm = rewriter.create<ConstantOp>(loc, SmallVector<int64_t>{1, 0});
    auto transposeTy = TensorType::get(
        SmallVector<TypedAttr>{whereDims[1], whereDims[0]}, si64Ty);
    auto wTranspose =
        rewriter.create<TransposeOp>(loc, transposeTy, newWhereOp, perm);

    // Then we add the broadcasted transposed indexed and the broadcasted
    // expanded range.
    auto bcastResTy =
        TensorType::get(SmallVector<TypedAttr>{bsRef, wRef}, si64Ty);
    auto indexes = rewriter.create<AddOp>(
        loc,
        rewriter.create<Static::BroadcastToOp>(loc, bcastResTy, wTranspose),
        rewriter.create<Static::BroadcastToOp>(loc, bcastResTy, rangeExpanded));

    // Make sure we don't drop parameter declarations from the original where.
    ArrayRef<ParamDeclAttr> oldDecls = whereOp.getOutputParamDecls();
    assert(oldDecls.size() == 1 && "mo.where declares more than one new param");
    whereOp.setOutputParamDecls({});
    auto numResAttr = KGEN::ParamOperatorAttr::get(
        KGEN::POC::MulNuw, SmallVector<TypedAttr>{bsAttr, wRef});
    rewriter.create<KGEN::ParamDeclareOp>(loc, oldDecls[0], numResAttr);

    // Replace the original where op, with the new result.
    rewriter.replaceOpWithNewOp<Static::ReshapeOp>(whereOp, whereOp.getType(),
                                                   indexes);
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

  LogicalResult matchAndRewrite(MO::BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value input, weight;
    Operation *broadcast;
    if (!m_Op<BatchMatmulOp>(m_Any(&input),
                             m_Op<Static::BroadcastToOp>(
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

    auto shapeTy = rewriter.getType<ShapeType>();
    MO::TensorType reshapeTy =
        inputTy.cloneWith(ShapeAttr::get(newShape, shapeTy));
    Value reshape =
        rewriter.create<Static::ReshapeOp>(op->getLoc(), reshapeTy, input)
            .getResult();

    ArrayRef<TypedAttr> weightShape = weightTy.getDims();
    MO::TensorType matmulTy = inputTy.cloneWith(
        ShapeAttr::get({newShape[0], weightShape[1]}, shapeTy));

    Value matmul =
        rewriter.create<MatmulOp>(op->getLoc(), matmulTy, reshape, weight)
            .getResult();

    rewriter.replaceOpWithNewOp<Static::ReshapeOp>(op, bmmTy, matmul);

    return mlir::success();
  }
};

/// A pattern that detects `mo.conv` with 1x1 filter, 1s for strides and
/// dilations, and rewrites it into a matmul with some static reshapes.
///
/// Example:
///
/// input  : tensor<[N, H, W, Ci]>
/// filter : tensor<[1, 1, Ci, Co]>
///
///   res = conv(input, filter)[strides = {1, ...}, dilations = {1, ...}] : Sh
/// ===>
///   input'  = reshape(input)  : tensor<[N, H, W, Ci]> -> tensor<[N*H*W, Ci]>
///   filter' = reshape(filter) : tensor<[1, 1, Ci, Co]> -> tensor<[Ci, Co]>
///   matmul  = matmul(input', filter')
///   res     = reshape(matmul) : ... -> Sh
struct ConvToMatmulPattern : public OpRewritePattern<ConvOp> {
  ConvToMatmulPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(ConvOp op,
                                PatternRewriter &rewriter) const override {
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

    // Check strides and dilations
    auto areConstantOnes = [](TensorValue val) {
      std::optional<SmallVector<int64_t>> maybeInts =
          tryGetConstantIntVector(val);
      return maybeInts.has_value() &&
             llvm::all_of(maybeInts.value(), [](int64_t n) { return n == 1; });
    };
    if (!areConstantOnes(op.getStrides()))
      return rewriter.notifyMatchFailure(op, "strides must be all 1s");
    if (!areConstantOnes(op.getStrides()))
      return rewriter.notifyMatchFailure(op, "dilations must be all 1s");

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

  LogicalResult matchAndRewrite(ConcatOp op,
                                PatternRewriter &rewriter) const override {

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

  LogicalResult matchAndRewrite(MO::PowOp op,
                                PatternRewriter &rewriter) const override {
    Value input;

    if (!m_Op<PowOp>(m_Any(&input),
                     m_Op<Static::BroadcastToOp>(m_ScalarConstant(2.0)))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input)
      return rewriter.notifyMatchFailure(op, "null input value");

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

  LogicalResult matchAndRewrite(MO::PowOp op,
                                PatternRewriter &rewriter) const override {
    Value input;

    if (!m_Op<PowOp>(m_Any(&input),
                     m_Op<Static::BroadcastToOp>(m_ScalarConstant(3.0)))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input)
      return rewriter.notifyMatchFailure(op, "null input value");
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

  LogicalResult matchAndRewrite(GatherOp op,
                                PatternRewriter &rewriter) const override {
    TensorValue input = op.getInput();
    TensorType inputTy = input.getType();
    if (!inputTy.hasRank())
      return rewriter.notifyMatchFailure(op, "unranked input");

    // Get input's 0-axis dimension
    int64_t inputAxis0Dim = inputTy.getIntDim(0);
    if (mlir::ShapedType::isDynamic(inputAxis0Dim))
      return rewriter.notifyMatchFailure(op, "input axis 0 isn't static");

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

  LogicalResult matchAndRewrite(MO::MulOp op,
                                PatternRewriter &rewriter) const override {
    Value input, input_erf;
    // gelu = x * 1/2 * (1 + erf(x/sqrt(2)))
    auto sq2 =
        m_Op<Static::BroadcastToOp>(m_ScalarConstant(llvm::numbers::sqrt2f));
    auto divsq2 = m_Op<Static::BroadcastToOp>(
        m_ScalarConstant(llvm::numbers::inv_sqrt2f));

    auto x_erf = m_Any(&input_erf);
    auto erf = m_Op<ErfOp>(m_Optional(
        m_Op<DivOp>(x_erf, sq2), m_BinaryCommutativeOp<MulOp>(x_erf, divsq2)));
    auto one = m_Op<Static::BroadcastToOp>(m_ScalarConstant(1.0f));
    auto add = m_BinaryCommutativeOp<AddOp>(one, erf);

    auto half = m_Op<Static::BroadcastToOp>(m_ScalarConstant(0.5f));
    auto x = m_Any(&input);
    // TODO: write an associative commutative matcher?
    auto gelu = m_Optional(m_BinaryCommutativeOp<MulOp>(
                               m_BinaryCommutativeOp<MulOp>(x, half), add),
                           m_BinaryCommutativeOp<MulOp>(
                               half, m_BinaryCommutativeOp<MulOp>(x, add)));

    if (!gelu.match(op->getResult(0)))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !input_erf)
      return rewriter.notifyMatchFailure(op, "null input value");
    if (!(input == input_erf))
      return rewriter.notifyMatchFailure(op, "inputs don't match");

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

  LogicalResult matchAndRewrite(ElementWiseUnary op,
                                PatternRewriter &rewriter) const override {
    Value perm;
    Operation *opaqueTransposeOp;
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

    auto transposeOp = cast<TransposeOp>(opaqueTransposeOp);
    TensorValue input = transposeOp.getInput();
    TensorValue opRes = op.getResult();
    rewriter.replaceAllUsesWith(op->getResults(), transposeOp.getResult());
    op.setInput(input);
    opRes.setType(input.getType());
    transposeOp.getInputMutable().assign(opRes);
    transposeOp->moveAfter(op);

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

  LogicalResult matchAndRewrite(ElementWiseBinary op,
                                PatternRewriter &rewriter) const override {
    Value input1, input2, perm1, perm2;
    auto transpose1P = m_Op<TransposeOp>(m_Any(&input1), m_Any(&perm1));
    auto transpose2P = m_Op<TransposeOp>(m_Any(&input2), m_Any(&perm2));
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

    Type input1Ty = input1.getType();
    if (input1Ty != input2.getType())
      return rewriter.notifyMatchFailure(op, "input types are different");

    TensorValue opRes = op.getResult();
    op.setLhsInput(input1);
    op.setRhsInput(input2);
    Type oldOpResTy = opRes.getType();
    opRes.setType(cast<MO::TensorType>(input1Ty));
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

  LogicalResult matchAndRewrite(Static::BroadcastToOp op,
                                PatternRewriter &rewriter) const override {
    Value input, perm;
    Operation *opaqueTransposeOp;
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
struct SinkReshapeThroughBroadcastedElementwiseBinaryOp
    : public mlir::OpInterfaceRewritePattern<ElementWiseBinary> {
  SinkReshapeThroughBroadcastedElementwiseBinaryOp(MLIRContext *ctx)
      : OpInterfaceRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(ElementWiseBinary op,
                                PatternRewriter &rewriter) const override {

    // Match pattern and get results.
    TensorValue inputTensor, elemwiseTensor;
    auto reshapeP = m_Op<Static::ReshapeOp>(m_Any(&inputTensor));
    auto broadcastP = m_Op<Static::BroadcastToOp>(m_Any(&elemwiseTensor));
    auto elemwiseOpP1 = m_Op<ElementWiseBinary>(reshapeP, broadcastP);
    auto elemwiseOpP2 = m_Op<ElementWiseBinary>(broadcastP, reshapeP);

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

    // Ensure the smallest rank tensor is `elemwise`
    TensorType reshapedTensorTy = reshapedTensor.getType();
    TensorType elemwiseTensorTy = elemwiseTensor.getType();
    TensorType broadcastTensorTy = broadcastTensor.getType();
    TensorType inputTensorTy = inputTensor.getType();
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

    // Now we are sure moving the broadcast up will not change results.
    // Intuition:
    //  - we know slices across the trailing dimension will have the same
    //     corresponding elemwise operands.
    //  - we know the trailing dimensions before and after reshape are the same
    TensorValue newBroadcast = rewriter.create<Static::BroadcastToOp>(
        op->getLoc(), inputTensor.getType(), elemwiseTensor);
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

    TensorValue output = rewriter.create<Static::ReshapeOp>(
        op->getLoc(), oldType, newOp.getResult());

    rewriter.replaceAllUsesWith(op.getResult(), output);
    return mlir::success();
  }
};

/// Drops reshapes through Elementwise Unary Ops.
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
struct SinkReshapeThroughBroadcastedElementwiseUnaryOp
    : public mlir::OpInterfaceRewritePattern<ElementWiseUnary> {
  SinkReshapeThroughBroadcastedElementwiseUnaryOp(MLIRContext *ctx)
      : OpInterfaceRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(ElementWiseUnary op,
                                PatternRewriter &rewriter) const override {
    TensorValue inputTensor;
    auto reshapeP = m_Op<Static::ReshapeOp>(m_Any(&inputTensor));
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

    auto newOp = cast<ElementWiseUnary>(rewriter.clone(*op));
    newOp.setInput(inputTensor);
    newOp.getResult().setType(inputTensor.getType());
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
struct SinkSameReshapesThroughElementwiseBinaryOp
    : public mlir::OpInterfaceRewritePattern<ElementWiseBinary> {
  SinkSameReshapesThroughElementwiseBinaryOp(MLIRContext *ctx)
      : OpInterfaceRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(ElementWiseBinary op,
                                PatternRewriter &rewriter) const override {
    TensorValue input1Tensor, input2Tensor;
    auto reshapeP1 = m_Op<Static::ReshapeOp>(m_Any(&input1Tensor));
    auto reshapeP2 = m_Op<Static::ReshapeOp>(m_Any(&input2Tensor));
    auto elemwiseOpP = m_Op<ElementWiseBinary>(reshapeP1, reshapeP2);

    if (!elemwiseOpP.match(op))
      return rewriter.notifyMatchFailure(op, "Pattern not matched.");

    if (!input1Tensor || !input2Tensor)
      return rewriter.notifyMatchFailure(op, "Null values found in pattern.");

    if (input1Tensor.getType().getDims() != input2Tensor.getType().getDims())
      return rewriter.notifyMatchFailure(
          op, "Inputs are not the same shape. Not compatible.");

    // TODO(#18823) find a more general solution
    if (isValueFromReshapeThatSqueezesFinalDims(op.getLhsInput()) ||
        isValueFromReshapeThatSqueezesFinalDims(op.getRhsInput())) {
      return rewriter.notifyMatchFailure(
          op, "Don't sink reshape that squeezes final dim.");
    }

    auto newOp = cast<ElementWiseBinary>(rewriter.clone(*op));
    newOp.setLhsInput(input1Tensor);
    newOp.setRhsInput(input2Tensor);
    newOp.getResult().setType(input1Tensor.getType());
    TensorValue output = rewriter.create<Static::ReshapeOp>(
        op->getLoc(), op.getResult().getType(), newOp.getResult());

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
  static bool isSqueezingReshape(Operation *op,
                                 ArrayRef<TypedAttr> inferredInDims,
                                 size_t squeezeAxis) {
    // TODO(#13348) remove this check once we have shape inference for SplitOp
    if (auto rebindOp = dyn_cast<RebindOp>(op)) {
      if (!rebindOp->hasOneUse())
        return false;
      op = *rebindOp->user_begin();
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

  LogicalResult matchAndRewrite(SplitOp op,
                                PatternRewriter &rewriter) const override {
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

      if (!isSqueezingReshape(*res.user_begin(), inferredResDims, splitAxis)) {
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

  LogicalResult matchAndRewrite(MO::ElementWiseUnary op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();

    auto inType = input.getType().cast<TensorType>();

    if (!(inType.hasStaticShape() && inType.getRank() == 2 &&
          inType.getIntDim(0) != 1 && inType.getIntDim(1) == 1))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    Value opRes = op.getResult();

    TensorType newType = TensorType::get(
        SmallVector<int64_t>{inType.getIntDim(0)}, inType.getElementType());
    auto reshape = rewriter.create<Static::ReshapeOp>(loc, newType, input);
    op.setInput(reshape.getResult());
    op.getResult().setType(newType);
    rewriter.setInsertionPointAfter(op);
    auto out = rewriter.create<Static::ReshapeOp>(loc, inType, op.getResult());
    rewriter.replaceAllUsesExcept(opRes, out.getResult(), out);
    return mlir::success();
  }
};

struct ElementWiseBinaryVectorPattern
    : public mlir::OpInterfaceRewritePattern<ElementWiseBinary> {
  ElementWiseBinaryVectorPattern(MLIRContext *ctx)
      : OpInterfaceRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::ElementWiseBinary op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value a = op.getLhsInput();
    Value b = op.getRhsInput();

    auto inType = a.getType().cast<TensorType>();

    if (!(inType.hasStaticShape() && inType.getRank() == 2 &&
          inType.getIntDim(0) != 1 && inType.getIntDim(1) == 1))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    Value opRes = op.getResult();

    TensorType newType = TensorType::get(
        SmallVector<int64_t>{inType.getIntDim(0)}, inType.getElementType());
    auto reshapeA = rewriter.create<Static::ReshapeOp>(loc, newType, a);
    auto reshapeB = rewriter.create<Static::ReshapeOp>(loc, newType, b);
    op.setLhsInput(reshapeA.getResult());
    op.setRhsInput(reshapeB.getResult());
    op.getResult().setType(newType);
    rewriter.setInsertionPointAfter(op);
    auto out = rewriter.create<Static::ReshapeOp>(loc, inType, op.getResult());
    rewriter.replaceAllUsesExcept(opRes, out.getResult(), out);
    return mlir::success();
  }
};

struct BroadcastReshapePattern : public OpRewritePattern<Static::ReshapeOp> {
  BroadcastReshapePattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(Static::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    Value input;

    if (!m_Op<Static::ReshapeOp>(m_Op<Static::BroadcastToOp>(m_Any(&input)))
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

  LogicalResult matchAndRewrite(MO::MatmulOp op,
                                PatternRewriter &rewriter) const override {
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

/// A pattern that detects a matmul with a vector weight and a constant
/// input, and transposes it.
struct SplitToSlicePattern : public OpRewritePattern<SplitOp> {
  SplitToSlicePattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::SplitOp op,
                                PatternRewriter &rewriter) const override {
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

    // Replace each result with a split.
    for (Value result : op.getResults()) {
      SmallVector<int64_t> start, stop;

      // Build a start / stop combination for each split.
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

struct SymbolicOptimizationPass
    : public MO::impl::SymbolicOptimizationBase<SymbolicOptimizationPass> {
  using SymbolicOptimizationBase::SymbolicOptimizationBase;
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    // Transpose swimming.
    patterns.add<SinkTransposeThroughUnaryOpPattern,
                 SinkTransposeThroughBinaryOpPattern,
                 SinkTransposeThroughStaticBroadcastToOpPattern>(context);
    TransposeOp::getCanonicalizationPatterns(patterns, context);

    // Reshape insertion/swimming.
    patterns.add<ElementWiseUnaryVectorPattern, ElementWiseBinaryVectorPattern,
                 BroadcastReshapePattern,
                 SinkReshapeThroughBroadcastedElementwiseUnaryOp,
                 SinkReshapeThroughBroadcastedElementwiseBinaryOp,
                 SinkSameReshapesThroughElementwiseBinaryOp>(context);

    // Broadcast swimming.
    patterns.add<CommuteBroadcastWithFlatennedWhere>(context);
    Static::BroadcastToOp::getCanonicalizationPatterns(patterns, context);

    // Miscellaneous rewrites.
    patterns.add<ConvertBatchMatmulToMatmulPattern, ConvToMatmulPattern,
                 FlattenNestedConcatPattern, Pow2Pattern, Pow3Pattern,
                 GathersToSplitPattern, GeluPattern, GemvPattern,
                 SplitAndSqueezesPattern, SplitToSlicePattern>(context);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
