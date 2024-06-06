//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/RMOSupport/Broadcast.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GenericML/GraphCompiler/Matchers.h"
#include "GenericML/GraphCompiler/RMODialect/RMOOps.h"
#include "GenericML/GraphCompiler/Transforms/RMODialect/RMOTransforms.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetOperations.h"

using namespace M;
using namespace graph_compiler::matchers;

namespace M::RMO {
#define GEN_PASS_DEF_LEGALIZERMOOPS
#include "GenericML/GraphCompiler/Transforms/RMODialect/RMOTransforms.h.inc"
} // namespace M::RMO

namespace {

/// NOTE: RMO's rebinds are asserting in nature!
TensorValue rebindToShape(MO::GraphOp graphOp, TensorValue input,
                          TensorType newTy, mlir::RewriterBase &rewriter,
                          ArrayRef<ParamDeclAttr> newOutputParamDecl = {}) {
  return rewriter.create<RMO::RebindTensorShapeOp>(input.getLoc(), newTy, input,
                                                   newOutputParamDecl);
}

/// `outputParamDecls` contains the param declarations for all outputs.
/// This function fills `outParamDecls` with only the param declaration needed
/// for `targetTy`. Needed for multi-outputs op.
void getParamDeclsFor(TensorType targetTy,
                      ArrayRef<KGEN::ParamDeclAttr> allParamDecls,
                      llvm::SmallVectorImpl<ParamDeclAttr> &outParamDecls) {
  auto addAttrIfDecl = [&](TypedAttr attr) {
    KGEN::ParamDeclRefAttr paramRefAttr =
        dyn_cast<KGEN::ParamDeclRefAttr>(attr);
    if (!paramRefAttr)
      return;
    for (KGEN::ParamDeclAttr decl : allParamDecls)
      if (decl.getName() == paramRefAttr.getName()) {
        outParamDecls.push_back(decl);
      }
  };

  addAttrIfDecl(targetTy.getShapeAttr());
  if (targetTy.hasRank())
    for (TypedAttr dimAttr : targetTy.getTypedShapeAttr().getValues())
      addAttrIfDecl(dimAttr);
}

/// Lower the given RMO::MOAnalogue op to it's MO analogue given by `generator`.
LogicalResult
lowerMOAnalogue(RMO::MOAnalogue op, PatternRewriter &rewriter,
                const std::function<MO::ParamDeclarationInterface(
                    PatternRewriter &rewriter, const Location &, ArrayRef<Type>,
                    ArrayRef<Value>)> &generator) {
  auto graphOp = op->getParentOfType<MO::GraphOp>();

  size_t numOperands = op->getNumOperands();
  size_t numResults = op->getNumResults();

  // Infer the MO types to binds
  SmallVector<Type> moOperandTypes, moResultTypes;
  SmallVector<SmallVector<KGEN::ParamDeclAttr>> moInputParamDecls;
  SmallVector<SmallVector<KGEN::ParamDeclAttr>> moOutputParamDecls;
  op.inferMOTypes(op->getContext(), graphOp, op->getOperands(),
                  op->getAttrDictionary(), op->getResults(), moOperandTypes,
                  moResultTypes, moInputParamDecls, moOutputParamDecls);

  // Rebind inputs to what the MO op would expect
  SmallVector<Value> newOperands;
  for (size_t i = 0; i < numOperands; i++) {
    Value val = op->getOperand(i);
    Type targetType = moOperandTypes[i];
    TensorValue tensorVal = dyn_cast<TensorValue>(val);
    if (!tensorVal) {
      ASSERT_STREAM(targetType == val.getType())
          << "Disagreement on type for non-tensor argument.";
      newOperands.push_back(tensorVal);
      continue;
    }
    SmallVector<KGEN::ParamDeclAttr> &curInputDecls = moInputParamDecls[i];

    TensorValue reboundValue =
        rebindToShape(graphOp, tensorVal, cast<TensorType>(targetType),
                      rewriter, curInputDecls);
    newOperands.push_back(reboundValue);
  }

  // Build the MO op.
  SmallVector<KGEN::ParamDeclAttr> moOutputParamDeclsFlattened;
  for (const auto &resultDecls : moOutputParamDecls) {
    moOutputParamDeclsFlattened.append(resultDecls);
  }
  MO::ParamDeclarationInterface newOp =
      generator(rewriter, op->getLoc(), moResultTypes, newOperands);
  newOp.setOutputParamDecls(moOutputParamDeclsFlattened);

  // Rebind the output of the MO op to what the original op had and replace
  for (size_t i = 0; i < numResults; i++) {
    Value newValue = newOp->getResult(i);
    Value oldValue = op->getResult(i);
    TensorType targetType = cast<TensorType>(op->getResultTypes()[i]);

    TensorValue newTensorValue = dyn_cast<TensorValue>(newValue);
    if (!newTensorValue) {
      rewriter.replaceAllUsesWith(oldValue, newValue);
      continue;
    }

    TensorValue reboundValue;
    if (numResults == 1) {
      reboundValue =
          rebindToShape(graphOp, newTensorValue, targetType, rewriter,
                        op.getMOAnalogueOutputParamDecls());
    } else {
      // For multiple-output ops, op.getMOAnalogueOutputParamDecls() returns all
      // param declarations with all outputs. We need to filter and keep only
      // the param decls for every different output (to avoid param
      // redefinitions).
      llvm::SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
      getParamDeclsFor(targetType, op.getMOAnalogueOutputParamDecls(),
                       outputParamDecls);
      reboundValue = rebindToShape(graphOp, newTensorValue, targetType,
                                   rewriter, outputParamDecls);
    }
    rewriter.replaceAllUsesWith(oldValue, reboundValue);
  }

  return success();
}

template <typename RMOOpT, typename MOOpT>
struct LegalizeRMOMOAnalogue : public OpRewritePattern<RMOOpT> {
  LegalizeRMOMOAnalogue(MLIRContext *ctx) : OpRewritePattern<RMOOpT>(ctx, 1) {}
  LogicalResult matchAndRewrite(RMOOpT op,
                                PatternRewriter &rewriter) const override {
    auto generator =
        [&op](PatternRewriter &rewriter, const Location &loc,
              ArrayRef<Type> resultTypes,
              ArrayRef<Value> operands) -> MO::ParamDeclarationInterface {
      auto result = rewriter.create<MOOpT>(loc, resultTypes, operands);
      // Copy all attributes (except the outputParamDecls).
      for (auto attr : op->getAttrs())
        if (attr.getName() != op.getOutputParamDeclsAttrName())
          rewriter.modifyOpInPlace(result, [&] {
            result->setAttr(attr.getName(), attr.getValue());
          });
      return cast<MO::ParamDeclarationInterface>(result.getOperation());
    };
    return lowerMOAnalogue(op, rewriter, generator);
  }
};

struct LowerRebindTensorShapeOp
    : public OpRewritePattern<RMO::RebindTensorShapeOp> {
  LowerRebindTensorShapeOp(MLIRContext *ctx) : OpRewritePattern(ctx, 1) {}
  LogicalResult matchAndRewrite(RMO::RebindTensorShapeOp op,
                                PatternRewriter &rewriter) const override {

    TensorValue operand = op.getInput();
    TensorValue result = op.getResult();
    TypedAttr operandShape = operand.getType().getShapeAttr();
    TypedAttr resultShape = result.getType().getShapeAttr();

    MO::GraphOp graphOp = op->getParentOfType<MO::GraphOp>();
    Location loc = op->getLoc();

    // Create KGEN parameter containing result of shape equality statement
    IntegerType boolTy = rewriter.getI1Type();
    TypedAttr cmpAttr =
        KGEN::emitMLIROperationCall(MOSH::EqualOp::getOperationName(), {},
                                    {operandShape, resultShape}, boolTy);
    KGEN::ParamDeclAttr condAttr =
        graphOp.getNewParamOfType("RMO_SH_EQ", boolTy);
    rewriter.create<KGEN::ParamDeclareOp>(loc, condAttr, cmpAttr);

    // Add assertion, then return a rebind
    StringAttr msg = mlir::cast<StringAttr>(op.getMessageAttr());
    TypedAttr assertAttr = MO::AssertOp::emitStaticCall(
        KGEN::ParamDeclRefAttr::get(condAttr), msg.str());

    auto chainTy = rewriter.getType<ChainType>();
    KGEN::ParamDeclAttr chainAttr =
        graphOp.getNewParamOfType("RMO_SH_CH", chainTy);
    rewriter.create<KGEN::ParamDeclareOp>(loc, chainAttr, assertAttr);

    // Return rebound result
    auto moRebind = rewriter.create<MO::RebindOp>(
        loc, result.getType(), operand, op.getOutputParamDecls());
    rewriter.replaceOp(op, moRebind);

    return success();
  }
};

template <typename RMOOpT, typename MOAnalogueOpT>
struct LowerThirdPartyBinaryElemwiseOp : public OpRewritePattern<RMOOpT> {
  LowerThirdPartyBinaryElemwiseOp(MLIRContext *ctx)
      : OpRewritePattern<RMOOpT>(ctx, 1) {}
  LogicalResult matchAndRewrite(RMOOpT op,
                                PatternRewriter &rewriter) const override {
    auto graphOp = op->template getParentOfType<MO::GraphOp>();
    Location loc = op->getLoc();

    TensorValue x = op.getInputX();
    TensorValue y = op.getInputY();
    auto xDtype = x.getType().getDtype();
    auto yDtype = y.getType().getDtype();
    auto resultDtype = op.getResult().getType().getDtype();

    // Deal with dtype casting if it is needed.
    if (xDtype != resultDtype) {
      x = rewriter.create<RMO::MOCastOp>(loc,
                                         x.getType().cloneWith(resultDtype), x);
    }
    if (yDtype != resultDtype) {
      y = rewriter.create<RMO::MOCastOp>(loc,
                                         y.getType().cloneWith(resultDtype), y);
    }

    TensorValue xFinal = x;
    TensorValue yFinal = y;
    if (x.getType().getTypedShapeAttr() != y.getType().getTypedShapeAttr()) {
      TensorValue xShape =
          rewriter.create<RMO::MOShapeOfOp>(loc, ValueRange{x}, graphOp);
      TensorValue yShape =
          rewriter.create<RMO::MOShapeOfOp>(loc, ValueRange{y}, graphOp);

      TensorValue commonShape = rewriter.create<RMO::MOBroadcastShapeOp>(
          loc, ValueRange{xShape, yShape}, graphOp);

      TensorValue xBcast = rewriter.create<RMO::MOBroadcastToOp>(
          loc, ValueRange{x, commonShape}, graphOp);
      TensorValue yBcast = rewriter.create<RMO::MOBroadcastToOp>(
          loc, ValueRange{y, commonShape}, graphOp);
      xFinal = xBcast;
      yFinal = yBcast;
    }

    TensorValue answer = rewriter.create<MOAnalogueOpT>(
        loc, ValueRange{xFinal, yFinal}, graphOp);
    TensorValue reboundAnswer = rebindToShape(
        graphOp, answer, op.getType(), rewriter, op.getOutputParamDecls());

    rewriter.replaceAllUsesWith(op, reboundAnswer);
    return success();
  }
};

template <typename RMOOpT, typename MOAnalogueOpT>
struct LowerThirdPartyBinaryComparisonOp : public OpRewritePattern<RMOOpT> {
  LowerThirdPartyBinaryComparisonOp(MLIRContext *ctx)
      : OpRewritePattern<RMOOpT>(ctx, 1) {}
  LogicalResult matchAndRewrite(RMOOpT op,
                                PatternRewriter &rewriter) const override {
    auto graphOp = op->template getParentOfType<MO::GraphOp>();
    Location loc = op->getLoc();

    TensorValue x = op.getInputX();
    TensorValue y = op.getInputY();

    TensorValue xFinal = x;
    TensorValue yFinal = y;
    if (x.getType().getTypedShapeAttr() != y.getType().getTypedShapeAttr()) {
      TensorValue xShape =
          rewriter.create<RMO::MOShapeOfOp>(loc, ValueRange{x}, graphOp);
      TensorValue yShape =
          rewriter.create<RMO::MOShapeOfOp>(loc, ValueRange{y}, graphOp);

      TensorValue commonShape = rewriter.create<RMO::MOBroadcastShapeOp>(
          loc, ValueRange{xShape, yShape}, graphOp);

      TensorValue xBcast = rewriter.create<RMO::MOBroadcastToOp>(
          loc, ValueRange{x, commonShape}, graphOp);
      TensorValue yBcast = rewriter.create<RMO::MOBroadcastToOp>(
          loc, ValueRange{y, commonShape}, graphOp);
      xFinal = xBcast;
      yFinal = yBcast;
    }

    TensorValue answer = rewriter.create<MOAnalogueOpT>(
        loc, ValueRange{xFinal, yFinal}, graphOp);
    TensorValue reboundAnswer = rebindToShape(
        graphOp, answer, op.getType(), rewriter, op.getOutputParamDecls());

    rewriter.replaceAllUsesWith(op, reboundAnswer);
    return success();
  }
};

/// Returns the input values with asserting rebinds to ensure parameterized
/// dimensions are equal.
///
/// Broadcasting in RMO does not allow parameterized dimensions to broadcast.
/// As such, parameterized dimensions need an asserting rebind to the expected
/// output. Afterwards, a normal broadcast can be applied.
std::pair<TensorValue, TensorValue>
rebindShapesForBroadcasting(Location loc, PatternRewriter &rewriter,
                            TensorValue x, TensorValue y) {
  TensorType xType = x.getType();
  TensorType yType = y.getType();
  ShapeAttr xShapeAttr = xType.getTypedShapeAttr();
  ShapeAttr yShapeAttr = yType.getTypedShapeAttr();

  if (xShapeAttr == yShapeAttr) {
    return {x, y};
  }

  ArrayRef<TypedAttr> xShape = xShapeAttr.getValues();
  ArrayRef<TypedAttr> yShape = yShapeAttr.getValues();
  SmallVector<TypedAttr> xRebindShape{xShape};
  SmallVector<TypedAttr> yRebindShape{yShape};

  size_t xSize = xShape.size();
  size_t ySize = yShape.size();
  size_t minSize = std::min(xSize, ySize);

  auto idxTy = IndexType::get(xType.getContext());
  auto intDim1 = IntegerAttr::get(idxTy, 1ll);

  // Note: dimensions match from the innermost to outermost.
  // Thus, accesses are with `size - i - 1`.
  for (size_t i = 0; i < minSize; ++i) {
    // Parameters in x get rebound if the value in y is a static integer besides
    // a static one dim.
    if (TensorType::isParameterizedDimParam(xRebindShape[xSize - i - 1]) &&
        TensorType::isIntegerDimParam(yShape[ySize - i - 1]) &&
        yShape[ySize - i - 1] != intDim1)
      xRebindShape[xSize - i - 1] = yShape[ySize - i - 1];

    // Parameters in y get rebound to the value in x as long as it isn't a
    // static one dim.
    if (TensorType::isParameterizedDimParam(yRebindShape[ySize - i - 1]) &&
        xShape[xSize - i - 1] != intDim1)
      yRebindShape[ySize - i - 1] = xShape[xSize - i - 1];
  }

  TensorType xRebindType =
      xType.cloneWith(ShapeAttr::get(xRebindShape, xShapeAttr.getType()));
  TensorType yRebindType =
      yType.cloneWith(ShapeAttr::get(yRebindShape, yShapeAttr.getType()));

  if (xRebindType != xType)
    x = rewriter.create<RMO::RebindTensorShapeOp>(loc, xRebindType, x);

  if (yRebindType != yType)
    y = rewriter.create<RMO::RebindTensorShapeOp>(loc, yRebindType, y);

  return {x, y};
}

template <typename RMOOpT, typename MOAnalogueOpT>
struct LowerCompositeBinaryElemwiseOp : public OpRewritePattern<RMOOpT> {
  LowerCompositeBinaryElemwiseOp(MLIRContext *ctx)
      : OpRewritePattern<RMOOpT>(ctx, 1) {}
  LogicalResult matchAndRewrite(RMOOpT op,
                                PatternRewriter &rewriter) const override {
    auto graphOp = op->template getParentOfType<MO::GraphOp>();
    Location loc = op->getLoc();

    TensorValue x = op.getInputX();
    TensorValue y = op.getInputY();
    auto xDtype = x.getType().getDtype();
    auto yDtype = y.getType().getDtype();
    auto resultDtype = op.getResult().getType().getDtype();

    // Deal with dtype casting if it is needed.
    if (xDtype != resultDtype) {
      x = rewriter.create<RMO::MOCastOp>(loc,
                                         x.getType().cloneWith(resultDtype), x);
    }
    if (yDtype != resultDtype) {
      y = rewriter.create<RMO::MOCastOp>(loc,
                                         y.getType().cloneWith(resultDtype), y);
    }

    auto [xRebound, yRebound] =
        rebindShapesForBroadcasting(loc, rewriter, x, y);

    TensorValue xFinal = xRebound;
    TensorValue yFinal = yRebound;
    if (xRebound.getType().getTypedShapeAttr() !=
        yRebound.getType().getTypedShapeAttr()) {
      TensorValue xShape =
          rewriter.create<RMO::MOShapeOfOp>(loc, ValueRange{xRebound}, graphOp);
      TensorValue yShape =
          rewriter.create<RMO::MOShapeOfOp>(loc, ValueRange{yRebound}, graphOp);

      TensorValue commonShape = rewriter.create<RMO::MOBroadcastShapeOp>(
          loc, ValueRange{xShape, yShape}, graphOp);

      TensorValue xBcast = rewriter.create<RMO::MOBroadcastToOp>(
          loc, ValueRange{xRebound, commonShape}, graphOp);
      TensorValue yBcast = rewriter.create<RMO::MOBroadcastToOp>(
          loc, ValueRange{yRebound, commonShape}, graphOp);
      xFinal = xBcast;
      yFinal = yBcast;
    }

    TensorValue answer = rewriter.create<MOAnalogueOpT>(
        loc, ValueRange{xFinal, yFinal}, graphOp);
    TensorValue reboundAnswer = rebindToShape(
        graphOp, answer, op.getType(), rewriter, op.getOutputParamDecls());

    rewriter.replaceAllUsesWith(op, reboundAnswer);
    return success();
  }
};

template <typename RMOOpT, typename MOAnalogueOpT>
struct LowerCompositeBinaryComparisonOp : public OpRewritePattern<RMOOpT> {
  LowerCompositeBinaryComparisonOp(MLIRContext *ctx)
      : OpRewritePattern<RMOOpT>(ctx, 1) {}
  LogicalResult matchAndRewrite(RMOOpT op,
                                PatternRewriter &rewriter) const override {
    auto graphOp = op->template getParentOfType<MO::GraphOp>();
    Location loc = op->getLoc();

    TensorValue x = op.getInputX();
    TensorValue y = op.getInputY();
    auto xDtype = x.getType().getDtype();
    auto yDtype = y.getType().getDtype();

    // Deal with dtype casting if it is needed.
    ErrorOr<DType> promotedDtype = M::RMO::promoteDtype(xDtype, yDtype);
    ASSERT_STREAM(!promotedDtype.isError())
        << "failed to calculate valid promotion dtype in lowering. This should "
           "be impossible due to it being calculated when inferring the return "
           "type"
        << promotedDtype.takeError();
    DType comparisonDtype = promotedDtype.takeValue();
    if (xDtype != comparisonDtype) {
      x = rewriter.create<RMO::MOCastOp>(
          loc, x.getType().cloneWith(comparisonDtype), x);
    }
    if (yDtype != comparisonDtype) {
      y = rewriter.create<RMO::MOCastOp>(
          loc, y.getType().cloneWith(comparisonDtype), y);
    }

    auto [xRebound, yRebound] =
        rebindShapesForBroadcasting(loc, rewriter, x, y);

    TensorValue xFinal = xRebound;
    TensorValue yFinal = yRebound;
    if (xRebound.getType().getTypedShapeAttr() !=
        yRebound.getType().getTypedShapeAttr()) {
      TensorValue xShape =
          rewriter.create<RMO::MOShapeOfOp>(loc, ValueRange{xRebound}, graphOp);
      TensorValue yShape =
          rewriter.create<RMO::MOShapeOfOp>(loc, ValueRange{yRebound}, graphOp);

      TensorValue commonShape = rewriter.create<RMO::MOBroadcastShapeOp>(
          loc, ValueRange{xShape, yShape}, graphOp);

      TensorValue xBcast = rewriter.create<RMO::MOBroadcastToOp>(
          loc, ValueRange{xRebound, commonShape}, graphOp);
      TensorValue yBcast = rewriter.create<RMO::MOBroadcastToOp>(
          loc, ValueRange{yRebound, commonShape}, graphOp);
      xFinal = xBcast;
      yFinal = yBcast;
    }

    TensorValue answer = rewriter.create<MOAnalogueOpT>(
        loc, ValueRange{xFinal, yFinal}, graphOp);
    TensorValue reboundAnswer = rebindToShape(
        graphOp, answer, op.getType(), rewriter, op.getOutputParamDecls());

    rewriter.replaceAllUsesWith(op, reboundAnswer);
    return success();
  }
};

struct LowerBroadcastLike : public OpRewritePattern<RMO::BroadcastLikeOp> {
  LowerBroadcastLike(MLIRContext *ctx) : OpRewritePattern(ctx, 1) {}
  LogicalResult matchAndRewrite(RMO::BroadcastLikeOp op,
                                PatternRewriter &rewriter) const override {
    auto graphOp = op->getParentOfType<MO::GraphOp>();
    Location loc = op->getLoc();

    TensorValue shapeLike = rewriter.create<RMO::MOShapeOfOp>(
        loc, ValueRange{op.getShapeLike()}, graphOp);
    TensorValue answer = rewriter.create<RMO::MOBroadcastToOp>(
        loc, ValueRange{op.getInput(), shapeLike}, graphOp);
    TensorValue reboundAnswer = rebindToShape(
        graphOp, answer, op.getType(), rewriter, op.getOutputParamDecls());

    rewriter.replaceAllUsesWith(op, reboundAnswer);
    return success();
  }
};

struct LowerConcat : public OpRewritePattern<RMO::ConcatOp> {
  LowerConcat(MLIRContext *ctx) : OpRewritePattern(ctx, 1) {}
  LogicalResult matchAndRewrite(RMO::ConcatOp op,
                                PatternRewriter &rewriter) const override {

    // Create a tensor value for axis.
    Location loc = op->getLoc();
    TensorValue axis =
        rewriter.create<MO::ConstantOp>(loc, op.getAxis().getInt());

    // Replace the `rmo.concat` op with a new `mo.concat` op.
    rewriter.replaceOpWithNewOp<MO::ConcatOp>(op, op.getResult().getType(),
                                              axis, op.getInputs(),
                                              op.getOutputParamDecls());
    return success();
  }
};

struct LowerSigmoid : public OpRewritePattern<RMO::SigmoidOp> {
  LowerSigmoid(MLIRContext *ctx) : OpRewritePattern(ctx, 1) {}
  LogicalResult matchAndRewrite(RMO::SigmoidOp op,
                                PatternRewriter &rewriter) const override {
    auto graphOp = op->getParentOfType<MO::GraphOp>();
    Location loc = op->getLoc();

    // Let's decompse the sigmoid operation into RMO analogue operations to
    // perform this calculation:
    //
    // 1/(1+e^-x)
    //

    TensorValue x = op.getInput();

    TensorValue one = rewriter.create<MO::ConstantOp>(loc, 1.0f);

    TensorValue oneAsInputDtype =
        rewriter.create<RMO::MOCastOp>(loc, one, x.getType().getDtype());

    TensorValue tensorOne = RMO::broadcastTo(rewriter, loc, oneAsInputDtype, x);

    TensorValue negatedX =
        rewriter.create<RMO::MONegativeOp>(loc, ValueRange{x}, graphOp);
    TensorValue eToTheNegatedX =
        rewriter.create<RMO::MOExpOp>(loc, ValueRange{negatedX}, graphOp);

    TensorValue denominator = rewriter.create<RMO::MOAddOp>(
        loc, ValueRange{tensorOne, eToTheNegatedX}, graphOp);
    TensorValue answer = rewriter.create<RMO::MODivOp>(
        loc, ValueRange{tensorOne, denominator}, graphOp);

    TensorValue reboundAnswer = rebindToShape(
        graphOp, answer, op.getType(), rewriter, op.getOutputParamDecls());

    rewriter.replaceAllUsesWith(op, reboundAnswer);
    return success();
  }
};

struct LegalizeRMOOps
    : public M::RMO::impl::LegalizeRMOOpsBase<LegalizeRMOOps> {
  using LegalizeRMOOpsBase::LegalizeRMOOpsBase;
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet blockPatterns(context);

    // Lowering MOAnalogue Operations
    blockPatterns.add<
        // Elementwise unary ops
        LegalizeRMOMOAnalogue<RMO::MOAbsOp, MO::AbsOp>,
        LegalizeRMOMOAnalogue<RMO::MOCeilOp, MO::CeilOp>,
        LegalizeRMOMOAnalogue<RMO::MOCosOp, MO::CosOp>,
        LegalizeRMOMOAnalogue<RMO::MOErfOp, MO::ErfOp>,
        LegalizeRMOMOAnalogue<RMO::MOExpOp, MO::ExpOp>,
        LegalizeRMOMOAnalogue<RMO::MOFloorOp, MO::FloorOp>,
        LegalizeRMOMOAnalogue<RMO::MOGeluOp, MO::GeluOp>,
        LegalizeRMOMOAnalogue<RMO::MOLogOp, MO::LogOp>,
        LegalizeRMOMOAnalogue<RMO::MOLog1pOp, MO::Log1pOp>,
        LegalizeRMOMOAnalogue<RMO::MOLogSoftmaxOp, MO::LogSoftmaxOp>,
        LegalizeRMOMOAnalogue<RMO::MONegativeOp, MO::NegativeOp>,
        LegalizeRMOMOAnalogue<RMO::MONotOp, MO::NotOp>,
        LegalizeRMOMOAnalogue<RMO::MOReluOp, MO::ReluOp>,
        LegalizeRMOMOAnalogue<RMO::MORoundOp, MO::RoundOp>,
        LegalizeRMOMOAnalogue<RMO::MORoundEvenOp, MO::RoundEvenOp>,
        LegalizeRMOMOAnalogue<RMO::MORsqrtOp, MO::RsqrtOp>,
        LegalizeRMOMOAnalogue<RMO::MOSinOp, MO::SinOp>,
        LegalizeRMOMOAnalogue<RMO::MOSqrtOp, MO::SqrtOp>,
        LegalizeRMOMOAnalogue<RMO::MOTanhOp, MO::TanhOp>,
        LegalizeRMOMOAnalogue<RMO::MOTruncOp, MO::TruncOp>,
        LegalizeRMOMOAnalogue<RMO::MOSoftmaxOp, MO::SoftmaxOp>,
        // Elementwise binary ops
        LegalizeRMOMOAnalogue<RMO::MOAddOp, MO::AddOp>,
        LegalizeRMOMOAnalogue<RMO::MOMulOp, MO::MulOp>,
        LegalizeRMOMOAnalogue<RMO::MOModOp, MO::ModOp>,
        LegalizeRMOMOAnalogue<RMO::MOMinOp, MO::MinOp>,
        LegalizeRMOMOAnalogue<RMO::MOMaxOp, MO::MaxOp>,
        LegalizeRMOMOAnalogue<RMO::MODivOp, MO::DivOp>,
        LegalizeRMOMOAnalogue<RMO::MOPowOp, MO::PowOp>,
        LegalizeRMOMOAnalogue<RMO::MOSubOp, MO::SubOp>,
        LegalizeRMOMOAnalogue<RMO::MOAndOp, MO::AndOp>,
        LegalizeRMOMOAnalogue<RMO::MOOrOp, MO::OrOp>,
        LegalizeRMOMOAnalogue<RMO::MOXorOp, MO::XorOp>,
        // Comparison binary ops
        LegalizeRMOMOAnalogue<RMO::MOEqualOp, MO::EqualOp>,
        LegalizeRMOMOAnalogue<RMO::MOGreaterOp, MO::GreaterOp>,
        LegalizeRMOMOAnalogue<RMO::MOGreaterEqualOp, MO::GreaterEqualOp>,
        LegalizeRMOMOAnalogue<RMO::MONotEqualOp, MO::NotEqualOp>,
        // Comparison unary ops
        LegalizeRMOMOAnalogue<RMO::MOIsNaNOp, MO::IsNaNOp>,
        LegalizeRMOMOAnalogue<RMO::MOIsInfOp, MO::IsInfOp>,
        // Other ops
        LegalizeRMOMOAnalogue<RMO::MOShapeOfOp, MO::ShapeOfOp>,
        LegalizeRMOMOAnalogue<RMO::MOBroadcastShapeOp, MO::BroadcastShapeOp>,
        LegalizeRMOMOAnalogue<RMO::MOReshapeOp, MO::ReshapeOp>,
        LegalizeRMOMOAnalogue<RMO::MOBroadcastToOp, MO::BroadcastToOp>,
        LegalizeRMOMOAnalogue<RMO::MOCastOp, MO::CastOp>,
        LegalizeRMOMOAnalogue<RMO::MOMeanOp, MO::MeanOp>,
        LegalizeRMOMOAnalogue<RMO::MOReduceMulOp, MO::ReduceMulOp>,
        LegalizeRMOMOAnalogue<RMO::MOReduceAddOp, MO::ReduceAddOp>,
        LegalizeRMOMOAnalogue<RMO::MOReduceMinOp, MO::ReduceMinOp>,
        LegalizeRMOMOAnalogue<RMO::MOReduceMaxOp, MO::ReduceMaxOp>,
        LegalizeRMOMOAnalogue<RMO::MOArgMinOp, MO::ArgMinOp>,
        LegalizeRMOMOAnalogue<RMO::MOArgMaxOp, MO::ArgMaxOp>,
        LegalizeRMOMOAnalogue<RMO::MOScatterOp, MO::ScatterOp>,
        LegalizeRMOMOAnalogue<RMO::MOScatterAddOp, MO::ScatterAddOp>,
        LegalizeRMOMOAnalogue<RMO::MOScatterMulOp, MO::ScatterMulOp>,
        LegalizeRMOMOAnalogue<RMO::MOScatterMinOp, MO::ScatterMinOp>,
        LegalizeRMOMOAnalogue<RMO::MOScatterMaxOp, MO::ScatterMaxOp>,
        LegalizeRMOMOAnalogue<RMO::MOConvOp, MO::ConvOp>,
        LegalizeRMOMOAnalogue<RMO::MOConvTransposeOp, MO::ConvTransposeOp>,
        LegalizeRMOMOAnalogue<RMO::MOAvgPoolOp, MO::AvgPoolOp>,
        LegalizeRMOMOAnalogue<RMO::MOAvgPoolCeilOp, MO::AvgPoolCeilOp>,
        LegalizeRMOMOAnalogue<RMO::MOMaxPoolOp, MO::MaxPoolOp>,
        LegalizeRMOMOAnalogue<RMO::MOMaxPoolCeilOp, MO::MaxPoolCeilOp>,
        LegalizeRMOMOAnalogue<RMO::MORangeOp, MO::RangeOp>,
        LegalizeRMOMOAnalogue<RMO::MOResizeLinearOp, MO::ResizeLinearOp>,
        LegalizeRMOMOAnalogue<RMO::MOResizeNearestOp, MO::ResizeNearestOp>,
        LegalizeRMOMOAnalogue<RMO::MOSelectOp, MO::SelectOp>,
        LegalizeRMOMOAnalogue<RMO::MOGatherOp, MO::GatherOp>,
        LegalizeRMOMOAnalogue<RMO::MOGatherNdOp, MO::GatherNdOp>,
        LegalizeRMOMOAnalogue<RMO::MOMatmulOp, MO::MatmulOp>,
        LegalizeRMOMOAnalogue<RMO::MOBatchMatmulOp, MO::BatchMatmulOp>,
        LegalizeRMOMOAnalogue<RMO::MOTileOp, MO::TileOp>,
        LegalizeRMOMOAnalogue<RMO::MOSqueezeShapeOp, MO::SqueezeShapeOp>,
        LegalizeRMOMOAnalogue<RMO::MOUnsqueezeShapeOp, MO::UnsqueezeShapeOp>,
        LegalizeRMOMOAnalogue<RMO::MOTransposeOp, MO::TransposeOp>,
        LegalizeRMOMOAnalogue<RMO::MOTopKOp, MO::TopKOp>,
        LegalizeRMOMOAnalogue<RMO::MOBottomKOp, MO::BottomKOp>,
        LegalizeRMOMOAnalogue<RMO::LinAlg::MOBandPartOp,
                              MO::LinAlg::BandPartOp>,
        LegalizeRMOMOAnalogue<RMO::LinAlg::MOSolveOp, MO::LinAlg::SolveOp>,
        LegalizeRMOMOAnalogue<RMO::MOConcatOp, MO::ConcatOp>,
        LegalizeRMOMOAnalogue<RMO::MOSplitOp, MO::SplitOp>,
        LegalizeRMOMOAnalogue<RMO::MOPadConstantOp, MO::PadConstantOp>,
        LegalizeRMOMOAnalogue<RMO::MOPadReflectOp, MO::PadReflectOp>,
        LegalizeRMOMOAnalogue<RMO::MOPadRepeatOp, MO::PadRepeatOp>,
        LegalizeRMOMOAnalogue<RMO::MOSliceOp, MO::SliceOp>,
        LegalizeRMOMOAnalogue<RMO::MORoiAlignOp, MO::RoiAlignOp>,
        LegalizeRMOMOAnalogue<RMO::MORandomNormalOp, MO::RandomNormalOp>,
        LegalizeRMOMOAnalogue<RMO::MOScatterNdOp, MO::ScatterNdOp>,
        LegalizeRMOMOAnalogue<RMO::MOScatterNdAddOp, MO::ScatterNdAddOp>,
        LegalizeRMOMOAnalogue<RMO::MOScatterNdMaxOp, MO::ScatterNdMaxOp>,
        LegalizeRMOMOAnalogue<RMO::MOScatterNdMinOp, MO::ScatterNdMinOp>,
        LegalizeRMOMOAnalogue<RMO::MOScatterNdMulOp, MO::ScatterNdMulOp>,
        LegalizeRMOMOAnalogue<RMO::MONonMaximumSuppressionOp,
                              MO::NonMaximumSuppressionOp>,
        LegalizeRMOMOAnalogue<RMO::MOArgNonzeroOp, MO::ArgNonzeroOp>,
        LegalizeRMOMOAnalogue<RMO::MOCumSumOp, MO::CumSumOp>>(context);

    // Lowering rebind
    if (!skipRMORebind)
      blockPatterns.add<LowerRebindTensorShapeOp>(context);

    // Lowering composite operations
    blockPatterns.add<
        LowerCompositeBinaryElemwiseOp<RMO::AddOp, RMO::MOAddOp>,
        LowerCompositeBinaryElemwiseOp<RMO::SubOp, RMO::MOSubOp>,
        LowerCompositeBinaryElemwiseOp<RMO::MulOp, RMO::MOMulOp>,
        LowerCompositeBinaryElemwiseOp<RMO::DivOp, RMO::MODivOp>,
        LowerCompositeBinaryElemwiseOp<RMO::ModOp, RMO::MOModOp>,
        LowerCompositeBinaryElemwiseOp<RMO::PowOp, RMO::MOPowOp>,
        LowerCompositeBinaryElemwiseOp<RMO::MinOp, RMO::MOMinOp>,
        LowerCompositeBinaryElemwiseOp<RMO::MaxOp, RMO::MOMaxOp>,
        // Comparison ops lower the same as other elemwise ops.
        LowerCompositeBinaryComparisonOp<RMO::EqualOp, RMO::MOEqualOp>,
        LowerCompositeBinaryComparisonOp<RMO::NotEqualOp, RMO::MONotEqualOp>,
        LowerCompositeBinaryComparisonOp<RMO::GreaterOp, RMO::MOGreaterOp>,
        LowerCompositeBinaryComparisonOp<RMO::GreaterEqualOp,
                                         RMO::MOGreaterEqualOp>,
        LowerBroadcastLike, LowerSigmoid, LowerConcat>(context);

    // Lowering third party specific operations.
    // They have more flexible semantics that separate them from standard rmo
    // ops.
    blockPatterns.add<
        LowerThirdPartyBinaryElemwiseOp<RMO::ThirdPartyAddOp, RMO::MOAddOp>,
        LowerThirdPartyBinaryElemwiseOp<RMO::ThirdPartySubOp, RMO::MOSubOp>,
        LowerThirdPartyBinaryElemwiseOp<RMO::ThirdPartyMulOp, RMO::MOMulOp>,
        LowerThirdPartyBinaryElemwiseOp<RMO::ThirdPartyDivOp, RMO::MODivOp>,
        LowerThirdPartyBinaryElemwiseOp<RMO::ThirdPartyModOp, RMO::MOModOp>,
        LowerThirdPartyBinaryElemwiseOp<RMO::ThirdPartyPowOp, RMO::MOPowOp>,
        LowerThirdPartyBinaryElemwiseOp<RMO::ThirdPartyMinOp, RMO::MOMinOp>,
        LowerThirdPartyBinaryElemwiseOp<RMO::ThirdPartyMaxOp, RMO::MOMaxOp>,
        // Comparison ops lower the same as other elemwise ops.
        LowerThirdPartyBinaryComparisonOp<RMO::ThirdPartyEqualOp,
                                          RMO::MOEqualOp>,
        LowerThirdPartyBinaryComparisonOp<RMO::ThirdPartyNotEqualOp,
                                          RMO::MONotEqualOp>,
        LowerThirdPartyBinaryComparisonOp<RMO::ThirdPartyGreaterOp,
                                          RMO::MOGreaterOp>,
        LowerThirdPartyBinaryComparisonOp<RMO::ThirdPartyGreaterEqualOp,
                                          RMO::MOGreaterEqualOp>>(context);

    // Lowering third party specific operations.
    // They have more flexible semantics that separate them from standard rmo
    // ops.
    blockPatterns.add<
        LowerThirdPartyBinaryElemwiseOp<RMO::ThirdPartyAddOp, RMO::MOAddOp>,
        LowerThirdPartyBinaryElemwiseOp<RMO::ThirdPartySubOp, RMO::MOSubOp>,
        LowerThirdPartyBinaryElemwiseOp<RMO::ThirdPartyMulOp, RMO::MOMulOp>,
        LowerThirdPartyBinaryElemwiseOp<RMO::ThirdPartyDivOp, RMO::MODivOp>,
        LowerThirdPartyBinaryElemwiseOp<RMO::ThirdPartyModOp, RMO::MOModOp>,
        LowerThirdPartyBinaryElemwiseOp<RMO::ThirdPartyPowOp, RMO::MOPowOp>,
        LowerThirdPartyBinaryElemwiseOp<RMO::ThirdPartyMinOp, RMO::MOMinOp>,
        LowerThirdPartyBinaryElemwiseOp<RMO::ThirdPartyMaxOp, RMO::MOMaxOp>,
        // Comparison ops lower the same as other elemwise ops.
        LowerThirdPartyBinaryComparisonOp<RMO::ThirdPartyEqualOp,
                                          RMO::MOEqualOp>,
        LowerThirdPartyBinaryComparisonOp<RMO::ThirdPartyNotEqualOp,
                                          RMO::MONotEqualOp>,
        LowerThirdPartyBinaryComparisonOp<RMO::ThirdPartyGreaterOp,
                                          RMO::MOGreaterOp>,
        LowerThirdPartyBinaryComparisonOp<RMO::ThirdPartyGreaterEqualOp,
                                          RMO::MOGreaterEqualOp>>(context);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(blockPatterns))))
      return signalPassFailure();
  }
};

} // namespace
