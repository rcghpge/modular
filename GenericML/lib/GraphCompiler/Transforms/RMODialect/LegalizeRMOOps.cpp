//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

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
      GENERICML_ASSERT(targetType == val.getType())
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
    Type targetType = op->getResultTypes()[i];

    TensorValue newTensorValue = dyn_cast<TensorValue>(newValue);
    if (!newTensorValue) {
      rewriter.replaceAllUsesWith(oldValue, newValue);
      continue;
    }

    // NOTE: this may not work for multiple return types.
    TensorValue reboundValue =
        rebindToShape(graphOp, newTensorValue, cast<TensorType>(targetType),
                      rewriter, op.getMOAnalogueOutputParamDecls());
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
          result->setAttr(attr.getName(), attr.getValue());
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
    TypedAttr assertAttr =
        MOSH::AssertOp::emitStaticCall(KGEN::ParamDeclRefAttr::get(condAttr),
                                       "RMO rebind shape check failed.");

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

// This is an example of how a "composite RMO op" would work.
struct LowerBroadcastedAddOp : public OpRewritePattern<RMO::BroadcastAddOp> {
  LowerBroadcastedAddOp(MLIRContext *ctx) : OpRewritePattern(ctx, 1) {}
  LogicalResult matchAndRewrite(RMO::BroadcastAddOp op,
                                PatternRewriter &rewriter) const override {
    auto graphOp = op->getParentOfType<MO::GraphOp>();

    TensorValue x = op.getInputX();
    TensorValue y = op.getInputY();

    Location loc = op->getLoc();
    TensorValue shapeX =
        rewriter.create<RMO::MOShapeOfOp>(loc, ValueRange{x}, graphOp);
    TensorValue shapeY =
        rewriter.create<RMO::MOShapeOfOp>(loc, ValueRange{y}, graphOp);
    TensorValue commonShape = rewriter.create<RMO::MOBroadcastShapeOp>(
        loc, ValueRange{shapeX, shapeY}, graphOp);

    TensorValue bcastX = rewriter.create<RMO::MOBroadcastToOp>(
        loc, ValueRange{x, commonShape}, graphOp);
    TensorValue bcastY = rewriter.create<RMO::MOBroadcastToOp>(
        loc, ValueRange{y, commonShape}, graphOp);

    TensorValue answer =
        rewriter.create<RMO::MOAddOp>(loc, ValueRange{bcastX, bcastY}, graphOp);
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
        LegalizeRMOMOAnalogue<RMO::MOSigmoidOp, MO::SigmoidOp>,
        LegalizeRMOMOAnalogue<RMO::MOSinOp, MO::SinOp>,
        LegalizeRMOMOAnalogue<RMO::MOSqrtOp, MO::SqrtOp>,
        LegalizeRMOMOAnalogue<RMO::MOTanhOp, MO::TanhOp>,
        LegalizeRMOMOAnalogue<RMO::MOTruncOp, MO::TruncOp>,
        // Elementwise binary ops
        LegalizeRMOMOAnalogue<RMO::MOAddOp, MO::AddOp>,
        LegalizeRMOMOAnalogue<RMO::MOMulOp, MO::MulOp>,
        LegalizeRMOMOAnalogue<RMO::MOModOp, MO::ModOp>,
        LegalizeRMOMOAnalogue<RMO::MOMinOp, MO::MinOp>,
        LegalizeRMOMOAnalogue<RMO::MOMaxOp, MO::MaxOp>,
        LegalizeRMOMOAnalogue<RMO::MODivOp, MO::DivOp>,
        LegalizeRMOMOAnalogue<RMO::MOPowOp, MO::PowOp>,
        LegalizeRMOMOAnalogue<RMO::MOSubOp, MO::SubOp>,
        // Comparison binary ops
        LegalizeRMOMOAnalogue<RMO::MOEqualOp, MO::EqualOp>,
        LegalizeRMOMOAnalogue<RMO::MOGreaterOp, MO::GreaterOp>,
        LegalizeRMOMOAnalogue<RMO::MOGreaterEqualOp, MO::GreaterEqualOp>,
        LegalizeRMOMOAnalogue<RMO::MONotEqualOp, MO::NotEqualOp>,
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
        LegalizeRMOMOAnalogue<RMO::MOResizeNearestOp, MO::ResizeNearestOp>>(
        context);

    // Lowering rebind
    if (!skipRMORebind)
      blockPatterns.add<LowerRebindTensorShapeOp>(context);

    // Lowering composite operations
    blockPatterns.add<LowerBroadcastedAddOp>(context);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(blockPatterns))))
      return signalPassFailure();
  }
};

} // namespace
