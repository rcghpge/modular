//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

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
                  op->getResults(), moOperandTypes, moResultTypes,
                  moInputParamDecls, moOutputParamDecls);

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
        [](PatternRewriter &rewriter, const Location &loc,
           ArrayRef<Type> resultTypes,
           ArrayRef<Value> operands) -> MO::ParamDeclarationInterface {
      auto result = rewriter.create<MOOpT>(loc, resultTypes, operands);
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
        rewriter.create<RMO::ShapeOfOp>(loc, ValueRange{x}, graphOp);
    TensorValue shapeY =
        rewriter.create<RMO::ShapeOfOp>(loc, ValueRange{y}, graphOp);
    TensorValue commonShape = rewriter.create<RMO::BroadcastShapeOp>(
        loc, ValueRange{shapeX, shapeY}, graphOp);

    TensorValue bcastX = rewriter.create<RMO::BroadcastToOp>(
        loc, ValueRange{x, commonShape}, graphOp);
    TensorValue bcastY = rewriter.create<RMO::BroadcastToOp>(
        loc, ValueRange{y, commonShape}, graphOp);

    TensorValue answer =
        rewriter.create<RMO::AddOp>(loc, ValueRange{bcastX, bcastY}, graphOp);
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
    blockPatterns
        .add<LegalizeRMOMOAnalogue<RMO::AddOp, MO::AddOp>,
             LegalizeRMOMOAnalogue<RMO::MulOp, MO::MulOp>,
             LegalizeRMOMOAnalogue<RMO::ModOp, MO::ModOp>,
             LegalizeRMOMOAnalogue<RMO::MinOp, MO::MinOp>,
             LegalizeRMOMOAnalogue<RMO::MaxOp, MO::MaxOp>,
             LegalizeRMOMOAnalogue<RMO::DivOp, MO::DivOp>,
             LegalizeRMOMOAnalogue<RMO::PowOp, MO::PowOp>,
             LegalizeRMOMOAnalogue<RMO::SubOp, MO::SubOp>,
             LegalizeRMOMOAnalogue<RMO::ShapeOfOp, MO::ShapeOfOp>,
             LegalizeRMOMOAnalogue<RMO::BroadcastShapeOp, MO::BroadcastShapeOp>,
             LegalizeRMOMOAnalogue<RMO::ReshapeOp, MO::ReshapeOp>,
             LegalizeRMOMOAnalogue<RMO::BroadcastToOp, MO::BroadcastToOp>,
             LegalizeRMOMOAnalogue<RMO::CastOp, MO::CastOp>>(context);

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
