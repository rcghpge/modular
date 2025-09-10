//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "GenericML/GraphCompiler/Conversion/RMOSupport/Broadcast.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/ShapeFunctions.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GenericML/GraphCompiler/RMODialect/RMOOps.h"
#include "GenericML/GraphCompiler/Transforms/RMODialect/RMOTransforms.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/DeviceSpecs.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace M;

namespace M::RMO {
#define GEN_PASS_DEF_LEGALIZERMOOPS
#include "GenericML/GraphCompiler/Transforms/RMODialect/RMOTransforms.h.inc"
} // namespace M::RMO

namespace {

TensorValue shapeToIndicesTensor(OpBuilder &builder, Location loc,
                                 ShapeAttr shape, DeviceRefAttr device) {
  auto rank = static_cast<int64_t>(shape.getRank());

  auto indicesType =
      TensorType::getVectorSI64Type(builder.getContext(), rank, device);

  auto shapeConstant = builder.create<MOSH::ParamToValueOp>(loc, shape);
  return builder.create<Shape::ToTensorOp>(loc, indicesType, shapeConstant);
}

// Return whether the shapes in `a` and `b` can be rebound to each other.
bool isCompatibleShape(TensorType a, TensorType b) {
  if (a.hasRank() && b.hasRank()) {
    if (a.getRank() != b.getRank())
      return false;

    // Cannot rebind two different integer dims
    for (size_t i = 0; i < a.getRank(); i++) {
      if (a.isIntegerDim(i) && b.isIntegerDim(i) &&
          a.getIntDim(i) != b.getIntDim(i))
        return false;
    }
  }

  return true;
}

template <typename T>
using ToTensorValueType = TensorValue;

/// Convenience function for promoting the dtype of a sequence of tensor values.
template <typename... Tensors>
static std::tuple<ToTensorValueType<Tensors>...>
promoteDTypes(OpBuilder &builder, Location loc, DType targetDType,
              Tensors... values) {
  auto promote = [&](auto value) -> TensorValue {
    auto tensor = cast<TensorValue>(value);

    if (tensor.getType().getDtype() != targetDType) {
      auto targetType = tensor.getType().cloneWith(targetDType);
      value = builder.create<RMO::MOCastOp>(loc, targetType, value);
    }
    return value;
  };

  return {cast<TensorValue>(promote(values))...};
}

/// NOTE: RMO's rebinds are asserting in nature!
std::pair<TensorValue, bool>
_rebindToShape(MO::GraphOp graphOp, TensorValue input, TensorType newTy,
               mlir::RewriterBase &rewriter,
               ArrayRef<ParamDeclAttr> newOutputParamDecl = {}) {
  auto res = rewriter.create<RMO::RebindTensorShapeOp>(
      input.getLoc(), newTy, input, newOutputParamDecl,
      rewriter.getStringAttr(""));

  return {res.getResult(), isCompatibleShape(input.getType(), newTy)};
}

TensorValue rebindToShape(MO::GraphOp graphOp, OpOperand &input,
                          TensorType newTy, mlir::RewriterBase &rewriter,
                          ArrayRef<ParamDeclAttr> newOutputParamDecl = {}) {

  TensorValue val;
  bool verifierSuccess;
  std::tie(val, verifierSuccess) =
      _rebindToShape(graphOp, cast<TensorValue>(input.get()), newTy, rewriter,
                     newOutputParamDecl);

  if (!verifierSuccess) {
    Operation *parent = input.getOwner();
    parent->emitError() << "[LegalizeRMOOps] RMO rebinding failed for op "
                           "in operand index #"
                        << input.getOperandNumber()
                        << " original and expected types are not compatible. "
                           "\n\toriginal type is     : "
                        << input.get().getType()
                        << "\n\tRMO expected type is : " << newTy;
  }

  return val;
}

TensorValue rebindToShape(MO::GraphOp graphOp, OpResult input, TensorType newTy,
                          mlir::RewriterBase &rewriter,
                          ArrayRef<ParamDeclAttr> newOutputParamDecl = {}) {
  TensorValue val;
  bool verifierSuccess;
  std::tie(val, verifierSuccess) = _rebindToShape(
      graphOp, cast<TensorValue>(input), newTy, rewriter, newOutputParamDecl);

  if (!verifierSuccess) {
    Operation *parent = input.getOwner();
    parent->emitError()
        << "[LegalizeRMOOps] RMO rebinding failed for op in result index #"
        << input.getResultNumber()
        << " original and expected types are not compatible. \n\toriginal type "
           "is     : "
        << input.getType() << "\n\tRMO expected type is : " << newTy;
  }

  return val;
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

/// Lower the given RMO::MOAnalogue op to its MO analogue given by `generator`.
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
  op.inferMOTypes(graphOp, moOperandTypes, moResultTypes, moInputParamDecls,
                  moOutputParamDecls);

  // Rebind inputs to what the MO op would expect
  SmallVector<Value> newOperands;
  for (size_t i = 0; i < numOperands; i++) {
    Value val = op->getOperand(i);
    Type targetType = moOperandTypes[i];
    TensorValue tensorVal = dyn_cast<TensorValue>(val);
    if (!tensorVal) {
      ASSERT_STREAM(targetType == val.getType(),
                    << "Disagreement on type for non-tensor argument.");
      newOperands.push_back(val);
      continue;
    }
    SmallVector<KGEN::ParamDeclAttr> &curInputDecls = moInputParamDecls[i];

    TensorValue reboundValue =
        rebindToShape(graphOp, op->getOpOperand(i),
                      cast<TensorType>(targetType), rewriter, curInputDecls);

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

    // For now, we don't have very strong verifiers around devices. To avoid
    // rebinds which change the device reference, copy the device in the
    // output annotation.
    newTensorValue.setType(
        newTensorValue.getType().cloneWith(targetType.getDeviceRef()));

    if (!newTensorValue) {
      rewriter.replaceAllUsesWith(oldValue, newValue);
      continue;
    }

    TensorValue reboundValue;
    if (numResults == 1) {
      reboundValue =
          rebindToShape(graphOp, newOp->getOpResult(i), targetType, rewriter,
                        op.getMOAnalogueOutputParamDecls());
    } else {
      // For multiple-output ops, op.getMOAnalogueOutputParamDecls() returns all
      // param declarations with all outputs. We need to filter and keep only
      // the param decls for every different output (to avoid param
      // redefinitions).
      llvm::SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
      getParamDeclsFor(targetType, op.getMOAnalogueOutputParamDecls(),
                       outputParamDecls);
      reboundValue = rebindToShape(graphOp, newOp->getOpResult(i), targetType,
                                   rewriter, outputParamDecls);
    }
    rewriter.replaceAllUsesWith(oldValue, reboundValue);
  }

  return success();
}

struct LegalizeRMOMOAnalogue
    : public OpInterfaceRewritePattern<RMO::MOAnalogue> {

  LegalizeRMOMOAnalogue(MLIRContext *ctx, mlir::PatternBenefit benefit = 0)
      : OpInterfaceRewritePattern<RMO::MOAnalogue>(ctx, benefit) {}

  LogicalResult matchAndRewrite(RMO::MOAnalogue op,
                                PatternRewriter &rewriter) const override {
    auto generator =
        [&op](PatternRewriter &rewriter, const Location &loc,
              ArrayRef<Type> resultTypes,
              ArrayRef<Value> operands) -> MO::ParamDeclarationInterface {
      // All rmo ops have the form "rmo.mo.opname".
      StringAttr name = rewriter.getStringAttr(op->getName().stripDialect());

      auto result = rewriter.create(loc, name, ValueRange(operands),
                                    TypeRange(resultTypes));

      NamedAttrList newAttrs(result->getAttrs());

      for (auto attr : op->getAttrs()) {
        if (attr.getName() != op.getMOAnalogueOutputParamDeclsAttrName())
          newAttrs.set(attr.getName(), attr.getValue());
      }

      rewriter.modifyOpInPlace(result, [&] { result->setAttrs(newAttrs); });
      return cast<MO::ParamDeclarationInterface>(result);
    };
    return lowerMOAnalogue(op, rewriter, generator);
  }
};

struct LegalizeMutableLoadOp : public OpRewritePattern<RMO::MOMutableLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::MOMutableLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto graphOp = op->getParentOfType<MO::GraphOp>();

    SmallVector<Type> moOperandTypes, moResultTypes;
    SmallVector<SmallVector<KGEN::ParamDeclAttr>> moInputParamDecls;
    SmallVector<SmallVector<KGEN::ParamDeclAttr>> moOutputParamDecls;

    RMO::MOAnalogue rma = op;
    rma.inferMOTypes(graphOp, moOperandTypes, moResultTypes, moInputParamDecls,
                     moOutputParamDecls);

    SmallVector<Value> moOperands;
    moOperands.push_back(op.getInBuffer());
    moOperands.push_back(op.getInChain());

    auto moOp = rewriter.create<MO::MutableLoadOp>(op->getLoc(), moResultTypes,
                                                   moOperands);
    moOp->setDiscardableAttrs(op->getRawDictionaryAttrs());
    OpResult outTensorRes = moOp->getOpResult(0);
    auto reboundOutput = rebindToShape(graphOp, outTensorRes,
                                       op.getOutTensor().getType(), rewriter);
    rewriter.replaceAllUsesWith(op.getOutTensor(), reboundOutput);
    rewriter.replaceAllUsesWith(op.getOutChain(), moOp.getOutChain());

    op->erase();

    return success();
  }
};

struct LegalizeMutableStoreOp : public OpRewritePattern<RMO::MOMutableStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::MOMutableStoreOp op,
                                PatternRewriter &rewriter) const override {

    auto graphOp = op->getParentOfType<MO::GraphOp>();

    SmallVector<Type> moOperandTypes, moResultTypes;
    SmallVector<SmallVector<KGEN::ParamDeclAttr>> moInputParamDecls;
    SmallVector<SmallVector<KGEN::ParamDeclAttr>> moOutputParamDecls;

    RMO::MOAnalogue rmaOp = op;
    rmaOp.inferMOTypes(graphOp, moOperandTypes, moResultTypes,
                       moInputParamDecls, moOutputParamDecls);

    SmallVector<Value> moOperands;

    auto reboundInTensor =
        rebindToShape(graphOp, op.getInTensorMutable(),
                      cast<TensorType>(moOperandTypes[1]), rewriter);

    moOperands.push_back(op.getInBuffer());
    moOperands.push_back(reboundInTensor);
    moOperands.push_back(op.getInChain());

    auto moOp = rewriter.create<MO::MutableStoreOp>(op->getLoc(), moResultTypes,
                                                    moOperands);
    moOp->setDiscardableAttrs(op->getRawDictionaryAttrs());
    rewriter.replaceAllUsesWith(op.getOutChain(), moOp.getOutChain());

    op->erase();

    return success();
  }
};

struct LegalizeTransferOp : public OpRewritePattern<RMO::MOTransferOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::MOTransferOp op,
                                PatternRewriter &rewriter) const override {
    // No dynamic dims permitted in RMO at this point: check that explicitly.
    [[maybe_unused]] TensorType inTy =
        cast<TensorType>(op.getInput().getType());
    [[maybe_unused]] TensorType outTy =
        cast<TensorType>(op.getResult().getType());
    ASSERT_STREAM(!inTy.hasUnknownShape(),
                  << "rmo.mo.transfer input must not have dynamic dims; got "
                  << inTy);
    ASSERT_STREAM(!outTy.hasUnknownShape(),
                  << "rmo.mo.transfer result must not have dynamic dims; got "
                  << outTy);

    rewriter.replaceOpWithNewOp<MO::TransferOp>(
        op, op.getInput(), outTy.getDeviceRef(), op.getInChain());

    return success();
  }
};

struct LegalizeMutableStoreSliceOp
    : public OpRewritePattern<RMO::MOMutableStoreSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::MOMutableStoreSliceOp op,
                                PatternRewriter &rewriter) const override {

    auto graphOp = op->getParentOfType<MO::GraphOp>();

    SmallVector<Type> moOperandTypes, moResultTypes;
    SmallVector<SmallVector<KGEN::ParamDeclAttr>> moInputParamDecls;
    SmallVector<SmallVector<KGEN::ParamDeclAttr>> moOutputParamDecls;

    RMO::MOAnalogue rmaOp = op;
    rmaOp.inferMOTypes(graphOp, moOperandTypes, moResultTypes,
                       moInputParamDecls, moOutputParamDecls);

    SmallVector<Value> moOperands;

    moOperands.push_back(op.getInBuffer());

    for (size_t i = 1; i < moOperandTypes.size() - 1; i++) {
      SmallVector<KGEN::ParamDeclAttr> &curInputDecls = moInputParamDecls[i];

      auto rebound = rebindToShape(graphOp, op->getOpOperand(i),
                                   cast<TensorType>(moOperandTypes[i]),
                                   rewriter, curInputDecls);

      moOperands.push_back(rebound);
    }

    moOperands.push_back(op.getInChain());

    auto moOp = rewriter.create<MO::MutableStoreSliceOp>(
        op->getLoc(), moResultTypes, moOperands, op->getAttrs());
    moOp->setDiscardableAttrs(op->getRawDictionaryAttrs());
    rewriter.replaceAllUsesWith(op.getOutChain(), moOp.getOutChain());

    op->erase();

    return success();
  }
};

struct LowerRebindTensorShapeOp
    : public OpRewritePattern<RMO::RebindTensorShapeOp> {
  using OpRewritePattern::OpRewritePattern;

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

    // Add assertion, then return a rebind.
    std::string fullMsg = "rebind shape check failed";
    llvm::raw_string_ostream ss(fullMsg);
    StringAttr msg = mlir::cast<StringAttr>(op.getMessageAttr());
    if (!msg.empty()) {
      ss << " with error: " << msg;
    }

    TypedAttr assertAttr = MO::AssertOp::emitStaticCall(
        KGEN::ParamDeclRefAttr::get(condAttr), fullMsg);

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
  using OpRewritePattern<RMOOpT>::OpRewritePattern;

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

    auto answer = rewriter.create<MOAnalogueOpT>(
        loc, ValueRange{xFinal, yFinal}, graphOp);
    TensorValue reboundAnswer =
        rebindToShape(graphOp, answer->getOpResult(0), op.getType(), rewriter,
                      op.getOutputParamDecls());

    rewriter.replaceAllUsesWith(op, reboundAnswer);
    return success();
  }
};

template <typename RMOOpT, typename MOAnalogueOpT>
struct LowerThirdPartyBinaryComparisonOp : public OpRewritePattern<RMOOpT> {
  using OpRewritePattern<RMOOpT>::OpRewritePattern;

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

    auto answer = rewriter.create<MOAnalogueOpT>(
        loc, ValueRange{xFinal, yFinal}, graphOp);
    TensorValue reboundAnswer =
        rebindToShape(graphOp, answer->getOpResult(0), op.getType(), rewriter,
                      op.getOutputParamDecls());

    rewriter.replaceAllUsesWith(op, reboundAnswer);
    return success();
  }
};

template <typename RMOOpT, typename MOAnalogueOpT>
struct LowerCompositeBinaryElemwiseOp : public OpRewritePattern<RMOOpT> {
  using OpRewritePattern<RMOOpT>::OpRewritePattern;

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

    auto answer = rewriter.create<MOAnalogueOpT>(
        loc, ValueRange{xFinal, yFinal}, graphOp);
    TensorValue reboundAnswer =
        rebindToShape(graphOp, answer->getOpResult(0), op.getType(), rewriter,
                      op.getOutputParamDecls());

    rewriter.replaceAllUsesWith(op, reboundAnswer);
    return success();
  }
};

template <typename RMOOpT, typename MOAnalogueOpT>
struct LowerCompositeBinaryComparisonOp : public OpRewritePattern<RMOOpT> {
  using OpRewritePattern<RMOOpT>::OpRewritePattern;

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
    ASSERT_STREAM(
        !promotedDtype.isError(),
        << "failed to calculate valid promotion dtype in lowering. This should "
           "be impossible due to it being calculated when inferring the return "
           "type"
        << promotedDtype.takeError());
    DType comparisonDtype = promotedDtype.takeValue();
    if (xDtype != comparisonDtype) {
      x = rewriter.create<RMO::MOCastOp>(
          loc, x.getType().cloneWith(comparisonDtype), x);
    }
    if (yDtype != comparisonDtype) {
      y = rewriter.create<RMO::MOCastOp>(
          loc, y.getType().cloneWith(comparisonDtype), y);
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

    auto answer = rewriter.create<MOAnalogueOpT>(
        loc, ValueRange{xFinal, yFinal}, graphOp);
    TensorValue reboundAnswer =
        rebindToShape(graphOp, answer->getOpResult(0), op.getType(), rewriter,
                      op.getOutputParamDecls());

    rewriter.replaceAllUsesWith(op, reboundAnswer);
    return success();
  }
};

struct LowerBroadcastLike : public OpRewritePattern<RMO::BroadcastLikeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::BroadcastLikeOp op,
                                PatternRewriter &rewriter) const override {
    auto graphOp = op->getParentOfType<MO::GraphOp>();
    Location loc = op->getLoc();

    TensorValue shapeLike = rewriter.create<RMO::MOShapeOfOp>(
        loc, ValueRange{op.getShapeLike()}, graphOp);
    auto answer = rewriter.create<RMO::MOBroadcastToOp>(
        loc, ValueRange{op.getInput(), shapeLike}, graphOp);
    TensorValue reboundAnswer =
        rebindToShape(graphOp, answer->getOpResult(0), op.getType(), rewriter,
                      op.getOutputParamDecls());

    rewriter.replaceAllUsesWith(op, reboundAnswer);
    return success();
  }
};

struct LowerConcat : public OpRewritePattern<RMO::ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::ConcatOp op,
                                PatternRewriter &rewriter) const override {

    // Create a tensor value for axis.
    Location loc = op->getLoc();
    int64_t axisInt = op.getAxis().getInt();
    TensorValue axis = rewriter.create<MO::ConstantOp>(loc, axisInt);

    // Replace the `rmo.concat` op with a new `mo.concat` op.
    TensorValue concat =
        rewriter.create<MO::ConcatOp>(loc, op.getResult().getType(), axis,
                                      op.getInputs(), op.getOutputParamDecls());

    // If there is a user specified rebind for the axis dim, generate the
    // required assert.
    if (TypedAttr outDimAttr =
            mlir::dyn_cast_or_null<TypedAttr>(op.getOutDimAttr())) {
      MO::GraphOp graphOp = op->getParentOfType<MO::GraphOp>();
      SmallVector<ParamBinding> paramBindings;
      ParamExprBuilder b(graphOp, loc, &paramBindings);

      int64_t rank = op.getResult().getType().getRank();
      axisInt = (axisInt < 0) ? axisInt + rank : axisInt;

      SmallVector<TypedAttr> inputDims;
      for (auto input : op.getOperands())
        inputDims.push_back(mlir::cast<TensorType>(input.getType())
                                .getTypedShapeAttr()[axisInt]);

      Var concatDimSum =
          b.createVar(KGEN::ParamOperatorAttr::get(KGEN::POC::Add, inputDims));
      Var newDim = b.createVar(outDimAttr);
      b.createAssert(concatDimSum == newDim,
                     "[concat] out_dim must be equivalent to the sum of the "
                     "input dims along the concat axis");

      // Add all ops for the assert to the graph.
      for (auto binding : paramBindings)
        binding.toOp(loc, rewriter);
    }

    rewriter.replaceOp(op, concat);

    return success();
  }
};

struct LowerReshape : public OpRewritePattern<RMO::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // If the newShape contains `-1`, at runtime, ensure all other dimensions
    // multiplied together do not equal 0. Otherwise, the shape of the `-1` dim
    // is not clear.
    ArrayRef<TypedAttr> dims = op.getNewShape().getValues();
    bool containsNegOne = llvm::any_of(dims, [](auto dim) {
      auto intDim = dyn_cast<IntegerAttr>(dim);
      return intDim && intDim.getSInt() == -1;
    });
    if (containsNegOne) {
      SmallVector<TypedAttr> nonNegOneDims;
      for (auto dim : dims) {
        auto intDim = dyn_cast<IntegerAttr>(dim);
        if (intDim && intDim.getSInt() == -1)
          continue;

        nonNegOneDims.push_back(dim);
      }

      Location loc = op.getLoc();
      MO::GraphOp graphOp = op->getParentOfType<MO::GraphOp>();

      SmallVector<ParamBinding> paramBindings;
      ParamExprBuilder b(graphOp, loc, &paramBindings);

      // Add a 1 to the shapes to multiply to ensure the denominator is valid
      SmallVector<TypedAttr> mulDims = nonNegOneDims;
      mulDims.emplace_back(ShapeAttr::getDim(getContext(), 1));
      Var denominator = b.createVar(
          KGEN::ParamOperatorAttr::get(KGEN::POC::MulNoWrap, mulDims));

      // Make sure we check for 0 denominator
      b.createAssert(denominator != 0,
                     "[reshape] cannot infer dimension when a specified output "
                     "dimension is 0");

      // Add all ops for the assert to the graph.
      for (auto binding : paramBindings)
        binding.toOp(loc, rewriter);
    }

    // Since the shape is already resolved and can't contain any unknown dims,
    // simply lower to static reshape.
    rewriter.replaceOpWithNewOp<MO::Static::ReshapeOp>(
        op, op.getResult().getType(), op.getInput());
    return success();
  }
};

struct LowerBroadcastTo : public OpRewritePattern<RMO::BroadcastToOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::BroadcastToOp op,
                                PatternRewriter &rewriter) const override {
    // Since the shape is already resolved and can't contain any unknown dims,
    // simply lower to static broadcast_to.
    rewriter.replaceOpWithNewOp<MO::Static::BroadcastToOp>(
        op, op.getResult().getType(), op.getInput());
    return success();
  }
};

struct LowerSigmoid : public OpRewritePattern<RMO::SigmoidOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::SigmoidOp op,
                                PatternRewriter &rewriter) const override {
    auto graphOp = op->getParentOfType<MO::GraphOp>();
    Location loc = op->getLoc();

    // Let's decompose the sigmoid operation into RMO analogue operations to
    // perform this calculation:
    //
    // 1/(1+e^-x)
    //

    TensorValue x = op.getInput();

    TensorValue one =
        rewriter.create<MO::ConstantOp>(loc, 1.0f, x.getType().getDeviceRef());

    TensorValue oneAsInputDtype =
        rewriter.create<RMO::MOCastOp>(loc, one, x.getType().getDtype());

    TensorValue tensorOne = RMO::broadcastTo(rewriter, loc, oneAsInputDtype, x);

    TensorValue negatedX =
        rewriter.create<RMO::MONegativeOp>(loc, ValueRange{x}, graphOp);
    TensorValue eToTheNegatedX =
        rewriter.create<RMO::MOExpOp>(loc, ValueRange{negatedX}, graphOp);

    TensorValue denominator = rewriter.create<RMO::MOAddOp>(
        loc, ValueRange{tensorOne, eToTheNegatedX}, graphOp);
    auto answer = rewriter.create<RMO::MODivOp>(
        loc, ValueRange{tensorOne, denominator}, graphOp);

    TensorValue reboundAnswer =
        rebindToShape(graphOp, answer->getOpResult(0), op.getType(), rewriter,
                      op.getOutputParamDecls());

    rewriter.replaceAllUsesWith(op, reboundAnswer);
    return success();
  }
};

struct LowerMatmul : public OpRewritePattern<RMO::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    TensorValue lhs = op.getInputX();
    TensorValue rhs = op.getInputY();
    TensorType lhsType = lhs.getType();
    TensorType rhsType = rhs.getType();
    ShapeAttr lhsShape = lhsType.getTypedShapeAttr();
    ShapeAttr rhsShape = rhsType.getTypedShapeAttr();

    size_t lhsRank = lhsShape.getRank();
    size_t rhsRank = rhsShape.getRank();
    size_t fullRank = std::max(size_t(2), std::max(lhsRank, rhsRank));

    // Deal with dtype casting if it is needed.
    auto lhsDtype = lhsType.getDtype();
    auto rhsDtype = rhsType.getDtype();
    auto resultDtype = op.getResult().getType().getDtype();
    if (lhsDtype != resultDtype) {
      lhsType = lhsType.cloneWith(resultDtype);
      lhs = rewriter.create<RMO::MOCastOp>(loc, lhsType, lhs);
    }
    if (rhsDtype != resultDtype) {
      rhsType = rhsType.cloneWith(resultDtype);
      rhs = rewriter.create<RMO::MOCastOp>(loc, rhsType, rhs);
    }

    // Reshape to ensure both tensors have same rank.
    auto [lhsReshapeTarget, rhsReshapeTarget] =
        expandMatmulInputsForBroadcasting(lhsShape, rhsShape);

    auto lhsReshapeShape = ShapeAttr::get(lhsReshapeTarget, lhsShape.getType());
    if (lhsReshapeShape != lhsShape)
      lhs = rewriter.create<MO::Static::ReshapeOp>(
          loc, lhsType.cloneWith(lhsReshapeShape), lhs);

    auto rhsReshapeShape = ShapeAttr::get(rhsReshapeTarget, rhsShape.getType());
    if (rhsReshapeShape != rhsShape)
      rhs = rewriter.create<MO::Static::ReshapeOp>(
          loc, rhsType.cloneWith(rhsReshapeShape), rhs);

    // Calculated broadcast shapes and broadcast outer dims.
    auto intDim1 = MOSH::ShapeAttr::getDim(lhsShape.getContext(), 1ll);
    SmallVector<TypedAttr> lhsBroadcastTarget{lhsReshapeTarget};
    SmallVector<TypedAttr> rhsBroadcastTarget{rhsReshapeTarget};
    for (size_t i = 0; i < fullRank - 2; ++i) {
      TypedAttr dim;
      if (lhsReshapeTarget[i] == intDim1)
        dim = rhsReshapeTarget[i];
      else
        dim = lhsReshapeTarget[i];

      lhsBroadcastTarget[i] = dim;
      rhsBroadcastTarget[i] = dim;
    }

    auto lhsBroadcastShape =
        ShapeAttr::get(lhsBroadcastTarget, lhsShape.getType());
    if (lhsBroadcastShape != lhsReshapeShape)
      lhs = rewriter.create<MO::Static::BroadcastToOp>(
          loc, lhsType.cloneWith(lhsBroadcastShape), lhs);

    auto rhsBroadcastShape =
        ShapeAttr::get(rhsBroadcastTarget, rhsShape.getType());
    if (rhsBroadcastShape != rhsReshapeShape)
      rhs = rewriter.create<MO::Static::BroadcastToOp>(
          loc, rhsType.cloneWith(rhsBroadcastShape), rhs);

    // Execute the actual matmul.
    SmallVector<TypedAttr> matmulTarget{lhsBroadcastTarget};
    matmulTarget[fullRank - 1] = rhsBroadcastTarget[fullRank - 1];
    auto matmulShape = ShapeAttr::get(matmulTarget, rhsShape.getType());
    auto matmulType = op.getType().cloneWith(matmulShape);
    TensorValue matmul;
    if (fullRank == 2)
      matmul = rewriter.create<MO::MatmulOp>(loc, matmulType, lhs, rhs);
    else
      matmul = rewriter.create<MO::BatchMatmulOp>(loc, matmulType, lhs, rhs);

    // Reshape to remove extra one dimensions if they were introduced by vector
    // inputs.
    if (matmulShape != op.getType().getTypedShapeAttr())
      matmul =
          rewriter.create<MO::Static::ReshapeOp>(loc, op.getType(), matmul);

    rewriter.replaceAllUsesWith(op, matmul);
    return success();
  }
};

struct LowerConv : public OpRewritePattern<RMO::ConvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::ConvOp conv,
                                PatternRewriter &rewriter) const override {
    Location loc = conv.getLoc();

    ShapeAttr strides = conv.getStrides();
    ShapeAttr paddings = conv.getPaddings();
    ShapeAttr dilations = conv.getDilations();
    int64_t groups = conv.getNumGroups();
    StringAttr inputLayout = conv.getInputLayoutAttr();

    auto resultType = conv.getType();

    // Promote inputs
    auto [promotedInput, promotedFilter] =
        promoteDTypes(rewriter, conv.getLoc(), resultType.getDtype(),
                      conv.getInput(), conv.getFilter());

    auto device = getHostDevice(conv.getContext());
    auto stridesConstant = shapeToIndicesTensor(rewriter, loc, strides, device);

    auto paddingConstant =
        shapeToIndicesTensor(rewriter, loc, paddings, device);

    auto dilationsConstant =
        shapeToIndicesTensor(rewriter, loc, dilations, device);

    auto groupsConstant = rewriter.create<MO::ConstantOp>(loc, groups, device);

    rewriter.replaceOpWithNewOp<RMO::MOConvOp>(
        conv, conv.getType(), promotedInput, promotedFilter, stridesConstant,
        dilationsConstant, paddingConstant, groupsConstant, inputLayout,
        rewriter.getStringAttr(""));
    return success();
  }
};

struct LowerConvTranspose : public OpRewritePattern<RMO::ConvTransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::ConvTransposeOp conv,
                                PatternRewriter &rewriter) const override {
    Location loc = conv.getLoc();

    ShapeAttr strides = conv.getStrides();
    ShapeAttr paddings = conv.getPaddings();
    ShapeAttr dilations = conv.getDilations();
    ShapeAttr outputPaddings = conv.getOutputPaddings();
    StringAttr inputLayout = conv.getInputLayoutAttr();

    TensorType resultType = conv.getType();

    // Promote inputs
    auto [promotedInput, promotedFilter] =
        promoteDTypes(rewriter, conv.getLoc(), resultType.getDtype(),
                      conv.getInput(), conv.getFilter());

    // DeviceRefAttr device = promotedInput.getType().getDeviceRef();
    auto device = getHostDevice(conv.getContext());
    TensorValue stridesConstant =
        shapeToIndicesTensor(rewriter, loc, strides, device);

    TensorValue paddingConstant =
        shapeToIndicesTensor(rewriter, loc, paddings, device);

    TensorValue outputPaddingConstant =
        shapeToIndicesTensor(rewriter, loc, outputPaddings, device);

    TensorValue dilationsConstant =
        shapeToIndicesTensor(rewriter, loc, dilations, device);

    rewriter.replaceOpWithNewOp<RMO::MOConvTransposeOp>(
        conv, conv.getType(), promotedInput, promotedFilter, stridesConstant,
        dilationsConstant, paddingConstant, outputPaddingConstant, inputLayout,
        rewriter.getStringAttr(""));
    return success();
  }
};

struct LowerAvgPool : public OpRewritePattern<RMO::AvgPoolOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::AvgPoolOp avg_pool,
                                PatternRewriter &rewriter) const override {
    Location loc = avg_pool.getLoc();

    ShapeAttr filterShape = avg_pool.getFilterShape();
    ShapeAttr strides = avg_pool.getStrides();
    ShapeAttr paddings = avg_pool.getPaddings();
    ShapeAttr dilations = avg_pool.getDilations();
    auto ceilMode = avg_pool.getCeilMode();
    auto countBoundary = avg_pool.getCountBoundary();
    TensorType resultType = avg_pool.getType();

    // Promote inputs
    auto [promotedInput] =
        promoteDTypes(rewriter, avg_pool.getLoc(), resultType.getDtype(),
                      avg_pool.getInput());

    auto device = getHostDevice(avg_pool.getContext());
    TensorValue filterShapeConstant =
        shapeToIndicesTensor(rewriter, loc, filterShape, device);

    TensorValue stridesConstant =
        shapeToIndicesTensor(rewriter, loc, strides, device);

    TensorValue paddingConstant =
        shapeToIndicesTensor(rewriter, loc, paddings, device);

    TensorValue dilationsConstant =
        shapeToIndicesTensor(rewriter, loc, dilations, device);

    if (ceilMode) {
      rewriter.replaceOpWithNewOp<RMO::MOAvgPoolCeilOp>(
          avg_pool, avg_pool.getType(), promotedInput, filterShapeConstant,
          stridesConstant, dilationsConstant, paddingConstant, countBoundary);
    } else {
      rewriter.replaceOpWithNewOp<RMO::MOAvgPoolOp>(
          avg_pool, avg_pool.getType(), promotedInput, filterShapeConstant,
          stridesConstant, dilationsConstant, paddingConstant, countBoundary);
    }

    return success();
  }
};

struct LowerMaxPool : public OpRewritePattern<RMO::MaxPoolOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::MaxPoolOp max_pool,
                                PatternRewriter &rewriter) const override {
    Location loc = max_pool.getLoc();

    ShapeAttr filterShape = max_pool.getFilterShape();
    ShapeAttr strides = max_pool.getStrides();
    ShapeAttr paddings = max_pool.getPaddings();
    ShapeAttr dilations = max_pool.getDilations();
    auto ceilMode = max_pool.getCeilMode();

    TensorType resultType = max_pool.getType();

    // Promote inputs
    auto [promotedInput] =
        promoteDTypes(rewriter, max_pool.getLoc(), resultType.getDtype(),
                      max_pool.getInput());

    auto device = getHostDevice(max_pool.getContext());
    TensorValue filterShapeConstant =
        shapeToIndicesTensor(rewriter, loc, filterShape, device);

    TensorValue stridesConstant =
        shapeToIndicesTensor(rewriter, loc, strides, device);

    TensorValue paddingConstant =
        shapeToIndicesTensor(rewriter, loc, paddings, device);

    TensorValue dilationsConstant =
        shapeToIndicesTensor(rewriter, loc, dilations, device);

    if (ceilMode) {
      rewriter.replaceOpWithNewOp<RMO::MOMaxPoolCeilOp>(
          max_pool, max_pool.getType(), promotedInput, filterShapeConstant,
          stridesConstant, dilationsConstant, paddingConstant);
    } else {
      rewriter.replaceOpWithNewOp<RMO::MOMaxPoolOp>(
          max_pool, max_pool.getType(), promotedInput, filterShapeConstant,
          stridesConstant, dilationsConstant, paddingConstant);
    }

    return success();
  }
};

struct LowerSelect : public OpRewritePattern<RMO::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::SelectOp op,
                                PatternRewriter &rewriter) const override {
    auto graphOp = op->template getParentOfType<MO::GraphOp>();
    Location loc = op->getLoc();

    TensorValue cond = op.getCond();
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

    TensorValue condFinal = cond;
    TensorValue xFinal = x;
    TensorValue yFinal = y;
    if (x.getType().getTypedShapeAttr() != y.getType().getTypedShapeAttr() ||
        x.getType().getTypedShapeAttr() != cond.getType().getTypedShapeAttr()) {
      TensorValue condShape =
          rewriter.create<RMO::MOShapeOfOp>(loc, ValueRange{cond}, graphOp);
      TensorValue xShape =
          rewriter.create<RMO::MOShapeOfOp>(loc, ValueRange{x}, graphOp);
      TensorValue yShape =
          rewriter.create<RMO::MOShapeOfOp>(loc, ValueRange{y}, graphOp);

      TensorValue xyShape = rewriter.create<RMO::MOBroadcastShapeOp>(
          loc, ValueRange{xShape, yShape}, graphOp);
      TensorValue commonShape = rewriter.create<RMO::MOBroadcastShapeOp>(
          loc, ValueRange{condShape, xyShape}, graphOp);

      TensorValue condBcast = rewriter.create<RMO::MOBroadcastToOp>(
          loc, ValueRange{cond, commonShape}, graphOp);
      TensorValue xBcast = rewriter.create<RMO::MOBroadcastToOp>(
          loc, ValueRange{x, commonShape}, graphOp);
      TensorValue yBcast = rewriter.create<RMO::MOBroadcastToOp>(
          loc, ValueRange{y, commonShape}, graphOp);
      condFinal = condBcast;
      xFinal = xBcast;
      yFinal = yBcast;
    }

    auto answer = rewriter.create<RMO::MOSelectOp>(
        loc, ValueRange{condFinal, xFinal, yFinal}, graphOp);
    TensorValue reboundAnswer =
        rebindToShape(graphOp, answer->getOpResult(0), op.getType(), rewriter,
                      op.getOutputParamDecls());

    rewriter.replaceAllUsesWith(op, reboundAnswer);
    return success();
  }
};

struct LowerTopK : public OpRewritePattern<RMO::TopKOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::TopKOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = getContext();

    // Get the device from the input tensor
    TensorValue input = op.getInput();
    DeviceRefAttr deviceRef = input.getType().getDeviceRef();
    if (!deviceRef)
      return op.emitError() << "rmo.top_k lowering requires input tensor to "
                               "have an assigned device";

    // Marshal out dimension information for assertion.
    auto inputType = input.getType();
    auto inputShape = inputType.getTypedShapeAttr();
    const int64_t kVal = op.getK();
    const int64_t rank = inputShape.getRank();
    // Normalize axis. `inferReturnTypes` checks -rank <= axis < rank already.
    int64_t axisVal = op.getAxis();
    axisVal = axisVal < 0 ? axisVal + rank : axisVal;

    // Add parametric check for k <= axis dimension
    MO::GraphOp graphOp = op->getParentOfType<MO::GraphOp>();
    SmallVector<ParamBinding> paramBindings;
    ParamExprBuilder b(graphOp, loc, &paramBindings);

    // Get the dimension along the axis (could be symbolic).
    TypedAttr axisDimAttr = inputShape[axisVal];
    Var axisDim = b.createVar(axisDimAttr);
    Var kVar = b.createVar(
        IntegerAttr::get(rewriter.getIntegerType(64, /*isSigned=*/true), kVal));

    // Create assertion that k <= dimension size
    b.createAssert(kVar <= axisDim,
                   "k value exceeds dimension size along specified axis");

    // Add parameter bindings to the graph.
    for (const ParamBinding &binding : paramBindings)
      binding.toOp(loc, rewriter);

    // Convert attributes to tensor constants expected by `mo.top_k`.
    auto k =
        rewriter.create<MO::ConstantOp>(loc, op.getK(), getHostDevice(ctx));
    auto axis =
        rewriter.create<MO::ConstantOp>(loc, op.getAxis(), getHostDevice(ctx));
    auto sorted = rewriter.create<MO::ConstantOp>(loc, /*value=*/true,
                                                  getHostDevice(ctx));

    // Create the MO::TopKOp with proper result types
    auto topkOp = rewriter.create<MO::TopKOp>(
        op.getLoc(), op.getValues().getType(), op.getIndices().getType(), input,
        k, axis, sorted, op.getOutputParamDecls());

    rewriter.replaceOp(op, topkOp->getResults());
    return success();
  }
};

struct LowerSlice : public OpRewritePattern<RMO::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::SliceOp op,
                                PatternRewriter &rewriter) const override {
    // Note that rmo.slice has special restrictions compared to mo.slice:
    // - Requires start, stop to be in [-dim, dim) so that there is no clamping.
    // - Requires steps to be a statically known positive integer.
    // These are to simplify rmo.slice's shape function, reducing bloat in
    // algebraic dims computed in the rmo.slice type.
    Location loc = op.getLoc();
    MLIRContext *ctx = getContext();
    TensorValue input = op.getInput();
    auto inputShape = cast<MOSH::ShapeAttr>(input.getType().getShapeAttr());
    ShapeAttr startsAttr = op.getStarts();
    ShapeAttr stopsAttr = op.getStops();
    ShapeAttr stepsAttr = op.getSteps();
    auto maybeSteps = MO::tryGetConstantIntVector(stepsAttr);
    if (!maybeSteps ||
        llvm::any_of(*maybeSteps, [](int64_t step) { return step == 0; }))
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("rmo.slice expects static steps != 0 but got {0}",
                             stepsAttr));

    // Convert start/stop/step to tensors expected by mo.slice.
    auto indicesType = TensorType::getVectorSI64Type(
        ctx, startsAttr.getRank(),
        // Always put indices on CPU, as expected by the `mo.slice` kernel.
        DeviceRefAttr::get(ctx, kCPULabel, /*id=*/0));
    auto toTensorOp = [loc = loc, indicesType = indicesType,
                       &rewriter](ShapeAttr attr) -> Value {
      auto constOp = rewriter.create<MOSH::ParamToValueOp>(loc, attr);
      return rewriter.create<Shape::ToTensorOp>(loc, indicesType, constOp);
    };

    auto starts = toTensorOp(startsAttr);
    auto stops = toTensorOp(stopsAttr);
    auto steps = toTensorOp(stepsAttr);

    MO::GraphOp graphOp = op->getParentOfType<MO::GraphOp>();
    SmallVector<ParamBinding> paramBindings;
    ParamExprBuilder b(graphOp, loc, &paramBindings);
    for (size_t axis = 0; axis < input.getType().getRank(); axis++) {
      // NOTE: checking for:
      // - step != 0
      // - step > 0 => start <= stop
      // - step < 0 => start >= stop
      // is handled in the mo.slice shape function.

      Var start = b.createVar(startsAttr[axis]);
      Var stop = b.createVar(stopsAttr[axis]);
      auto maybeStartConst = tryGetConstantInt(startsAttr[axis]);
      if (!maybeStartConst)
        b.createAssert(start >= 0,
                       "[rmo.slice] requires positive start when symbolic");
      auto maybeStopConst = tryGetConstantInt(stopsAttr[axis]);
      if (!maybeStopConst)
        b.createAssert(stop >= 0,
                       "[rmo.slice] requires positive stop when symbolic");

      // Restrict start/stop to within [-dim, dim).
      Var dim = b.createVar(inputShape[axis]);
      b.createAssert(start >= -dim && start < dim,
                     "[rmo.slice] requires start in [-dim, dim)");
      b.createAssert(stop >= -dim && stop < dim,
                     "[rmo.slice] requires stop in [-dim, dim)");
    }

    // Create the mo.slice op.
    auto res = rewriter.create<MO::SliceOp>(op->getLoc(), op.getType(), input,
                                            starts, stops, steps);
    res->setDiscardableAttrs(op->getRawDictionaryAttrs());
    rewriter.replaceAllOpUsesWith(op, res);
    return success();
  }
};

struct LowerShapeToTensor : public OpRewritePattern<RMO::ShapeToTensorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::ShapeToTensorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TypedAttr shape = op.getShape();
    TensorType resTy = op.getType();

    auto shConst = rewriter.create<MOSH::ParamToValueOp>(loc, shape);
    rewriter.replaceOpWithNewOp<Shape::ToTensorOp>(op, resTy, shConst);
    return success();
  }
};

struct LowerThirdPartyPadConstantOp
    : public OpRewritePattern<RMO::ThirdPartyPadConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::ThirdPartyPadConstantOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TensorValue inputTensor = op.getInput();
    TensorType inputType = inputTensor.getType();
    TensorType resultType = op.getType();

    // We enforce some invariants. We could also allow si32 pad sizes, but onnx
    // and torch use si64. So we keep things simple now.
    if (!inputType.hasRank() || !resultType.hasRank()) {
      return rewriter.notifyMatchFailure(
          loc, "rmo.3p.pad.constant expects ranked input and result");
    }
    TensorValue signedPadTensor = op.getPaddings();
    TensorType padType = signedPadTensor.getType();
    if (!padType.hasRank() || padType.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          loc, "rmo.3p.pad.constant padding must be a vector of static shape");
    }
    int64_t numDims = inputType.getRank();
    if (padType.getDtype() != DType::si64) {
      return rewriter.notifyMatchFailure(
          loc, "rmo.3p.pad.constant paddings must be si64");
    }
    DeviceRefAttr deviceRef = inputType.getDeviceRef();
    if (deviceRef != padType.getDeviceRef()) {
      op.emitError("Devices don't match: deviceRef: ")
          << deviceRef << " padType.getDeviceRef(): " << padType.getDeviceRef();
      assert(false && "devices don't match");
      return failure();
    }

    // We will split the paddings into negative (pure slice) and positive (pure
    // pad) components.
    auto zeros = rewriter.create<MO::ConstantOp>(
        loc, SmallVector<int64_t>(numDims * 2, 0), deviceRef);
    auto negPadTensor = rewriter.create<RMO::MOMinOp>(loc, zeros.getType(),
                                                      signedPadTensor, zeros);

    // The negative pads need to be split into begin (even indices) and end (odd
    // indices) pad sizes. We will do this with two gathers.
    SmallVector<int64_t> oddIndices;
    SmallVector<int64_t> evenIndices;
    oddIndices.reserve(numDims);
    evenIndices.reserve(numDims);
    for (int64_t r = 0; r < numDims; ++r) {
      evenIndices.push_back(2 * r);
      oddIndices.push_back(2 * r + 1);
    }
    auto oddConst = rewriter.create<MO::ConstantOp>(loc, oddIndices, deviceRef);
    auto evenConst =
        rewriter.create<MO::ConstantOp>(loc, evenIndices, deviceRef);

    // To get the end indices, we need to take the negative indices and
    // increment them by the sizes of the dimensions, i.e. add the shape to it.
    // This is important because a zero padding size would not be the correct
    // end index.
    MO::TensorType oldPadType = signedPadTensor.getType();
    MO::TensorType sliceIdxType =
        oldPadType.cloneWith({numDims}, oldPadType.getElementType());
    auto shapeOf =
        rewriter.create<RMO::MOShapeOfOp>(loc, sliceIdxType, inputTensor);
    auto zero = rewriter.create<MO::ConstantOp>(loc, (int64_t)0, deviceRef);
    MO::TensorValue end = rewriter.create<RMO::MOGatherOp>(
        loc, sliceIdxType, negPadTensor, oddConst, zero);
    end = rewriter.create<RMO::MOAddOp>(loc, sliceIdxType, end, shapeOf);

    // A negative padding in the beginning is always equivalent to skipping
    // elements, so we multiply by -1 to get the slice start indices.
    auto negOnes = rewriter.create<MO::ConstantOp>(
        loc, SmallVector<int64_t>(numDims, -1), deviceRef);
    MO::TensorValue start = rewriter.create<RMO::MOGatherOp>(
        loc, sliceIdxType, negPadTensor, evenConst, zero);
    start = rewriter.create<RMO::MOMulOp>(loc, sliceIdxType, start, negOnes);

    // We can now create a slice which will apply the negative pad sizes. This
    // slice uses the trivial step, but its shape must be dynamic.
    MO::GraphOp graphOp = op->getParentOfType<MO::GraphOp>();
    SmallVector<KGEN::ParamDeclAttr> newDimDecls;
    SmallVector<TypedAttr> newShapeDims;
    for (int64_t r = 0; r < numDims; ++r) {
      KGEN::ParamDeclAttr newDecl = graphOp.getNewDimParam("RMO_NEG_PAD");
      newDimDecls.push_back(newDecl);
      newShapeDims.emplace_back(KGEN::ParamDeclRefAttr::get(newDecl));
    }
    auto step = rewriter.create<MO::ConstantOp>(
        loc, SmallVector<int64_t>(numDims, 1), deviceRef);
    auto slicedInput = rewriter.create<RMO::MOSliceOp>(
        loc, inputType.cloneWith(newShapeDims), inputTensor, start, end, step,
        newDimDecls);

    // Hurray, we have applied the negative pads. Extracting the positive pad
    // sizes is straightforward. Note that because we separate negative and
    // positive pad sizes, they never overlap.
    auto posPadTensor = rewriter.create<RMO::MOMaxOp>(loc, zeros.getType(),
                                                      signedPadTensor, zeros);

    // We can now apply the positive pad sizes to the sliced input.
    rewriter.replaceOpWithNewOp<RMO::MOPadConstantOp>(
        op, resultType, slicedInput, posPadTensor, op.getValue(),
        op.getOutputParamDeclsAttr());
    return success();
  }
};

struct LegalizeRMOOps
    : public M::RMO::impl::LegalizeRMOOpsBase<LegalizeRMOOps> {
  using LegalizeRMOOpsBase::LegalizeRMOOpsBase;
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet blockPatterns(context);

    // Default lowering of MOAnalogue Operations.
    // Setting benefit to zero so that more specific patterns may override.
    blockPatterns.add<LegalizeRMOMOAnalogue>(context, /*benefit=*/0);

    // Override for rmo.mo.transfer to handle chain and avoid param decls.
    blockPatterns.add<LegalizeTransferOp>(context);

    blockPatterns.add<LegalizeMutableLoadOp, LegalizeMutableStoreOp,
                      LegalizeMutableStoreSliceOp>(context);

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
        LowerCompositeBinaryComparisonOp<RMO::AndOp, RMO::MOAndOp>,
        LowerCompositeBinaryComparisonOp<RMO::OrOp, RMO::MOOrOp>,
        LowerCompositeBinaryComparisonOp<RMO::XorOp, RMO::MOXorOp>,
        LowerBroadcastLike, LowerConv, LowerConvTranspose, LowerAvgPool,
        LowerMaxPool, LowerSigmoid, LowerConcat, LowerReshape, LowerMatmul,
        LowerBroadcastTo, LowerSelect, LowerTopK, LowerSlice,
        LowerShapeToTensor>(context);

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
                                          RMO::MOGreaterEqualOp>,
        LowerThirdPartyPadConstantOp>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(blockPatterns))))
      return signalPassFailure();
  }
};

} // namespace
