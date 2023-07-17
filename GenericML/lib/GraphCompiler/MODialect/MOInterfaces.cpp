//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MO dialect interfaces.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/PatternMatch.h>

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// MatMulLike
//===----------------------------------------------------------------------===//

LogicalResult impl::verifyMatchingRanks(ArrayRef<TensorValue> values) {
  SmallVector<size_t> staticRanks;
  for (auto v : values)
    if (TensorType t = v.getType(); t.hasRank())
      staticRanks.emplace_back(t.getRank());

  bool allStaticRanks = staticRanks.size() == values.size();
  if (allStaticRanks)
    return success(llvm::all_equal(staticRanks));

  bool allDynamicRanks = staticRanks.empty();
  return success(allDynamicRanks);
}

LogicalResult impl::checkMatmulCompatibility(Operation *op, TensorValue A,
                                             TensorValue B, TensorValue C) {
  // We check that statically known ranks match each other (implicit
  // broadcasting is not supported).
  SmallVector<TensorValue> tensors{A, B, C};

  if (failed(verifyMatchingRanks(tensors))) {
    return op->emitOpError(
        "input and output ranks must be all dynamic, or all static and match");
  }

  TensorType aType = A.getType();
  TensorType bType = B.getType();

  // If either input is dynamically ranked, we don't try to verify.
  // When partially known ranks are supported we might refine this.
  if (!aType.hasRank() || !bType.hasRank())
    return success();

  if (aType.getRank() < 2) {
    return op->emitOpError(
               "expected input tensors to have at least rank 2, got ")
           << aType.getRank();
  }

  // If either inputs are dynamically shaped, we don't try to verify.
  if (!aType.hasStaticShape() || !bType.hasStaticShape())
    return success();

  // The op created from the framework should have no attributes and we should
  // verify those. However we can attach extra attributes to it in the lowering.
  // They will change some of the shapes (i.e allowing values to be transposed
  // implicitly).
  if (auto matmul = dyn_cast<MatmulOp>(op)) {
    if (matmul.getTransposeB().value_or(false) ||
        matmul.getPackedB().value_or(false))
      return success();
  }

  ArrayRef<TypedAttr> aMatShape = aType.getDims().take_back(2);
  ArrayRef<TypedAttr> bMatShape = bType.getDims().take_back(2);
  if (aMatShape[1] != bMatShape[0])
    return op->emitOpError("incompatible input/result matmul shapes");

  ArrayRef<TypedAttr> aMatBatches = aType.getDims().drop_back(2);
  ArrayRef<TypedAttr> bMatBatches = bType.getDims().drop_back(2);
  if (!aMatBatches.equals(bMatBatches))
    return op->emitOpError("incompatible input/result batch dimensions");

  TensorType resType = C.getType();

  // If the result is dynamically ranked/shaped, we defer to shape inference.
  if (!resType.hasStaticShape())
    return success();

  ArrayRef<TypedAttr> resMatShape = resType.getDims().take_back(2);
  if (resMatShape[0] != aMatShape[0] || resMatShape[1] != bMatShape[1])
    return op->emitOpError("incompatible input/result matmul shapes");

  ArrayRef<TypedAttr> resMatBatches = resType.getDims().drop_back(2);
  if (!aMatBatches.equals(resMatBatches))
    return op->emitOpError("incompatible input/result batch dimensions");

  return success();
}

LogicalResult impl::verifyMatmulLike(MatmulLike op) {
  return checkMatmulCompatibility(op.getOperation(), op.getTensorA(),
                                  op.getTensorB(), op.getTensorResult());
}

//===----------------------------------------------------------------------===//
// Reduction
//===----------------------------------------------------------------------===//

LogicalResult impl::verifyReduction(Reduction op) {
  // TODO: test
  if (failed(verifyMatchingRanks({op.getInput(), op.getResult()})))
    return op.emitOpError(
        "input and ranks ranks must be both dynamic, or both static and match");

  TensorValue axisArr = op.getAxis();
  TensorType inputTy = op.getInput().getType();

  std::optional<int64_t> maybeAxis = tryGetConstantInt(axisArr);
  if (!maybeAxis.has_value() || !inputTy.hasRank())
    return success();

  int64_t axis = maybeAxis.value();
  int64_t rank = inputTy.getRank();
  if (axis < -rank || axis > rank - 1) {
    return op.emitOpError() << "axis must be between [-" << rank << ", "
                            << (rank - 1) << "] found " << axis;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SlidingWindow
//===----------------------------------------------------------------------===//

LogicalResult SlidingWindow::verifyWindowParamShape(TensorValue windowParam,
                                                    StringRef name) {
  auto input =
      cast<MOTensorOpInterface>(this->getOperation()).getInputTensor(0);
  if (TensorType inputTy = input.getType(); inputTy.hasRank()) {
    if (TensorType windowParamTy = windowParam.getType();
        windowParamTy.hasStaticShape())
      if (static_cast<size_t>(windowParamTy.getIntDim(0)) != inputTy.getRank())
        return emitError(name) << " with static size must match rank of input";
  }
  return success();
}

LogicalResult
SlidingWindow::verifyWindowParamSizes(ArrayRef<TensorValue> values) {
  SmallPtrSet<Attribute, 4> sizes;
  for (auto v : values)
    if (TensorType t = v.getType(); t.hasRank())
      sizes.insert(t.getDim(0));

  if (sizes.size() > 1)
    return emitError("window parameter sizes must match if static");

  return success();
}

LogicalResult impl::verifySlidingWindow(SlidingWindow op) {
  if (failed(op.verifyWindowParamShape(op.getStrides(), "strides")))
    return failure();
  if (failed(op.verifyWindowParamShape(op.getDilations(), "dilations")))
    return failure();

  auto moOp = cast<MOTensorOpInterface>(op.getOperation());
  if (failed(verifyMatchingRanks(
          {moOp.getInputTensor(0), moOp.getOutputTensor(0)}))) {
    return op.emitError(
        "input and output ranks must be all dynamic, or all static and match");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MOSameOperandsAndResultElementType
//===----------------------------------------------------------------------===//

/// Get the element type of a tensor value.
static Type getElementType(Value v) {
  return v.getType().cast<MO::TensorType>().getElementType();
}

/// This follows the builtin verifySameOperandsAndResultElementType.
LogicalResult impl::verifyMOSameOperandsAndResultElementType(Operation *op) {
  if (failed(mlir::OpTrait::impl::verifyAtLeastNOperands(op, 1)) ||
      failed(mlir::OpTrait::impl::verifyAtLeastNResults(op, 1)))
    return failure();

  auto elementType = getElementType(op->getResult(0));

  // Verify result element type matches first result's element type.
  for (auto result : llvm::drop_begin(op->getResults(), 1)) {
    if (getElementType(result) != elementType) {
      return op->emitOpError(
          "requires the same element type for all operands and results");
    }
  }

  // Verify operand's element type matches first result's element type.
  for (auto operand : op->getOperands()) {
    if (getElementType(operand) != elementType) {
      return op->emitOpError(
          "requires the same element type for all operands and results");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MOSameOperandsAndResultElementShapeAttr
//===----------------------------------------------------------------------===//

/// Get the element type of a tensor value.
static TypedAttr getShapeAttr(Value v) {
  return v.getType().cast<MO::TensorType>().getShapeAttr();
}

LogicalResult impl::verifyMOSameOperandsAndResultShapeAttr(Operation *op) {
  if (failed(mlir::OpTrait::impl::verifyAtLeastNOperands(op, 1)) ||
      failed(mlir::OpTrait::impl::verifyAtLeastNResults(op, 1)))
    return failure();

  auto shapeAttr = getShapeAttr(op->getResult(0));

  // Verify result element type matches first result's element type.
  for (auto result : llvm::drop_begin(op->getResults(), 1)) {
    if (getShapeAttr(result) != shapeAttr) {
      return op->emitOpError(
          "requires the same shape for all operands and results");
    }
  }

  // Verify operand's element type matches first result's element type.
  for (auto operand : op->getOperands()) {
    if (getShapeAttr(operand) != shapeAttr) {
      return op->emitOpError(
          "requires the same shape for all operands and results");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MOSameOperandsElementType
//===----------------------------------------------------------------------===//

/// This follows the builtin verifySameOperandsAndResultElementType.
LogicalResult impl::verifyMOSameOperandsElementType(Operation *op) {
  if (failed(mlir::OpTrait::impl::verifyAtLeastNOperands(op, 1)) ||
      failed(mlir::OpTrait::impl::verifyAtLeastNResults(op, 1)))
    return failure();

  auto elementType = getElementType(op->getOperand(0));

  // Verify operand's element type matches first result's element type.
  for (auto operand : op->getOperands())
    if (getElementType(operand) != elementType)
      return op->emitOpError("requires the same element type for all operands");

  return success();
}

//===----------------------------------------------------------------------===//
// Deparameterization
//===----------------------------------------------------------------------===//

LogicalResult impl::defaultDeparameterize(ParamDeclarationInterface op,
                                          mlir::RewriterBase &rewriter) {
  ArrayRef<KGEN::ParamDeclAttr> decls = op.getOutputParamDecls();
  if (decls.empty())
    return failure();

  // The default deparameterization only supports a single result.
  if (op->getNumResults() != 1)
    return failure();
  Value res = op->getOpResult(0);

  auto resTy = dyn_cast<TensorType>(res.getType());
  if (!resTy)
    return failure();

  auto shapeTy = rewriter.getType<ShapeType>();
  TensorType deparameterizedResTy =
      resTy.hasRank()
          ? resTy.cloneWith(resTy.getIntDims(), resTy.getElementType())
          : resTy.cloneWith(KGEN::UnknownAttr::get(shapeTy));
  res.setType(deparameterizedResTy);

  rewriter.setInsertionPointAfter(op);
  auto rebindOp = rewriter.create<RebindOp>(op.getLoc(), resTy, res, decls);
  rewriter.replaceAllUsesExcept(res, rebindOp, rebindOp);
  op.setOutputParamDecls({});

  return success();
}

//===----------------------------------------------------------------------===//
// ShapeFuncMaterializationState
//===----------------------------------------------------------------------===//

ShapeFuncMaterializationState::ShapeFuncMaterializationState(MLIRContext *ctx) {
  auto moDialect = ctx->getLoadedDialect<MODialect>();
  shapeFuncModule = moDialect->getShapeFuncModule().clone();
}

void ShapeFuncMaterializationState::setRewriter(mlir::RewriterBase &rewriter) {
  this->rewriter = &rewriter;
}

ModuleOp ShapeFuncMaterializationState::getShapeFuncModule() {
  return shapeFuncModule.get();
}

mlir::RewriterBase &ShapeFuncMaterializationState::getRewriter() {
  return *rewriter;
}

void ShapeFuncMaterializationState::markShapeFuncAsUsed(
    KGEN::GeneratorOp func) {
  func.setExported(); // make sure it won't be removed during elaboration
  usedShapeFuncNames.insert(func.getNameAttr());
}

SmallVector<KGEN::FuncOp>
ShapeFuncMaterializationState::getUsedShapeFuncsAfterElaboration() {
  SmallVector<KGEN::FuncOp> funcs;
  for (StringAttr name : usedShapeFuncNames) {
    if (auto func = shapeFuncModule->lookupSymbol<KGEN::FuncOp>(name)) {
      func.setNotExported(); // this is no longer needed after elaboration
      funcs.push_back(func);
    }
  }
  return funcs;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/MODialect/MOInterfaces.cpp.inc"
