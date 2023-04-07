//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MO dialect interfaces.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "mlir/IR/BuiltinTypes.h"
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

/// Helper to print matrix shapes without element types.
static std::string asString(ArrayRef<TypedAttr> matShape) {
  std::string result;
  llvm::raw_string_ostream os(result);
  llvm::interleaveComma(
      matShape, os, [&](TypedAttr dim) { dim.print(os, /*elideType=*/true); });
  return result;
};

LogicalResult impl::verifyMatmulLike(MatmulLike op) {
  // We check that statically known ranks match each other (implicit
  // broadcasting is not supported).
  SmallVector<TensorValue> tensors{op.getTensorA(), op.getTensorB(),
                                   op.getTensorResult()};
  if (failed(verifyMatchingRanks(tensors))) {
    return op.emitError(
        "input and output ranks must be all dynamic, or all static and match");
  }

  TensorType aType = op.getTensorA().getType();
  TensorType bType = op.getTensorB().getType();

  // If either input is dynamically ranked, we don't try to verify.
  // When partially known ranks are supported we might refine this.
  if (!aType.hasRank() || !bType.hasRank())
    return success();

  if (aType.getRank() < 2) {
    return op.emitError("expected matmul-like input tensors to have at least "
                        "rank 2, got ")
           << aType.getRank();
  }

  // If either inputs are dynamically shaped, we don't try to verify.
  if (!aType.hasStaticShape() || !bType.hasStaticShape())
    return success();

  ArrayRef<TypedAttr> aMatShape = aType.getDims().take_back(2);
  ArrayRef<TypedAttr> bMatShape = bType.getDims().take_back(2);
  if (aMatShape[1] != bMatShape[0]) {
    return op.emitError(
               "expected matmul-like input matrix shapes (M, K) x (K, N), got ")
           << "(" << asString(aMatShape) << ") x (" << asString(bMatShape)
           << ")";
  }

  TensorType resType = op.getTensorResult().getType();

  // If the result is dynamically ranked/shaped, we defer to shape inference.
  if (!resType.hasStaticShape())
    return success();

  ArrayRef<TypedAttr> resMatShape = resType.getDims().take_back(2);
  if (resMatShape[0] != aMatShape[0] || resMatShape[1] != bMatShape[1]) {
    return op.emitError("expected matmul-like inputs (M, K) x (K, N) to "
                        "produce result matrix (M, N), got ")
           << "(" << asString(aMatShape) << ") x (" << asString(bMatShape)
           << ") -> (" << asString(resMatShape) << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Reduction
//===----------------------------------------------------------------------===//

LogicalResult impl::verifyReduction(Reduction op) {
  // TODO: test
  if (failed(verifyMatchingRanks({op.getInput(), op.getResult()})))
    return op.emitError(
        "input and ranks ranks must be both dynamic, or both static and match");

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
// ParameterizationInterface
//===----------------------------------------------------------------------===//

size_t ParameterizationInterface::numShapeParams = 0;
size_t ParameterizationInterface::numDimParam = 0;

KGEN::ParamDeclAttr ParameterizationInterface::getNewShapeParam() {
  auto shapeTy = MO::ShapeType::get(getContext());
  auto nameAttr = StringAttr::get(getContext(), "Sh" + Twine(numShapeParams++));
  return KGEN::ParamDeclAttr::get(nameAttr, shapeTy);
}

KGEN::ParamDeclAttr ParameterizationInterface::getNewDimParam() {
  auto nameAttr = StringAttr::get(getContext(), "D" + Twine(numDimParam++));
  return KGEN::ParamDeclAttr::get(nameAttr, IndexType::get(getContext()));
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
  op.setOutputParamDecls({});

  // The default deparameterization only supports a single result.
  if (op->getNumResults() != 1)
    return failure();

  auto resTy = dyn_cast<TensorType>(op->getResult(0).getType());
  if (!resTy)
    return failure();

  auto shapeTy = rewriter.getType<ShapeType>();
  TensorType unknownResTy;
  if (resTy.hasRank())
    unknownResTy = resTy.cloneWith(resTy.getIntDims(), resTy.getElementType());
  else
    unknownResTy = resTy.cloneWith(KGEN::UnknownAttr::get(shapeTy));

  Operation *newOp = rewriter.create(op.getLoc(), op->getName().getIdentifier(),
                                     op->getOperands(), {unknownResTy},
                                     op->getAttrDictionary().getValue());
  rewriter.replaceOpWithNewOp<RebindOp>(op, resTy, newOp->getResult(0), decls);

  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/MODialect/MOInterfaces.cpp.inc"
