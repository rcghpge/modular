//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MO dialect interfaces.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MODialect/MOInterfaces.h"

using namespace M;
using namespace MO;

LogicalResult
impl::verifyMatchingRanks(ArrayRef<TypedValue<M::MO::TensorType>> values) {
  SmallVector<size_t> ranks;
  for (auto v : values)
    if (TensorType t = v.getType(); t.hasRank())
      ranks.emplace_back(t.getRank());

  if (!ranks.empty())
    if (!llvm::all_equal(ranks))
      return failure();

  return success();
}

LogicalResult impl::verifyMatmulLike(MatmulLike op) {
  // We check that statically known ranks match each other (implicit
  // broadcasting is not supported).
  SmallVector<TypedValue<TensorType>> tensors{op.getTensorA(), op.getTensorB(),
                                              op.getTensorResult()};
  if (failed(verifyMatchingRanks(tensors)))
    return op.emitError("input and output ranks must match if static");

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

  ArrayRef<int64_t> aMatShape = op.getTensorAShape().take_back(2);
  ArrayRef<int64_t> bMatShape = op.getTensorBShape().take_back(2);
  if (aMatShape[1] != bMatShape[0]) {
    return op.emitError(
               "expected matmul-like input matrix shapes (M, K) x (K, N), got ")
           << "(" << aMatShape << ") x (" << bMatShape << ")";
  }

  TensorType resType = op.getTensorResult().getType();

  // If the result is dynamically ranked/shaped, we defer to shape inference.
  if (!resType.hasStaticShape())
    return success();

  ArrayRef<int64_t> resMatShape = op.getTensorResultShape().take_back(2);
  if (resMatShape[0] != aMatShape[0] || resMatShape[1] != bMatShape[1]) {
    return op.emitError("expected matmul-like inputs (M, K) x (K, N) to "
                        "produce result matrix (M, N), got ")
           << "(" << aMatShape << ") x (" << bMatShape << ") -> ("
           << resMatShape << ")";
  }

  return success();
}

LogicalResult SlidingWindow::verifyWindowParamShape(
    TypedValue<::M::MO::TensorType> windowParam, StringRef name) {
  auto input = cast<MOOpInterface>(this->getOperation()).getInputTensor(0);
  if (TensorType inputTy = input.getType(); inputTy.hasRank()) {
    if (TensorType windowParamTy = windowParam.getType();
        windowParamTy.hasStaticShape())
      if (static_cast<size_t>(windowParamTy.getShape()[0]) != inputTy.getRank())
        return emitError(name) << " with static size must match rank of input";
  }
  return success();
}

LogicalResult SlidingWindow::verifyWindowParamSizes(
    ArrayRef<TypedValue<M::MO::TensorType>> values) {
  SmallVector<size_t> sizes;
  for (auto v : values)
    if (TensorType t = v.getType(); t.hasStaticShape())
      sizes.emplace_back(t.getDimSize(0));

  if (!sizes.empty())
    if (!llvm::all_equal(sizes))
      return emitError("window parameter sizes must match if static");

  return success();
}

LogicalResult impl::verifySlidingWindow(SlidingWindow op) {
  if (failed(op.verifyWindowParamShape(op.getStrides(), "strides")))
    return failure();
  if (failed(op.verifyWindowParamShape(op.getDilations(), "dilations")))
    return failure();

  auto moOp = cast<MOOpInterface>(op.getOperation());
  if (failed(verifyMatchingRanks(
          {moOp.getInputTensor(0), moOp.getOutputTensor(0)})))
    return op.emitError("input and output ranks must match if static");

  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/MODialect/MOInterfaces.cpp.inc"
