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

LogicalResult impl::verifyMatmulLike(MatmulLike op) {
  TensorType aType = op.getTensorA().getType();
  TensorType bType = op.getTensorB().getType();

  // If either input is dynamically ranked, we don't try to verify.
  // When partially known ranks are supported we might refine this.
  if (!aType.hasRank() || !bType.hasRank())
    return success();

  // Implicit broadcasting is not supported, so static ranks must match.
  if (aType.getRank() != bType.getRank()) {
    return op.emitError("matmul-like input tensors are not the same rank, got ")
           << aType.getRank() << " and " << bType.getRank();
  }
  if (aType.getRank() < 2) {
    return op.emitError("expected matmul-like input tensors to have at least "
                        "rank 2, got ")
           << aType.getRank();
  }

  // If either inputs are dynamically shaped, we don't try to verify.
  if (!aType.hasStaticShape() || !bType.hasStaticShape())
    return success();

  llvm::ArrayRef<int64_t> aMatShape = op.getShapeA().take_back(2);
  llvm::ArrayRef<int64_t> bMatShape = op.getShapeB().take_back(2);
  if (aMatShape[1] != bMatShape[0]) {
    return op.emitError(
               "expected matmul-like input matrix shapes (M, K) x (K, N), got ")
           << "(" << aMatShape << ") x (" << bMatShape << ")";
  }

  TensorType resType = op.getTensorResult().getType();

  // If the result is dynamically ranked/shaped, we defer to shape inference.
  if (!resType.hasRank())
    return success();
  if (resType.getRank() != aType.getRank()) {
    return op.emitError("expected matmul-like result tensor to have the same "
                        "rank as the input (")
           << aType.getRank() << "), got " << resType.getRank();
  }

  if (!resType.hasStaticShape())
    return success();

  llvm::ArrayRef<int64_t> resMatShape = op.getShapeResult().take_back(2);
  if (resMatShape[0] != aMatShape[0] || resMatShape[1] != bMatShape[1]) {
    return op.emitError("expected matmul-like inputs (M, K) x (K, N) to "
                        "produce result matrix (M, N), got ")
           << "(" << aMatShape << ") x (" << bMatShape << ") -> ("
           << resMatShape << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/MODialect/MOInterfaces.cpp.inc"
