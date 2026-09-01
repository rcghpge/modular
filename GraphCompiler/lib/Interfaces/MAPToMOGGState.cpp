//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Interfaces/MAPToMOGGState.h"
#include "GraphCompiler/Interfaces/LowerableToMOGGInterface.h"

using namespace mlir;

namespace M::MAP {

LogicalResult MAPToMOGGState::lowerOp(Operation *op) {
  auto lowerable = dyn_cast<LowerableToMOGGInterface>(op);
  if (!lowerable)
    return op->emitOpError("MAPToMOGG: unsupported op");
  return lowerable.lowerToMOGG(*this);
}

LogicalResult MAPToMOGGState::lowerRegionBody(
    Block &block, llvm::function_ref<bool(Operation *)> shouldLower) {
  for (Operation &op : block) {
    if (!shouldLower(&op))
      continue;
    if (failed(lowerOp(&op)))
      return failure();
  }
  return success();
}

} // namespace M::MAP
