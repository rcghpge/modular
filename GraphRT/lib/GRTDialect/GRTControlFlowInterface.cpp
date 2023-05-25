//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/GRTDialect/GRTContolFlowInterface.h"
#include "GraphRT/MEFDialect/MEFOps.h"
#include "Support/Compiler/VerifyUtils.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"

mlir::LogicalResult M::GRT::verifyControlFlowInterfaceOp(Operation *op) {
  auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op);

  if (cfOp.getFlavor() == kLoopLike) {
    // The op's results must match the op's inputs.
    if (failed(checkMatchingTypes(op, "result", op->getResultTypes(),
                                  cfOp.getInputs().getTypes())))
      return failure();
  }

  // All the nested blocks must accept arguments matching the op's inputs.
  SmallVector<StringRef> regionNames = cfOp.getRegionNames();
  assert(regionNames.size() == cfOp->getNumRegions());
  for (size_t i = 0, n = cfOp->getNumRegions(); i < n; ++i) {
    Block *block = &cfOp->getRegion(i).front();
    if (failed(checkArgumentTypes(op, regionNames[i], block,
                                  cfOp.getInputs().getTypes())))
      return failure();
  }

  // All the nested blocks must return results matching the op's results.
  for (size_t i = 0, n = cfOp->getNumRegions(); i < n; ++i) {
    Block *block = &cfOp->getRegion(i).front();
    auto outputOp = cast<M::MEF::OutputOp>(block->getTerminator());
    SmallVector<Type> outputTypes =
        llvm::to_vector(outputOp.getOperands().getTypes());
    if (cfOp.getFlavor() == kLoopLike) {
      // Ok for nested blocks to return additional results, eg for
      // conditions.
      if (outputTypes.size() > op->getNumResults())
        outputTypes.erase(outputTypes.begin() + op->getNumResults(),
                          outputTypes.end());
    }
    if (failed(checkMatchingTypes(
            op,
            (Twine("mef.output operands in '") + regionNames[i] + "' block")
                .str(),
            outputTypes, op->getResultTypes())))
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GRTControlFlowInterface
//===----------------------------------------------------------------------===//

#include "GraphRT/GRTDialect/GRTControlFlowInterface.cpp.inc"
