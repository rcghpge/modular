//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DATALOW_UTILS
#define GENERICML_DATALOW_UTILS

#include "mlir/IR/Operation.h"
#include "llvm/Support/Casting.h"

namespace M {

// Checks if an operation depends on constants only.
template <typename... ConstantTypes>
bool dependsOnConstantsOnly(mlir::Operation *operation) {
  llvm::SmallVector<mlir::Operation *> next = {operation};
  mlir::DenseSet<mlir::Operation *> visited;
  while (!next.empty()) {
    mlir::Operation *op = next.back();
    next.pop_back();

    // Is a mlir::BlockArgument
    if (!op)
      return false;

    if (!visited.insert(op).second)
      continue;

    auto operands = op->getOperands();
    if (!llvm::range_size(operands)) {
      if (!llvm::isa<ConstantTypes...>(op))
        return false;

      continue;
    }

    for (auto operand : operands) {
      next.push_back(operand.getDefiningOp());
    }
  }

  return true;
}

} // namespace M

#endif
