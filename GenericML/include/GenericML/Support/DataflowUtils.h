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
inline bool dependsOnlyOnProperty(
    mlir::Operation *operation,
    llvm::function_ref<bool(mlir::Operation *)> propertyCheck =
        [](mlir::Operation *) { return true; }) {
  llvm::SmallVector<mlir::Operation *> next = {operation};
  mlir::DenseSet<mlir::Operation *> visited;
  while (!next.empty()) {
    mlir::Operation *op = next.back();
    next.pop_back();

    // Is a mlir::BlockArgument
    if (!op)
      return false;

    // Skip visited ops.
    if (!visited.insert(op).second)
      continue;

    // Check the property if `op` has no operands.
    auto operands = op->getOperands();
    if (!llvm::range_size(operands)) {
      if (!propertyCheck(op))
        return false;

      continue;
    }

    for (auto operand : operands)
      next.push_back(operand.getDefiningOp());
  }

  return true;
}

template <typename... ConstantTypes>
bool dependsOnConstantsOnly(mlir::Operation *operation) {
  auto propertyCheck = [](mlir::Operation *op) {
    return llvm::isa<ConstantTypes...>(op);
  };

  return dependsOnlyOnProperty(operation, propertyCheck);
}

} // namespace M

#endif
