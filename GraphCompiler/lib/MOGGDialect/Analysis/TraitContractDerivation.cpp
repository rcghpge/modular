//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/Analysis/TraitContractDerivation.h"

using namespace M;
using namespace M::MOGG;
using namespace mlir;

std::optional<TensorArgContract> M::MOGG::declaredExecuteOperandContract(
    const KernelDeclFunctionAdaptor &executeFn, unsigned operandIdx,
    bool isOutput) {
  llvm::ArrayRef<KernelDeclOperandAdaptor> args =
      isOutput ? executeFn.outputArguments : executeFn.inputArguments;
  if (operandIdx >= args.size())
    return std::nullopt;
  return args[operandIdx].getTraitContract();
}

std::optional<TensorArgContract>
M::MOGG::declaredShapeOperandContract(const KernelDeclFunctionAdaptor &shapeFn,
                                      unsigned operandIdx) {
  if (operandIdx >= shapeFn.inputArguments.size())
    return std::nullopt;
  return shapeFn.inputArguments[operandIdx].getTraitContract();
}
