//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Helper for mapping MO's operands and results to a kernel's arguments and
// results.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGSUPPORT_OPERANDMAPPER_H
#define GENERICML_GRAPHCOMPILER_MOGGSUPPORT_OPERANDMAPPER_H

#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"

#include "llvm/ADT/DenseSet.h"

namespace M::MOGG {

/// Operand mapping built from a `KernelDeclFunctionAdaptor`. Maps an
/// `Operation`'s operands and results to the underlying kernel's operands.
/// This is the mirror of parameter mapping which is for attributes and
/// kernel parameters. The act of validating an op is identical to the act
/// of mapping; to validate an op with its kernel, call
/// `mapMOOpToKernelImplementation` and check that it has not failed.
using KernelDeclIndexToOperandAdaptorMapping =
    llvm::MapVector<uint64_t, KernelDeclOperandAdaptor>;

struct KernelDeclOperandMapping {
  KernelDeclIndexToOperandAdaptorMapping operandMapping;
  KernelDeclIndexToOperandAdaptorMapping resultMapping;
};

ErrorOr<KernelDeclOperandMapping>
mapMOOpToKernelImplementation(Operation *op,
                              const KernelDeclFunctionAdaptor &adaptor);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGSUPPORT_OPERANDMAPPER_H
