//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Operand inference: maps an MO op's operands and results to a kernel's
// arguments and results. Shared by the MO->MOGG and MO->MAP lowering paths.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_OPERANDINFERENCE_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_OPERANDINFERENCE_H

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

/// Maps a kernel call's `operands` and `results` (given directly as value
/// ranges) to the kernel's arguments. This is the general entry point: it does
/// not require an `Operation` and so also serves destination-passing-style call
/// ops whose outputs are operands rather than results (e.g.
/// `mogg._call.opaque`). `opName` names the caller in diagnostics; the first
/// two flags mirror the `SameVariadicOperandSize` / extra-output-shape-argument
/// op properties. Set `resultLeavesByValue` when the kernel's function result
/// is returned rather than written through a destination, so `results` carries
/// no entry standing in for it.
/// `allowMutToImmut` lets a mutable operand satisfy an immutable kernel operand
/// (a sound read-only widening; the converse is never allowed). Only
/// ConformCallOperands opts in, so it can accept a `mogg._tensor.create` that
/// MAPToMOGG built as a mutable buffer for a fused mutable-load and then
/// recreate it immutable to match the callee.
ErrorOr<KernelDeclOperandMapping> mapToKernelImplementation(
    ValueRange operands, ValueRange results,
    const KernelDeclFunctionAdaptor &adaptor, StringRef opName,
    bool hasSameVariadicOperandSize, bool hasExtraOutputShapeArgument,
    bool resultLeavesByValue = false, bool allowMutToImmut = false);

/// Convenience overload that reads the operands/results and op properties off
/// `op`.
ErrorOr<KernelDeclOperandMapping>
mapMOOpToKernelImplementation(Operation *op,
                              const KernelDeclFunctionAdaptor &adaptor);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_OPERANDINFERENCE_H
