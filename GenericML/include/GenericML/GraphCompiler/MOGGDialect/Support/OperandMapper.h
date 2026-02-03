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

#include "KGEN/MOGGPreElab/MojoKernelAdaptor.h"

#include "llvm/ADT/DenseSet.h"

namespace M::MOGG {

/// Maps an index (operand or result) to a mojo kernel operand.
using IndexToOperandAdaptorMapping =
    llvm::MapVector<uint64_t, KGEN::MOGGPreElab::MojoKernelOperandAdaptor>;

struct OperandMapping {
  IndexToOperandAdaptorMapping operandMapping;
  IndexToOperandAdaptorMapping resultMapping;
};

/// Operand mapping will map an Operation*'s operands and results to the
/// underlying kernel's operands. This is the mirror of parameter mapping which
/// is for attributes and kernel parameters. The act of validating an op is
/// identical to the act of mapping. Therefore, the two are merged. To validate
/// an op with its kernel, call this function and check that it has not failed.
ErrorOr<OperandMapping> mapMOOpToKernelImplementation(
    Operation *op, const KGEN::MOGGPreElab::MojoKernelFunctionAdaptor &adaptor);

// The value is a vector because of variadics capturing multiple MO values.
using KernelIndexToOpIndex = llvm::MapVector<uint64_t, SmallVector<uint64_t>>;

struct InverseOperandMapping {
  KernelIndexToOpIndex inputArgumentsMapping;
  KernelIndexToOpIndex outputArgumentsMapping;
  std::optional<uint64_t> resultMapping;
};

InverseOperandMapping invertMapping(const OperandMapping &mapping);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGSUPPORT_OPERANDMAPPER_H
