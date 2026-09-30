//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// On-demand derivation of tensor-arg trait contracts from kernel declarations
// (GEX-4071).
//
// The trait contract of a kernel I/O tensor is a function of the declaration
// (`mogg.kernel.decl` / `mogg.shape_func.decl`), which is the source of truth
// for what a given operand position's contract is.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_TRAITCONTRACTDERIVATION_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_TRAITCONTRACTDERIVATION_H

#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"

#include <optional>

namespace M::MOGG {

/// The trait contract declared for one `execute` call operand: the input
/// (`isOutput == false`) or DPS-output operand at `operandIdx`. Returns
/// `std::nullopt` for a legacy or non-tensor argument, and for an out-of-range
/// index (exotic legacy variadic packing can legitimately shorten the operand
/// list; a trait-bound declaration never hits this).
std::optional<TensorArgContract>
declaredExecuteOperandContract(const KernelDeclFunctionAdaptor &executeFn,
                               unsigned operandIdx, bool isOutput);

/// The trait contract declared for one `mogg.call.shape` operand. Shape
/// functions carry role-less (`TensorArgRole::none`) contracts for their
/// trait-bound tensor arguments; legacy arguments yield `std::nullopt`.
std::optional<TensorArgContract>
declaredShapeOperandContract(const KernelDeclFunctionAdaptor &shapeFn,
                             unsigned operandIdx);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_TRAITCONTRACTDERIVATION_H
