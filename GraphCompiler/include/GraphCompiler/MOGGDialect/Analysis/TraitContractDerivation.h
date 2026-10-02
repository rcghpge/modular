//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Derives tensor-arg trait contracts on demand from the kernel declaration
// (`mogg.kernel.decl` / `mogg.shape_func.decl`), the source of truth for each
// operand position's contract (GEX-4071).
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_TRAITCONTRACTDERIVATION_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_TRAITCONTRACTDERIVATION_H

#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"

#include <optional>

namespace M::MOGG {

/// The trait contract declared for the input or DPS-output `execute` operand at
/// `operandIdx`. Returns `std::nullopt` for a legacy or non-tensor argument,
/// and for an out-of-range index (legacy variadic packing can shorten the
/// operand list; a trait-bound declaration never hits this).
std::optional<TensorArgContract>
declaredExecuteOperandContract(const KernelDeclFunctionAdaptor &executeFn,
                               unsigned operandIdx, bool isOutput);

/// The trait contract declared for one `mogg.call.shape` operand. Trait-bound
/// tensor arguments get role-less (`TensorArgRole::none`) contracts; legacy
/// arguments yield `std::nullopt`.
std::optional<TensorArgContract>
declaredShapeOperandContract(const KernelDeclFunctionAdaptor &shapeFn,
                             unsigned operandIdx);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_TRAITCONTRACTDERIVATION_H
