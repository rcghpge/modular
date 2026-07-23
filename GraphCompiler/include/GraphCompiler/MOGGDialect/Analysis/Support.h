//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_SUPPORT_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_SUPPORT_H

#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "KGEN/KGENDialect/KGENOps.h"

namespace M::MOGG {

class MojoRegistry;

/// Holds the Mojo registration name and function label for an MLIR operation.
struct MojoOperationInfo {
  StringRef operationName, functionLabel;
};

/// Returns the Mojo registration name and function label for `op`.
MojoOperationInfo getMojoOperationInfo(Operation *op);

/// Get the number of DPS outputs attached for the given Mojo code.
int64_t getNumDPSStyleOutputs(KGEN::GeneratorOp mojoCode);

/// Construct the kernel adaptor for the given operation.
ErrorOr<KernelDeclAdaptor> getMojoKernelAdaptor(const MojoRegistry &registry,
                                                Operation *op);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_SUPPORT_H
