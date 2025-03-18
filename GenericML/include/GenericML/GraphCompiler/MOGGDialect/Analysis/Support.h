//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_SUPPORT_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_SUPPORT_H

#include "KGEN/KGENDialect/KGENOps.h"

namespace M::MOGG {

class MojoLibraryAnalysis;

/// Get the number of DPS outputs attached for the given Mojo code.
int64_t getNumDPSStyleOutputs(KGEN::GeneratorOp mojoCode);

/// Returns the KGEN Generator associated with the given operation.
ErrorOr<KGEN::GeneratorOp>
getMojoKernelTarget(MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                    Operation *op);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_SUPPORT_H
