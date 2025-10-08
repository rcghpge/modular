//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_SUPPORT_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_SUPPORT_H

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MOGGPreElab/MojoKernelAdaptor.h"

namespace M::MOGG {

class MojoLibraryAnalysis;

/// Get the number of DPS outputs attached for the given Mojo code.
int64_t getNumDPSStyleOutputs(KGEN::GeneratorOp mojoCode);

/// Returns the KGEN Generator associated with the given operation.
template <class FuncT>
ErrorOr<FuncT>
getMojoKernelTarget(const MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                    Operation *op);

/// Returns the KGEN Generator associated with the given name and label.
template <class FuncT>
ErrorOr<FuncT>
getMojoKernelTarget(const MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                    StringRef mojoOperationName, StringRef functionLabel);

using KernelAdaptor = KGEN::MOGGPreElab::MojoKernelAdaptor<KGEN::GeneratorOp>;

/// Construct the kernel adaptor for the given operation.
template <class FuncT>
ErrorOr<KGEN::MOGGPreElab::MojoKernelAdaptor<FuncT>>
getMojoKernelAdaptor(const MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                     Operation *op);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_SUPPORT_H
