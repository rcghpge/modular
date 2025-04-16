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

namespace M::MOGG {
/// Verify the call to a MO or custom kernel.
ErrorOrSuccess mapMOOpToKernelImplementation(
    Operation *op,
    const KGEN::MOGGPreElab::MojoKernelFunctionAdaptor<KGEN::GeneratorOp>
        &adaptor);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGSUPPORT_OPERANDMAPPER_H
