//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Activation helpers for frontends converting to MO.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOSUPPORT_ACTIVATIONS_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOSUPPORT_ACTIVATIONS_H

#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M::MO {

/// Builds the operations needed to perform the PRelu operation in MO.
/// PRelu(x) = max(x, broadcast(0)) + alpha * min(x, broadcast(0))
TensorValue convertPRelu(OpBuilder &builder, Location loc, TensorValue x,
                         TensorValue alpha);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOSUPPORT_ACTIVATIONS_H
