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
#include <mlir/IR/Builders.h>

namespace M::MO {

/// Builds the operations needed to perform the PRelu operation in MO.
/// PRelu(x) = max(x, broadcast(0)) + alpha * min(x, broadcast(0))
TensorValue convertPRelu(OpBuilder &builder, Location loc, TensorValue x,
                         TensorValue alpha);

/// Builds the operations needed to perform the relu6 operation in MO.
/// ReLU6(x) = max(broadcast(0), min(x, broadcast(6)))
TensorValue convertRelu6(OpBuilder &builder, Location loc, TensorValue x);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOSUPPORT_ACTIVATIONS_H
