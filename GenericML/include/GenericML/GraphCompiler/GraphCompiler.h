//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MO_GRAPHCOMPILER_H
#define GENERICML_MO_GRAPHCOMPILER_H

#include "mlir/IR/DialectRegistry.h"

namespace M::MO {

/// Load dialects in Graph Compiler to the given registry.
void loadDialects(mlir::DialectRegistry &registry);

/// Register Graph Compiler passes. This function is not thread-safe.
void registerPasses();

} // namespace M::MO

#endif // GENERICML_MO_GRAPHCOMPILER_H
