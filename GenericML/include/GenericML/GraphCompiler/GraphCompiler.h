//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MO_GRAPHCOMPILER_H
#define GENERICML_MO_GRAPHCOMPILER_H

#include "LLCL/ForwardDecls.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/DialectRegistry.h"

namespace M::MO {

/// Load dialects in Graph Compiler to the given registry.
void loadDialects(mlir::DialectRegistry &registry);

/// Register Graph Compiler passes. This function is not thread-safe.
void registerPasses();

struct MoToMgpOptions;

/// Given an MLIR module in MO dialect form, compile down to MGP dialect for MEF
/// emission.
LogicalResult lowerMOToMGP(ModuleOp module,
                           const MoToMgpOptions &moToMgpOptions,
                           Runtime &runtime);

} // namespace M::MO

#endif // GENERICML_MO_GRAPHCOMPILER_H
