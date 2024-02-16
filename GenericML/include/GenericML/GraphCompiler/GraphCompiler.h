//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MO_GRAPHCOMPILER_H
#define GENERICML_MO_GRAPHCOMPILER_H

#include "LLCL/ForwardDecls.h"
#include "Support/Buffer.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/DialectRegistry.h"

namespace M::MO {

/// Load dialects in Graph Compiler to the given registry.
void loadDialects(mlir::DialectRegistry &registry);

/// Register Graph Compiler passes. This function is not thread-safe.
void registerPasses();

struct MoToMgpOptions;

/// Compile a module at the MO dialect level down to the MGP level.
LogicalResult lowerMOToMGP(ModuleOp module,
                           const MoToMgpOptions &moToMgpOptions,
                           Runtime &runtime);

/// Convert a module at the MGP level to a buffer in MEF format.
/// The module is left unchanged (even though we may need to apply some
/// rewrite to prepare for MEF.)
ErrorOr<BufferRef> convertMGPToMEF(ModuleOp module, bool enableMEFMetadata);

} // namespace M::MO

#endif // GENERICML_MO_GRAPHCOMPILER_H
