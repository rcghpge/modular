//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MO_GRAPHCOMPILER_H
#define GENERICML_MO_GRAPHCOMPILER_H

#include "AsyncRT/ForwardDecls.h"
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
                           const MoToMgpOptions &moToMgpOptions);

/// Convert a module at the MGP level to a buffer in MEF format.
/// Note, that this will destroy the dense resources in the MGP constants
/// in the module op, so it is a destructive conversion.
ErrorOr<BufferRef> convertMGPToMEF(ModuleOp module, bool enableMEFMetadata);

constexpr StringLiteral BugReportMessage =
    "=================================================================\n"
    "    !!! MAX AI Engine has encountered an internal error !!!\n"
    "=================================================================\n"
    "Please submit a bug report to https://github.com/modularml/max/issues "
    "and include the crash backtrace along with your command line "
    "invocation.\n";

} // namespace M::MO

#endif // GENERICML_MO_GRAPHCOMPILER_H
