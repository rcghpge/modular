//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// TODO: Move elsewhere, this logic is not driver-specific.

#ifndef MOF_DRIVER_DRIVERIMPL_H
#define MOF_DRIVER_DRIVERIMPL_H

#include "Support/Buffer.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/ML/DebugPrint.h"

namespace M::MOF {

/// Parses the buffer containing MLIR representation of a model in the MO graph
/// dialect.
OwningOpRef<ModuleOp> parseMODialect(M::BufferRef buffer, MLIRContext *ctx);

} // namespace M::MOF

#endif // MOF_DRIVER_DRIVERIMPL_H
