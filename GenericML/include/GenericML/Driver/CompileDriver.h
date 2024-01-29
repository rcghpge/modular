//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_COMPILEDRIVER_H
#define GENERICML_DRIVER_COMPILEDRIVER_H

#include "GenericML/GraphCompiler/Passes.h"
#include "LLCL/ForwardDecls.h"
#include "Support/Buffer.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M {

/// Given an MLIR module with a MEF-compatible encoding,
/// returns BufferRef with encoded MEF.
ErrorOr<BufferRef> convertToMEF(ModuleOp module, bool enableMEFMetadata);

} // namespace M

#endif // GENERICML_DRIVER_COMPILEDRIVER_H
