//===- GenericML/Driver/CompileDriver.h -----------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_COMPILEDRIVER_H
#define GENERICML_DRIVER_COMPILEDRIVER_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace llvm {
class MemoryBuffer;
}

namespace M {
/// Given an MLIR module with a BEF-compatible encoding, emit it to BEF and
/// return a MemoryBuffer containing the data.
std::unique_ptr<llvm::MemoryBuffer> convertToBEF(mlir::ModuleOp module,
                                                 bool enableBEFMetadata);
} // namespace M

#endif // GENERICML_DRIVER_COMPILEDRIVER_H
