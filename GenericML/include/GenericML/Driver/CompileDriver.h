//===----------------------------------------------------------------------===//
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
/// Given an MLIR module with a MEF-compatible encoding, emit it to MEF using
/// the provided output stream.
LogicalResult convertToMEF(mlir::ModuleOp module, raw_ostream &os,
                           bool enableMEFMetadata);
} // namespace M

#endif // GENERICML_DRIVER_COMPILEDRIVER_H
