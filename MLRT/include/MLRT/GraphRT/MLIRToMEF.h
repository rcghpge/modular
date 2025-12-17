//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the interface to the MLIRToMEF library.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_MLIR_TO_MEF_H
#define GRAPHRT_MLIR_TO_MEF_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace M {
class ConcatenationTree;

/// This function converts the specified MLIR module containing a executor
/// compatible program to the BinaryExecutableFormat (MEF) format, which is the
/// binary format that the executor takes.  When `enableMEFMetadata` is true
/// this adds metadata sections that allow MEF2MLIR to produce full fidelity
/// input.
///
/// On error, this emits the error message through the MLIR error handler, and
/// returns an empty MEFBuffer.
ConcatenationTree translateMLIRToMEF(ModuleOp module,
                                     bool enableMEFMetadata = false);

// Register this translation for GraphRTTranslate.
void registerMLIRToMEFTranslation(DialectRegistry &otherDialects);

} // namespace M

#endif // GRAPHRT_MLIR_TO_MEF_H
