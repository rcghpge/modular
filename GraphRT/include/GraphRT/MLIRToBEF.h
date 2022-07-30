//===- GraphRT/MLIRToBEF.h ------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the interface to the MLIRToBEF library.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_MLIR_TO_BEF_H
#define GRAPHRT_MLIR_TO_BEF_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace mlir {
class DialectRegistry;
} // end namespace mlir

namespace M {
class ConcatenationTree;

/// This function converts the specified MLIR module containing a executor
/// compatible program to the BinaryExecutableFormat (BEF) format, which is the
/// binary format that the executor takes.  When `enableBEFMetadata` is true
/// this adds metadata sections that allow BEF2MLIR to produce full fidelity
/// input.
///
/// On error, this emits the error message through the MLIR error handler, and
/// returns an empty BEFBuffer.
ConcatenationTree translateMLIRToBEF(ModuleOp module,
                                     bool enableBEFMetadata = false);

// Register this translation for GraphRTTranslate.
void registerMLIRToBEFTranslation(DialectRegistry &otherDialects);

} // namespace M

#endif // GRAPHRT_MLIR_TO_BEF_H
