//===- GraphRT/MLIRToBEF.h --------------------------------------*- C++ -*-===//
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

#include "Support/LLVM.h"

namespace M {

// This function converts the specified MLIR module containing a executor
// compatible program to the BinaryExecutableFormat (BEF) format, which is the
// binary format that the executor takes.
//
// On error, this emits the error message through the MLIR error handler, and
// returns an empty AlignedBuffer.
LogicalResult translateMLIRToBEF(ModuleOp module, raw_ostream &output);

// Register this translation for GraphRTTranslate.
void registerMLIRToBEFTranslation();

} // namespace M

#endif // GRAPHRT_MLIR_TO_BEF_H