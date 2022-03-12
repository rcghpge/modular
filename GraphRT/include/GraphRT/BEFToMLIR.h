//===- GraphRT/BEFToMLIR.h ------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the interface to the BEFToMLIR library.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_BEF_TO_MLIR_H
#define GRAPHRT_BEF_TO_MLIR_H

#include "Support/LLVM.h"

namespace M {

/// Read the specified a BEF file from the specified binary buffer, and
/// returning the created module on success.  On failure, report an error at the
/// specified location and return null.
ModuleOp translateBEFToMLIR(ArrayRef<uint8_t> buffer, Location loc);

// Register this translation for GraphRTTranslate.
void registerBEFToMLIRTranslation();

} // namespace M

#endif // GRAPHRT_BEF_TO_MLIR_H