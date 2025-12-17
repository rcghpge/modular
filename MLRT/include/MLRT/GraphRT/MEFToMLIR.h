//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the interface to the MEFToMLIR library.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_MEFTOMLIR_H
#define MLRT_GRAPHRT_MEFTOMLIR_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M {

/// Read the specified a MEF file from the specified binary buffer, and
/// returning the created module on success.  On failure, report an error at the
/// specified location and return null.
ModuleOp translateMEFToMLIR(ArrayRef<uint8_t> buffer, Location loc);

// Register this translation for GraphRTTranslate.
void registerMEFToMLIRTranslation();

} // namespace M

#endif // MLRT_GRAPHRT_MEFTOMLIR_H
