//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GLIDIR_FONT_H
#define GLIDIR_FONT_H

#include "Support/SymbolExport.h"
#include "llvm/ADT/StringRef.h"

namespace M::GlidIR {

MODULAR_CXX_EXPORT llvm::StringRef getFontData();
MODULAR_CXX_EXPORT llvm::StringRef getEmojiFontData();

} // namespace M::GlidIR

#endif // GLIDIR_FONT_H
