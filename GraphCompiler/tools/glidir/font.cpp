//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "font.h"
#include "Support/Compiler/MLIRDenseAttr.h"

namespace M::GlidIR {

alignas(M::kAsmResourceBlobAlignment) constexpr char fontDataArray[] = {
#embed MLIR_VIZ_FONT
};

alignas(M::kAsmResourceBlobAlignment) constexpr char emojiFontDataArray[] = {
#embed MLIR_VIZ_EMOJI_FONT
};

llvm::StringRef getFontData() { return {fontDataArray, sizeof(fontDataArray)}; }
llvm::StringRef getEmojiFontData() {
  return {emojiFontDataArray, sizeof(emojiFontDataArray)};
}

} // namespace M::GlidIR
