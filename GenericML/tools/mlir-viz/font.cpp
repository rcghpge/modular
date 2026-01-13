//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "font.h"
#include "Support/Compiler/MLIRDenseAttr.h"

namespace M::MlirViz {

alignas(M::kAsmResourceBlobAlignment) constexpr char fontDataArray[] = {
#embed MLIR_VIZ_FONT
};

llvm::StringRef getFontData() { return {fontDataArray, sizeof(fontDataArray)}; }

} // namespace M::MlirViz
