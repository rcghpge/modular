//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This registers BEFToMLIR as an MLIR translation for use with mlir-translate
// based tools.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-translate/Translation.h"
#include "GraphRT/BEFDialect/BEFDialect.h"
#include "GraphRT/BEFToMLIR.h"
#include "Support/MDialect/MDialect.h"
#include "Support/ML/SizeUtils.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/SourceMgr.h"
using namespace M;

void M::registerBEFToMLIRTranslation() {
  static mlir::TranslateToMLIRRegistration toMLIR(
      "import-bef", "Import a BEF file and translate to MLIR",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *ctx) {
        ctx->loadDialect<mlir::quant::QuantizationDialect>();
        ctx->loadDialect<MDialect>();
        const llvm::MemoryBuffer *input =
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
        Location loc =
            FileLineColLoc::get(ctx, input->getBufferIdentifier(), 0, 0);
        auto contents = llvm::ArrayRef<uint8_t>(
            reinterpret_cast<const uint8_t *>(input->getBufferStart()),
            input->getBufferSize());
        return translateBEFToMLIR(contents, loc).getOperation();
      },
      [](DialectRegistry &) {}, maxAttributeAlignment());
}
