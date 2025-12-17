//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This registers MEFToMLIR as an MLIR translation for use with mlir-translate
// based tools.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-translate/Translation.h"
#include "MLRT/GraphRT/MEFDialect/MEFDialect.h"
#include "MLRT/GraphRT/MEFToMLIR.h"
#include "Support/MDialect/MDialect.h"
#include "Support/ML/SizeUtils.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/SourceMgr.h"
using namespace M;

void M::registerMEFToMLIRTranslation() {
  static mlir::TranslateToMLIRRegistration toMLIR(
      "import-mef", "Import a MEF file and translate to MLIR",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *ctx) {
        ctx->loadDialect<mlir::quant::QuantDialect>();
        ctx->loadDialect<MDialect>();
        const llvm::MemoryBuffer *input =
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
        Location loc =
            FileLineColLoc::get(ctx, input->getBufferIdentifier(), 0, 0);
        auto contents = ArrayRef<uint8_t>(
            reinterpret_cast<const uint8_t *>(input->getBufferStart()),
            input->getBufferSize());
        return translateMEFToMLIR(contents, loc).getOperation();
      },
      [](DialectRegistry &) {}, maxAttributeAlignment());
}
