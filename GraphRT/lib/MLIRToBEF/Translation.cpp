//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This registers MLIRToBEF as an MLIR translation for use with
// bef-translate.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-translate/Translation.h"
#include "GraphRT/BEFDialect/BEFDialect.h"
#include "GraphRT/MLIRToBEF.h"
#include "Support/ConcatenationTree.h"
#include "mlir/IR/BuiltinOps.h"
using namespace M;

void M::registerMLIRToBEFTranslation(DialectRegistry &otherDialects) {
  static llvm::cl::opt<bool> enableBEFMetadata(
      "enable-bef-metadata",
      llvm::cl::desc("Enable metadata sections for better BEF2MLIR"),
      llvm::cl::init(false));

  static mlir::TranslateFromMLIRRegistration toBEF(
      "export-bef",
      [](ModuleOp module, llvm::raw_ostream &os) -> LogicalResult {
        auto data = translateMLIRToBEF(module, enableBEFMetadata);
        if (data.getSize() == 0)
          return failure();
        // write all the returned data out to a stream.
        data.traverse([&](ArrayRef<uint8_t> data) {
          os.write(reinterpret_cast<const char *>(data.data()), data.size());
        });
        return success();
      },
      [&](DialectRegistry &registry) {
        registry.insert<BEF::BEFDialect>();
        // If the caller provided any other dialects, add them.
        otherDialects.appendTo(registry);
      });
}
