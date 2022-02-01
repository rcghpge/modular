//===- Translation.cpp - MLIR to BEF translation registration -------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This registers MLIRToBEF as an MLIR translation for use with
// graphrt-translate.
//
//===----------------------------------------------------------------------===//

#include "mlir/Translation.h"
#include "GraphRT/MLIRToBEF.h"
#include "mlir/IR/BuiltinOps.h"
using namespace M;

void M::registerMLIRToBEFTranslation() {
  static llvm::cl::opt<bool> disableOptionalSections(
      "disable-optional-bef-sections",
      llvm::cl::desc("Disable optional sections for register types, attribute "
                     "types and attribute names."),
      llvm::cl::init(false));

  static mlir::TranslateFromMLIRRegistration toBEF(
      "export-bef",
      [](ModuleOp module, llvm::raw_ostream &os) -> LogicalResult {
        auto data = translateMLIRToBEF(module, disableOptionalSections);
        if (data.empty())
          return failure();
        os.write(reinterpret_cast<const char *>(data.data()), data.size());
        return success();
      });
}
