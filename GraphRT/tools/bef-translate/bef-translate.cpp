//===- bef-translate.cpp --------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFToMLIR.h"
#include "GraphRT/GRTDialect/GRTDialect.h"
#include "GraphRT/MLIRToBEF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
using namespace M;

int main(int argc, char **argv) {
  DialectRegistry extraDialectsForMLIRToBEF;
  extraDialectsForMLIRToBEF.insert<GRT::GRTDialect>();
  registerMLIRToBEFTranslation(extraDialectsForMLIRToBEF);

  registerBEFToMLIRTranslation();

  return failed(
      mlir::mlirTranslateMain(argc, argv, "GraphRT Translation Testing Tool"));
}
