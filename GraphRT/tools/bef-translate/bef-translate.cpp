//===- bef-translate.cpp - BEF Translation Driver -------------------------===//
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
#include "GraphRT/MLIRToBEF.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
using namespace M;

int main(int argc, char **argv) {
  registerMLIRToBEFTranslation();
  registerBEFToMLIRTranslation();

  return failed(
      mlir::mlirTranslateMain(argc, argv, "GraphRT Translation Testing Tool"));
}
