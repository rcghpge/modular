//===- graphrt-translate.cpp - GraphRT Translation Driver -----------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

//#include "graphrt/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"

int main(int argc, char **argv) {
  //graphrt::registerAllTranslations();
  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "GraphRT Translation Testing Tool"));
}
