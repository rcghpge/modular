//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "MLRT/GraphRT/GRTDialect/GRTDialect.h"
#include "MLRT/GraphRT/MEFToMLIR.h"
#include "MLRT/GraphRT/MLIRToMEF.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
using namespace M;

int main(int argc, char **argv) {
  DialectRegistry extraDialectsForMLIRToMEF;
  extraDialectsForMLIRToMEF.insert<GRT::GRTDialect>();
  extraDialectsForMLIRToMEF.insert<mlir::quant::QuantDialect>();
  registerMLIRToMEFTranslation(extraDialectsForMLIRToMEF);

  registerMEFToMLIRTranslation();

  return failed(
      mlir::mlirTranslateMain(argc, argv, "GraphRT Translation Testing Tool"));
}
