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

#include "GraphRT/GRTDialect/GRTDialect.h"
#include "GraphRT/MEFToMLIR.h"
#include "GraphRT/MIPDialect/MIPDialect.h"
#include "GraphRT/MLIRToMEF.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
using namespace M;

int main(int argc, char **argv) {
  DialectRegistry extraDialectsForMLIRToMEF;
  extraDialectsForMLIRToMEF.insert<GRT::GRTDialect, MIP::MIPDialect>();
  extraDialectsForMLIRToMEF.insert<mlir::quant::QuantizationDialect>();
  registerMLIRToMEFTranslation(extraDialectsForMLIRToMEF);

  registerMEFToMLIRTranslation();

  return failed(
      mlir::mlirTranslateMain(argc, argv, "GraphRT Translation Testing Tool"));
}
