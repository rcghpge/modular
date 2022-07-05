//===- modular-lsp-server.cpp ---------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Provides an LSP server for all the modular dialects.
//
//===----------------------------------------------------------------------===//

#include "Faux/Dialect/FauxDialect.h"
#include "GraphRT/BEFDialect/BEFDialect.h"
#include "GraphRT/GRTDialect/GRTDialect.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/MetaDialect/MetaDialect.h"
#include "TFLite/TFLDialect/TFLDialect.h"
#include "TFLite/TFLPDialect/TFLPDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;
using namespace M;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert< // Modular Dialects
      BEF::BEFDialect, Faux::FauxDialect, GRT::GRTDialect, KGEN::KGENDialect,
      KGEN::MetaDialect, TFL::TFLDialect, TFLP::TFLPDialect>();
  return failed(MlirLspServerMain(argc, argv, registry));
}
