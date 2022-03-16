//===- graphrt-opt.cpp - The graphrt-opt driver ---------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/InitAllDialects.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register MLIR stuff
  M::registerAllGraphRTDialects(registry);
  // M::registerAllPasses();

  // Register the standard passes we want.
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();

  return mlir::failed(mlir::MlirOptMain(argc, argv, "graphrt optimizer driver",
                                        registry,
                                        /*preloadDialectsInContext=*/false));
}
