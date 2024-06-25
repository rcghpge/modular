//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The graphrt-opt driver implementation.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/InitAllDialects.h"
#include "LLCL/CompilerSupport/Context.h"
#include "LLCL/Init/Init.h"
#include "Support/Context.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace M;

int main(int argc, char **argv) {
  DialectRegistry registry;

  // Register MLIR stuff
  registerAllGraphRTDialects(registry);
  // M::registerAllPasses();

  // Register the standard passes we want.
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();

  // Create our context.
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "graphrt-opt", Init::Options().withRuntimeOptions(
                         LLCL::RuntimeOptions().withLeakCheckedAllocator()));
  if (ctxOr.isError()) {
    llvm::errs() << "failed to create context: " << ctxOr.getError() << "\n";
    return 1;
  }
  registerContext(registry, *ctxOr);

  return failed(
      mlir::MlirOptMain(argc, argv, "graphrt optimizer driver", registry));
}
