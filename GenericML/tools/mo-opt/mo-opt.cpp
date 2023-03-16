//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace M;

int main(int argc, char **argv) {
  DialectRegistry registry;
  MO::loadDialects(registry);
  MO::registerPasses();

  return failed(
      mlir::MlirOptMain(argc, argv, "MO dialect test driver", registry));
}
