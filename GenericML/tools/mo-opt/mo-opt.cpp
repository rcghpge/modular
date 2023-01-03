//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MODialect/MODialect.h"
#include "GenericML/MOGGDialect/MOGGDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace M;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<MO::MODialect, MOGG::MOGGDialect>();

  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerCanonicalizerPass();

  return failed(
      mlir::MlirOptMain(argc, argv, "MO dialect test driver", registry));
}
