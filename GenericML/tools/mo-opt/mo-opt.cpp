//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace M;

int main(int argc, char **argv) {
  DialectRegistry registry;
  MO::loadDialects(registry);
  MO::registerPasses();

  // We'll need a runtime during compilation, both to support multi-threaded
  // compilation and to convey global options.
  std::unique_ptr<LLCL::Runtime> runtime = LLCL::createUniqueRuntime(
      LLCL::RuntimeOptions().withLeakCheckedAllocator());

  return failed(
      mlir::MlirOptMain(argc, argv, "MO dialect test driver", registry));
}
