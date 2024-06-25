//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "GenericML/MGPDialect/MGPDialect.h"
#include "GraphRT/GRTDialect/GRTDialect.h"
#include "GraphRT/MEFDialect/MEFDialect.h"
#include "LLCL/CompilerSupport/Context.h"
#include "LLCL/Init/Init.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace M;

int main(int argc, char **argv) {
  DialectRegistry registry;
  MO::loadDialects(registry);
  registry.insert<mlir::func::FuncDialect, MEF::MEFDialect, GRT::GRTDialect,
                  MGP::MGPDialect>();

  MO::registerPasses();

  // Create our context.
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "mo-opt", Init::Options().withRuntimeOptions(
                    LLCL::RuntimeOptions().withLeakCheckedAllocator()));
  if (ctxOr.isError()) {
    llvm::errs() << "failed to create context: " << ctxOr.getError() << "\n";
    return 1;
  }
  registerContext(registry, *ctxOr);

  return failed(
      mlir::MlirOptMain(argc, argv, "MO dialect test driver", registry));
}
