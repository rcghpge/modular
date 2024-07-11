//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Init/Init.h"
#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "GenericML/MGPDialect/MGPDialect.h"
#include "GraphRT/GRTDialect/GRTDialect.h"
#include "GraphRT/MEFDialect/MEFDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/PrettyStackTrace.h"

using namespace M;

int main(int argc, char **argv) {
  // Don't create an object of type llvm::InitLLVM here, as it will be created
  // in the call to `mlir::MlirOptMain` later (and we can only create one
  // instance of it). We do, however, want to register a message specific to the
  // graph compiler to show the user if we crash.
  llvm::setBugReportMsg(M::MO::BugReportMessage.data());

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
