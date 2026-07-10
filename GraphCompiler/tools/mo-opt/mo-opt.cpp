//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/CompilerSupport/Context.h"
#include "GraphCompiler/GraphCompiler/GraphCompiler.h"
#include "Init/Init.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "MLRT/GraphRT/MEFDialect/MEFDialect.h"
#include "Support/Context.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Support/Timing.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
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
  registry.insert<mlir::func::FuncDialect, MEF::MEFDialect>();

  MO::registerPasses();
  mlir::registerDefaultTimingManagerCLOptions();

  // Create our context.
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "mo-opt", Init::Options().withCPUDeviceOptions(
                    AsyncRT::CPUDeviceOptions().withLeakCheckedAllocator()));
  if (ctxOr.isError()) {
    llvm::errs() << "failed to create context: " << ctxOr.getError() << "\n";
    return 1;
  }
  registerContext(registry, *ctxOr);

  static llvm::cl::opt<bool> timeTrace{
      "time-trace",
      llvm::cl::desc("Turn on time profiler. Generates JSON file "
                     "called kgen.trace.json in the derived directory. "
                     "Currently integrated mainly for KGEN compilation.")};

  static llvm::cl::opt<int> timeTraceGranularity{
      "time-trace-granularity",
      llvm::cl::desc("Minimum time granularity (in microseconds) "
                     "traced by time profiler."),
      llvm::cl::init(0)};

  std::string inputFilename, outputFilename;
  std::tie(inputFilename, outputFilename) =
      registerAndParseCLIOptions(argc, argv, "mo optimizer driver", registry);
  KGEN::TraceProfiler tracer(timeTrace, timeTraceGranularity);
  return failed(
      MlirOptMain(argc, argv, inputFilename, outputFilename, registry));
}
