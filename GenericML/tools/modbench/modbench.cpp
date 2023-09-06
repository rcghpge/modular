//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Config/Version.h"
#include "GenericML/API/modular/c/common.h"
#include "Support/CommandLine.h"
#include "Support/CrashReporting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

static constexpr const char *kModbenchProductName = "modbench";

namespace {

struct ClOptions {
  cl::opt<std::string> modelPath{llvm::cl::Positional,
                                 cl::desc("Path to model file/directory")};
};

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  initCrashpadForProgram(argv[0], kModbenchProductName);
  llvm::setBugReportMsg(
      "Please submit a bug report to eap@modular.com and include the crash "
      "backtrace along with your command line invocation.\n");

  ClOptions opts;

  llvm::cl::SetVersionPrinter([](llvm::raw_ostream &os) {
    ModularVersion version = getModularVersion();
    os << llvm::formatv("{0} {1}.{2}.{3}{4} ({5})\n", kModbenchProductName,
                        version.major, version.minor, version.patch,
                        version.label, version.revision);
    os << "Modular version " << M_version() << "\n";
  });
  llvm::cl::ParseCommandLineOptions(argc, argv, kModbenchProductName);

  llvm::outs() << "Modular Benchmarking Tool\n";
  llvm::outs() << "Modular version: " << M_version() << "\n";
  llvm::outs() << "Compiling model... ";
  llvm::outs() << "OK (took " << /* TODO */ 0.0 << "s)\n";
  llvm::outs() << "Benchmarking\n";
  llvm::outs() << "Benchmark finished\n";
  llvm::outs() << "Average QPS: " << /* TODO */ 0.0 << "\n";

  return EXIT_SUCCESS;
}
