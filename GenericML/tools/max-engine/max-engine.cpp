//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Config/Version.h"
#include "GenericML/API/modular/c/common.h"
#include "GenericML/BenchmarkCLI/Main.h"
#include "GenericML/VisualizeCLI/Main.h"
#include "Support/CrashReporting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace M;

static constexpr const char *kEngineCLIProductName = "max-engine";

static void usage(llvm::raw_ostream &os) {
  os << "usage: " << kEngineCLIProductName
     << " <subcommand> <args...>\n"
        "Available subcommands:\n"
        "  benchmark - Benchmark a model with the Modular AI Engine\n"
        "  visualize - Generate a file that can be used to visualize a model "
        "with Netron.app\n";
}

static int handleHelp(int argc, char **argv) {
  usage(llvm::outs());
  return EXIT_SUCCESS;
}

static int handleVersion(int argc, char **argv) {
  ModularVersion version = getModularVersion();
  llvm::outs() << llvm::formatv(
      "{0} {1}.{2}.{3}{4} ({5})\n", kEngineCLIProductName, version.major,
      version.minor, version.patch, version.label, version.revision);
  llvm::outs() << "Modular version " << M_version() << "\n";
  return EXIT_SUCCESS;
}

static int handleUnknownSubcommand(int argc, char **argv) {
  llvm::errs() << "Unknown subcommand `" << argv[0] << "'\n";
  usage(llvm::errs());
  return EXIT_FAILURE;
}

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  initCrashpadForProgram(argv[0], kEngineCLIProductName);
  llvm::setBugReportMsg(
      "Please submit a bug report to eap@modular.com and include the crash "
      "backtrace along with your command line invocation.\n");

  if (argc < 2) {
    llvm::errs() << "Subcommand must be provided\n";
    usage(llvm::errs());
    return EXIT_FAILURE;
  }

  using HandlerFunc = int (*)(int argc, char **argv);
  auto handler = llvm::StringSwitch<HandlerFunc>(argv[1])
                     .Case("--help", handleHelp)
                     .Case("--version", handleVersion)
                     .Case("benchmark", engineCLIBenchmarkMain)
                     .Case("visualize", engineCLIVisualizeMain)
                     .Default(handleUnknownSubcommand);
  return handler(argc - 1, argv + 1);
}
