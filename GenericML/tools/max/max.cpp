//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Config/Version.h"
#include "GenericML/API/modular/c/common.h"
#include "GenericML/BenchmarkCLI/Main.h"
#include "Support/CrashReporting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

#ifndef MODULAR_PRODUCTION
#include "GenericML/VisualizeCLI/Main.h"
#endif // MODULAR_PRODUCTION

using namespace M;

static constexpr const char *kMAXProductName = "max";

static void usage(llvm::raw_ostream &os) {
  os << "usage: " << kMAXProductName
     << " <subcommand> <args...>\n"
        "Available subcommands:\n"
        "  benchmark - Benchmark a model with the Modular AI Engine\n"
#ifndef MODULAR_PRODUCTION
        "  visualize - Generate a file that can be used to visualize a model "
        "with Netron.app\n"
#endif // MODULAR_PRODUCTION
      ;
}

static int handleHelp(int argc, char **argv) {
  usage(llvm::outs());
  return EXIT_SUCCESS;
}

static int handleVersion(int argc, char **argv) {
  ModularVersion version = getModularVersion();
  llvm::outs() << llvm::formatv("{0} {1}.{2}.{3}{4} ({5})\n", kMAXProductName,
                                version.major, version.minor, version.patch,
                                version.label, version.revision);
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
  initCrashpadForProgram(argv[0], kMAXProductName);
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
                     .Case("benchmark", maxBenchmarkMain)
#ifndef MODULAR_PRODUCTION
                     .Case("visualize", maxVisualizeMain)
#endif // MODULAR_PRODUCTION
                     .Default(handleUnknownSubcommand);
  return handler(argc - 1, argv + 1);
}
