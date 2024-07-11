//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Init/Init.h"
#include "Config/Version.h"
#include "GenericML/API/max/c/common.h"
#include "SDK/EngineCLI/Benchmark/Main.h"
#include "SDK/EngineCLI/Visualize/Main.h"
#include "Support/CrashReporting/CrashReporting.h"
#include "Support/Driver/DriverSupport.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

#define DRIVER_OPTIONS_PATH "DriverOptions.inc"
#include "Support/Driver/OptTable.inc"

using namespace M;

static constexpr const char *kMAXProductName = "max";

namespace {
struct DriverOptTable : public llvm::opt::PrecomputedOptTable {
  DriverOptTable() : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};
} // namespace

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
  llvm::errs() << "Try " << kMAXProductName
               << " --help to see available subcommands.\n";
  return EXIT_FAILURE;
}

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::setBugReportMsg(
      "Please submit a bug report to https://github.com/modularml/max/issues "
      "and include the crash backtrace along with your command line "
      "invocation.\n");

  // Create our context. In the future, this could be used to automatically
  // gate entitlements, force re-authentication, etc. This also enables crash
  // reporting by default with the given settings.
  auto ctxOr = Init::createContext(
      "max", Init::Options().withEntitlementPolicy(Settings::kAlwaysSucceed));
  if (ctxOr.isError()) {
    llvm::errs() << "could not create context: " << ctxOr.getError() << "\n";
    return EXIT_FAILURE;
  }

  State state(kMAXProductName, {argv + 1, static_cast<size_t>(argc) - 1});
  if (state.arguments.empty())
    return state.reportError("Subcommand must be provided\n"
                             "Try " +
                             Twine(kMAXProductName) +
                             " --help to see available subcommands.");
  unsigned index = 0;
  DriverOptTable optTable;
  llvm::opt::InputArgList args(state.arguments.begin(), state.arguments.end());
  auto firstArg = optTable.ParseOneArg(args, index);
  switch (firstArg->getOption().getID()) {
  case options::OPT_help:
    return state.printHelp(
#include "DriverOptionsHelpText.inc"
    );
  case options::OPT_version:
    return handleVersion(argc - index, argv + index);
  case options::OPT_INPUT:
    // Usual case.  Break out and handle below.
    break;
  default:
    return state.reportError("unrecognized argument `" +
                             firstArg->getSpelling() + "'");
  }

  using HandlerFunc = int (*)(int argc, char **argv);
  auto handler = llvm::StringSwitch<HandlerFunc>(firstArg->getAsString(args))
                     .Case("benchmark", maxBenchmarkMain)
                     .Case("visualize", maxVisualizeMain)
                     .Default(handleUnknownSubcommand);
  return handler(argc - index, argv + index);
}
