//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/CompileDriver.h"
#include "Support/CommandLine.h"

using namespace M;

namespace {

struct CacheInfoCLIOptions {
  cl::opt<bool> printCacheDir{"print-cache-dir",
                              cl::desc("Print MAX cache directory")};
  cl::opt<bool> printWeightsCacheDir{
      "print-weights-cache-dir", cl::desc("Print MAX weights cache directory")};
};

} // namespace

int main(int argc, char **argv) {
  CacheInfoCLIOptions cli;
  llvm::cl::ParseCommandLineOptions(argc, argv, "MAX Cache Info Tool");

  bool any = false;
  bool ok = true;

  if (cli.printCacheDir) {
    any = true;
    if (auto cacheDir = getMaxCacheDir()) {
      llvm::outs() << *cacheDir << '\n';
    } else {
      ok = false;
      llvm::errs() << "unable to find MAX cache dir\n";
    }
  }
  if (cli.printWeightsCacheDir) {
    any = true;
    if (auto weightsCacheDir = getWeightsCacheDir()) {
      llvm::outs() << *weightsCacheDir << '\n';
    } else {
      ok = false;
      llvm::errs() << "unable to find MAX weights cache dir\n";
    }
  }

  if (!any) {
    llvm::errs()
        << "no options specified; see --help for available info requests\n";
    ok = false;
  }

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
