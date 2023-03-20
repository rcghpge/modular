//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/API/APIExecutorCommon.h"
#include "GenericML/API/APIValidatorCLOptions.h"

int main(int argc, char **argv) {
  M::APICLOptions clOptions(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv, "generic c-api test driver\n");
  auto successOr = M::runWithAPI(clOptions);
  if (successOr.isError()) {
    llvm::errs() << successOr.getError() << "\n";
    return 1;
  }
  return 0;
}
