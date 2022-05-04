//===- bef-executor.cpp ---------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is a command line driver to test GraphRT Primitives.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "GraphRT/Primitives.h"
#include "GraphRT/Tools/bef-executor/BEFExecutorMain.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/Support/CommandLine.h"

using namespace M;

int main(int argc, char **argv) {
  // Set up all our primitives.
  PrimitiveRegistry primitives;
  registerAllGraphRTPrimitives(primitives);
  primitives.finalize();

  M::registerBEFExecutorCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "bef-executor test driver\n");

  return BEFExecutorMain(argc, argv, "bef-executor test driver\n", primitives);
}
