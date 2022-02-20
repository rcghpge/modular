//===- bef-executor.cpp - BEF Executor Driver -----------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that allows writing test cases in BEF format.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Allocator.h"
#include "LLCL/Runtime.h"
using namespace LLCL;

int main(int argc, char **argv) {
  // TODO: open the input file, autodetect both binary and source forms like
  // firtool does.

  Runtime r(createMallocAllocator());
  printf("hello bef-executor");
  return 0;
}
