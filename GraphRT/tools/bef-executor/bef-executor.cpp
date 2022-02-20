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
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace LLCL;

static llvm::cl::opt<std::string> inputFilename( // NOLINT
    llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"));

enum class AllocatorType {
  // Allocator that just calls malloc/free.
  malloc,
  /// Allocator that does leak checking.
  leakChecker,
  /// Allocator that does profiling (and leak checking).
  profiler,
};

// Enable HostAllocator types to be specified on the command line.
static llvm::cl::opt<AllocatorType> allocatorType( // NOLINT
    "allocator", llvm::cl::desc("Specify allocator type:"),
    llvm::cl::values(clEnumValN(AllocatorType::malloc, "malloc",
                                "System malloc/free"),
                     clEnumValN(AllocatorType::leakChecker, "leak-checker",
                                "Allocator with leak checking"),
                     clEnumValN(AllocatorType::profiler, "profiler",
                                "Allocator with profiling and leak checking")),
    llvm::cl::init(AllocatorType::leakChecker));

/// Configure the memory allocator based on the command line option settings.
static std::unique_ptr<Allocator> getAllocator() {
  std::unique_ptr<Allocator> allocator;
  switch (allocatorType) {
  case AllocatorType::malloc:
    allocator = createMallocAllocator();
    llvm::outs() << "Initializing malloc allocator.\n";
    break;
  case AllocatorType::leakChecker:
    allocator = createLeakCheckAllocator(createMallocAllocator());
    llvm::outs() << "Initializing leak check allocator.\n";
    break;
  case AllocatorType::profiler:
    allocator = createProfilingAllocator(createMallocAllocator());
    llvm::outs() << "Initializing profiling allocator.\n";
    break;
  }
  llvm::outs().flush();
  return allocator;
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "bef-executor test driver\n");

  // TODO: Autodetect and translate source forms like firtool does for
  // convenience.

  // Read the input file from disk into a memory buffer.
  std::string errorMessage;
  auto fileBuffer = mlir::openInputFile(inputFilename, &errorMessage);
  if (!fileBuffer) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  mlir::MLIRContext context;

  Runtime r(getAllocator());
  printf("hello bef-executor");
  return 0;
}
