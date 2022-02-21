//===- bef-executor.cpp - BEF Executor Driver -----------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that allows writing test cases in BEF format.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Runtime/Runtime.h"
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

/// Configure the appropriate work queue based on the command line option
/// settings.
static std::unique_ptr<WorkQueue> getWorkQueue() {
  return createSingleThreadWorkQueue();
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

  Runtime runtime(getAllocator(), getWorkQueue());

  // Make an asyncValue
  auto *asyncValue = AsyncValue::createUnconstructed<int>(runtime);
  auto *av2 = AsyncValue::createConstructed<int>(runtime, 42);
  auto *av3 = AsyncValue::createReady<char>(runtime, 'x');

  // Whenever it gets completed, we print.
  asyncValue->andThen([=]() {
    printf("hello bef-executor %d\n", asyncValue->get<int>());
    asyncValue->dropRef();
    av2->markReady();
  });

  // Whenever it gets completed, we print.
  av2->andThen([=]() {
    printf("bef barf: %d %c\n", av2->get<int>(), av3->get<char>());
    av2->dropRef();
    av3->dropRef();
  });

  // Add a task that completes the AV and drops its reference.
  runtime.addTask([=]() {
    asyncValue->emplace<int>(1);
  });

  // Do all the work.
  runtime.quiesce();

  return 0;
}
