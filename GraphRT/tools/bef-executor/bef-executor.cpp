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
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/ReferenceCount.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace LLCL;

static llvm::cl::opt<std::string> inputFilename( // NOLINT
    llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"));

enum class AllocatorType {
  // Allocator that just calls malloc/free.
  kMalloc,
  /// Allocator that does leak checking.
  kLeakChecker,
  /// Allocator that does profiling (and leak checking).
  kProfiler,
};

// Enable HostAllocator types to be specified on the command line.
static llvm::cl::opt<AllocatorType> allocatorType( // NOLINT
    "allocator", llvm::cl::desc("Specify allocator type:"),
    llvm::cl::values(clEnumValN(AllocatorType::kMalloc, "malloc",
                                "System malloc/free"),
                     clEnumValN(AllocatorType::kLeakChecker, "leak-checker",
                                "Allocator with leak checking"),
                     clEnumValN(AllocatorType::kProfiler, "profiler",
                                "Allocator with profiling and leak checking")),
    llvm::cl::init(AllocatorType::kLeakChecker));

/// Configure the memory allocator based on the command line option settings.
static std::unique_ptr<Allocator> getAllocator() {
  std::unique_ptr<Allocator> allocator;
  switch (allocatorType) {
  case AllocatorType::kMalloc:
    allocator = createMallocAllocator();
    llvm::outs() << "Initializing malloc allocator.\n";
    break;
  case AllocatorType::kLeakChecker:
    allocator = createLeakCheckAllocator(createMallocAllocator());
    llvm::outs() << "Initializing leak check allocator.\n";
    break;
  case AllocatorType::kProfiler:
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

static void test1(Runtime &runtime) {
  printf("test1: raw\n");
  // Make an asyncValue
  auto *av1 = AsyncValue::createUnconstructed<int>(runtime);
  auto *av2 = AsyncValue::createConstructed<int>(runtime, 42);
  auto *av3 = AsyncValue::createReady<char>(runtime, 'x');

  // Whenever it gets completed, we print.
  av1->andThen([=]() {
    printf("hello bef-executor %d\n", av1->get<int>());
    av1->dropRef();
    av2->markReady();
  });

  // Whenever it gets completed, we print.
  av2->andThen([=]() {
    printf("bef barf: %d %c\n", av2->get<int>(), av3->get<char>());
    av2->dropRef();
    av3->dropRef();
  });

  // Add a task that completes the AV and drops its reference.
  runtime.addTask([=]() { av1->emplace<int>(1); });

  // Do all the work.
  runtime.quiesce();
}

static void test2(Runtime &runtime) {
  printf("test2: RCRef\n");
  // Make an asyncValue
  auto av1 = takeRCRef(AsyncValue::createUnconstructed<int>(runtime));
  auto av2 = takeRCRef(AsyncValue::createConstructed<int>(runtime, 42));
  auto av3 = takeRCRef(AsyncValue::createReady<char>(runtime, 'x'));

  // Whenever it gets completed, we print.
  av1->andThen([av1 = av1.copy(), av2 = av2.copy()]() {
    printf("hello bef-executor %d\n", av1->get<int>());
    av2->markReady();
  });

  // Whenever it gets completed, we print.
  av2->andThen([av2 = std::move(av2), av3 = std::move(av3)]() {
    printf("bef barf: %d %c\n", av2->get<int>(), av3->get<char>());
  });

  // Add a task that completes the AV and drops its reference.
  runtime.addTask([av1 = std::move(av1)]() { av1->emplace<int>(1); });

  // Do all the work.
  runtime.quiesce();
}

static void test3(Runtime &runtime) {
  printf("test3: AsyncValueRef\n");
  // Make an asyncValue
  auto av1 = AsyncValueRef<int>::createUnconstructed(runtime);
  auto av2 = AsyncValueRef<int>::createConstructed(runtime, 42);
  auto av3 = AsyncValueRef<char>::createReady(runtime, 'x');

  // Whenever it gets completed, we print.
  av1->andThen([av1 = av1.copy(), av2 = av2.copy()]() {
    printf("hello bef-executor %d\n", av1.get());
    av2->markReady();
  });

  // Whenever it gets completed, we print.
  av2->andThen([av2 = std::move(av2), av3 = std::move(av3)]() {
    printf("bef barf: %d %c\n", av2.get(), av3.get());
  });

  // Add a task that completes the AV and drops its reference.
  runtime.addTask([av1 = std::move(av1)]() { av1.emplace(1); });

  // Do all the work.
  runtime.quiesce();
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
  test1(runtime);
  test2(runtime);
  test3(runtime);
  return 0;
}
