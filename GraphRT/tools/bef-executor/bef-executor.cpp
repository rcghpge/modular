//===- bef-executor.cpp - BEF Executor Driver -----------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that allows writing test cases in BEF format.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFExecutor/BEFFile.h"
#include "GraphRT/BEFExecutor/Function.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/RCRef.h"
#include "LLCL/Support/ReferenceCounted.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace LLCL;
using namespace M;
using namespace llvm;

static cl::opt<std::string> inputFilename( // NOLINT
    cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::list<std::string> functions( // NOLINT
    "functions", cl::desc("Specify functions to run, defaults to all of them"),
    cl::ZeroOrMore, cl::MiscFlags::CommaSeparated);

enum class AllocatorType {
  // Allocator that just calls malloc/free.
  kMalloc,
  /// Allocator that does leak checking.
  kLeakChecker,
  /// Allocator that does profiling (and leak checking).
  kProfiler,
};

// Enable HostAllocator types to be specified on the command line.
static cl::opt<AllocatorType> allocatorType( // NOLINT
    "allocator", cl::desc("Specify allocator type:"),
    cl::values(clEnumValN(AllocatorType::kMalloc, "malloc",
                          "System malloc/free"),
               clEnumValN(AllocatorType::kLeakChecker, "leak-checker",
                          "Allocator with leak checking"),
               clEnumValN(AllocatorType::kProfiler, "profiler",
                          "Allocator with profiling and leak checking")),
    cl::init(AllocatorType::kLeakChecker));

/// Configure the memory allocator based on the command line option settings.
static std::unique_ptr<Allocator> getAllocator() {
  std::unique_ptr<Allocator> allocator;
  switch (allocatorType) {
  case AllocatorType::kMalloc:
    allocator = createMallocAllocator();
    printf("Initializing malloc allocator.\n");
    break;
  case AllocatorType::kLeakChecker:
    allocator = createLeakCheckAllocator(createMallocAllocator());
    printf("Initializing leak check allocator.\n");
    break;
  case AllocatorType::kProfiler:
    allocator = createProfilingAllocator(createMallocAllocator());
    printf("Initializing profiling allocator.\n");
    break;
  }
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

/// Run the specified BEF function as a testcase - this means passing in test
/// input and printing results if we can.
static void runFunctionTestcase(const Function *fn, Runtime &runtime) {
  // Skip anonymous functions.  These are bodies of control flow operators etc.
  if (fn->getName().empty())
    return;

  // If the function takes arguments, then we can't run it from this driver.
  // TODO: Should eventually handle chain inputs.
  if (fn->getNumArguments() != 0) {
    outs() << "--- Can not run '" << fn->getName()
           << "' because it has arguments.\n";
    outs().flush();
    return;
  }

  // Actually run the function.
  outs() << "--- Running '" << fn->getName() << "':\n";
  outs().flush();

  llvm::SmallVector<RCRef<AsyncValue>, 4> results;
  results.resize(fn->getNumResults());
  fn->execute(/*arguments=*/{}, results);

  // Block until the function results are fully resolved.
  runtime.await(results);

  // TODO: Print out the function results that we know about.

  // In this test driver, we want to make sure that every function completes
  // all execution before moving on to the next one.  This makes the leak
  // checker work better in the face of side effecting kernels that aren't
  // properly chained together (which is useful for testing).
  runtime.quiesce();

  // TODO: Verify no leaked ReferenceCounted instances or AsyncValue's!
}

int main(int argc, char **argv) {
  // Helper for printing errors below.
  auto emitToolError = [=](Twine message) -> int {
    errs() << argv[0] << ": " << message << "\n";
    return 1;
  };

  cl::ParseCommandLineOptions(argc, argv, "bef-executor test driver\n");

  LLCL::verifyNoLiveReferenceCountedObjects(
      "No objects should start allocated");

  // TODO: Autodetect and translate source forms like firtool does for
  // convenience.

  // Read the input file from disk into a memory buffer.
  std::string errorMessage;
  auto fileBuffer = mlir::openInputFile(inputFilename, &errorMessage);
  if (!fileBuffer)
    return emitToolError(errorMessage);

  // Set up all our primitives.
  PrimitiveRegistry primitives;
  primitives.add("test.something", abort);

  mlir::MLIRContext context;
  {
    Runtime runtime(getAllocator(), getWorkQueue());
    // Hard coded tests until we can run BEF files.
    test1(runtime);
    test2(runtime);
    test3(runtime);
    fflush(stdout); // Finish printf tests.

    // Open the BEF file.
    auto befFileOrError =
        BEFFile::open(fileBuffer->getBuffer(), primitives, runtime);
    if (auto *error = befFileOrError.getError()) {
      return emitToolError(" error opening '" + Twine(inputFilename) +
                           "': " + Twine(error));
      return 1;
    }
    auto befFile = std::move(befFileOrError.get());

    // Figure out which functions to execute for the test.
    std::vector<const Function *> functionsToExecute;
    // If no functions are specified on the command line. Try to run them all.
    if (functions.empty()) {
      functionsToExecute = befFile->getAllFunctions();
    } else {
      // Otherwise, run the named functions in order.
      for (auto &fnName : functions) {
        const Function *fn = befFile->getFunction(fnName);
        if (!fn)
          return emitToolError("couldn't find function '" + fnName +
                               "' to execute");
        functionsToExecute.push_back(fn);
      }
    }

    // Run each requested function as a standalone testcase.
    for (auto *fn : functionsToExecute)
      runFunctionTestcase(fn, runtime);
  }

  LLCL::verifyNoLiveReferenceCountedObjects(
      "All objects should be deallocated");

  return 0;
}
