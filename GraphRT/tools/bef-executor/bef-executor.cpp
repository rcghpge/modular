//===- bef-executor.cpp ---------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that allows writing test cases in BEF format.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFExecutor/BEFFile.h"
#include "GraphRT/BEFExecutor/Invokable.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "GraphRT/Primitives.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/Chain.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

using namespace LLCL;
using namespace M;
using namespace llvm;

enum class AllocatorType {
  // Allocator that just calls malloc/free.
  kMalloc,
  /// Allocator that does leak checking.
  kLeakChecker,
  /// Allocator that does profiling (and leak checking).
  kProfiler,
};

enum class WorkQueueType {
  /// WorkQueue that just uses a single thread and performs no synchronization.
  kSingleThread,
  /// WorkQueue that uses a configurable-size thread pool.
  kThreadPool,
};

/// Configure the memory allocator based on the command line option settings.
static std::unique_ptr<Allocator> getAllocator(AllocatorType allocatorType) {
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

  fflush(stdout);
  return allocator;
}

/// Configure the appropriate work queue based on the command line option
/// settings.
static std::unique_ptr<WorkQueue> getWorkQueue(WorkQueueType workqueueType) {
  std::unique_ptr<WorkQueue> queue;
  switch (workqueueType) {
  case WorkQueueType::kSingleThread:
    queue = createSingleThreadWorkQueue();
    printf("Initializing single thread work queue.\n");
    break;
  case WorkQueueType::kThreadPool:
    queue = createThreadPoolWorkQueue();
    printf("Initializing thread pool work queue.\n");
    break;
  }

  fflush(stdout);
  return queue;
}

/// Run the specified bef.invokable as a testcase - this means passing in test
/// input and printing results if we can.
static void runInvokableTestcase(const Invokable *invokable,
                                 AllocatorType allocatorType,
                                 WorkQueueType workqueueType,
                                 MLIRContext *context) {
  // Skip anonymous invokables.  These are bodies of control flow operators etc.
  if (invokable->getName().empty())
    return;

  auto &os = outs();

  // If the invokable takes arguments, then we can't run it from this driver.
  if (invokable->getNumArgs() != 0) {
    os << "--- Can not run '" << invokable->getName()
       << "' because it has arguments.\n";
    os.flush();
    return;
  }

  // If we are leak checking, remember how many AsyncValue's we started with.
  ssize_t numStartingLiveAsyncValues;
  if (AsyncValue::isAllocationTrackingEnabled())
    numStartingLiveAsyncValues = AsyncValue::getNumAllocatedInstances();

  {
    Runtime runtime(getAllocator(allocatorType), getWorkQueue(workqueueType));

    // Actually run the invokable.
    os << "--- Running '" << invokable->getName() << "':\n";
    os.flush();

    llvm::SmallVector<AnyAsyncValueRef, 4> results;
    results.resize(invokable->getNumResults());
    invokable->invoke(/*arguments=*/{}, results, runtime);

    // Block until the invokable results are fully resolved.
    await(results);

    // If anything printed to stdout, force it out to a stream, since we will be
    // printing with raw_ostream.
    fflush(stdout);

    // Check all the results for errors, and if we find any of them, emit them
    // through MLIR error handling.  We use MLIR here just because it makes
    // diagnostic verification easy.  We do this before printing the results
    // so the diagnostics and result printing don't get intermixed.
    for (auto &result : results) {
      if (auto *diag = result->getDiagnosticIfPresent()) {
        auto loc = diag->decodeLocation();
        auto mlirLoc = mlir::FileLineColLoc::get(context, loc.filename,
                                                 loc.line, loc.column);
        mlir::emitError(mlirLoc) << diag->getMessage();
      }
    }

    // Print out the invokable results that we know about.
    // Go ahead and print out the invokable results that we know about.
    if (!results.empty()) {
      os << "'" << invokable->getName() << "' returned ";
      llvm::interleaveComma(results, os, [&](const AnyAsyncValueRef &result) {
        // Handle error value by printing it inline, and emitting it as an error
        // through SourceMgr so we can validate it.
        if (auto *diag = result->getDiagnosticIfPresent()) {
          os << "<error: " << diag->getMessage() << '>';
          return;
        }

#define HANDLE_TYPE(TYPE)                                                      \
  if (result->isType<TYPE>()) {                                                \
    os << result->get<TYPE>();                                                 \
    return;                                                                    \
  }
        HANDLE_TYPE(bool)
        HANDLE_TYPE(int8_t)
        HANDLE_TYPE(int16_t)
        HANDLE_TYPE(int32_t)
        HANDLE_TYPE(int64_t)
        HANDLE_TYPE(uint8_t)
        HANDLE_TYPE(uint16_t)
        HANDLE_TYPE(uint32_t)
        HANDLE_TYPE(uint64_t)
        HANDLE_TYPE(float)
        HANDLE_TYPE(double)
        HANDLE_TYPE(std::string)
        if (result->isType<Chain>())
          os << "chain";
        else
          os << "<unknown type>";
      });
      os << '\n';
      os.flush();
      results.clear();
    }

    // In this test driver, we want to make sure that every invokable completes
    // all execution before moving on to the next one.  This makes the leak
    // checker work better in the face of side effecting kernels that aren't
    // properly chained together (which is useful for testing), so we close a
    // scope here so the Runtime object is destroyed.
  }

  // Make sure we're not leaking AsyncValues.
  if (AsyncValue::isAllocationTrackingEnabled()) {
    auto numLiveAsyncValues = AsyncValue::getNumAllocatedInstances();
    if (numLiveAsyncValues != numStartingLiveAsyncValues) {
      os << "Evaluation of invokable '" << invokable->getName() << "' leaked "
         << (numLiveAsyncValues - numStartingLiveAsyncValues)
         << " async values (before: " << numStartingLiveAsyncValues
         << ", after: " << numLiveAsyncValues << ")!\n";
      abort();
    }
  }
}

//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input file>"),
                                     cl::init("-"));
  cl::list<std::string> invokables(
      "invokables",
      cl::desc("Specify invokables to run, defaults to all of them"),
      cl::ZeroOrMore, cl::MiscFlags::CommaSeparated);
  // Enable HostAllocator types to be specified on the command line.
  cl::opt<AllocatorType> allocatorType(
      "allocator", cl::desc("Specify allocator type:"),
      cl::values(
          clEnumValN(AllocatorType::kMalloc, "malloc", "System malloc/free"),
          clEnumValN(AllocatorType::kLeakChecker, "leak-checker",
                     "Allocator with leak checking"),
          clEnumValN(AllocatorType::kProfiler, "profiler",
                     "Allocator with profiling and leak checking")),
      cl::init(AllocatorType::kLeakChecker));
  // Allow the command-line user to specify the type of work que to be used.
  cl::opt<WorkQueueType> workqueueType(
      "workqueue", cl::desc("Specify work queue type:"),
      cl::values(clEnumValN(WorkQueueType::kSingleThread, "single",
                            "Single thread, no synchronization"),
                 clEnumValN(WorkQueueType::kThreadPool, "threadpool",
                            "Configurable size thread pool")),
      cl::init(WorkQueueType::kSingleThread));
  static cl::opt<bool> verifyDiagnostics(
      "verify-diagnostics",
      cl::desc("Check that emitted diagnostics match "
               "expected-* lines on the corresponding line"),
      cl::init(false));

  cl::ParseCommandLineOptions(argc, argv, "bef-executor test driver\n");

  LLCL::verifyNoLiveReferenceCountedObjects(
      "No objects should start allocated");

  // Helper for printing errors below.
  auto emitToolError = [=](Twine message) -> int {
    errs() << argv[0] << ": " << message << "\n";
    return 1;
  };

  // TODO: Autodetect and translate source forms like firtool does for
  // convenience.

  // Read the input file from disk into a memory buffer.
  std::string errorMessage;
  auto fileBuffer = mlir::openInputFile(inputFilename, &errorMessage);
  if (!fileBuffer)
    return emitToolError(errorMessage);

  // Set up all our primitives.
  PrimitiveRegistry primitives;
  registerAllGraphRTPrimitives(primitives);
  primitives.finalize();

  assert(AsyncValue::getNumAllocatedInstances() == 0 &&
         "We have async values allocated before we started to do anything");

  // Use an LLVM SourceMgr so we can emit caret diagnostics, and support
  // diagnostic verification.
  auto bufferText = fileBuffer->getBuffer();
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(fileBuffer), llvm::SMLoc());

  bool hadToolError = false;

  mlir::MLIRContext context;
  {
    // Open the BEF file.
    auto befFileOrError = BEFFile::open(bufferText, primitives);
    if (auto *error = befFileOrError.getError())
      return emitToolError(" error opening '" + Twine(inputFilename) +
                           "': " + Twine(error));

    auto befFile = std::move(befFileOrError.get());

    // Figure out which invokables to execute for the test.
    std::vector<const Invokable *> invokablesToExecute;
    // If no invokables are specified on the command line. Try to run them all.
    if (invokables.empty()) {
      invokablesToExecute = befFile->getAllInvokables();
    } else {
      // Otherwise, run the named invokables in order.
      for (auto &fnName : invokables) {
        const Invokable *invokable = befFile->getInvokable(fnName);
        if (!invokable)
          return emitToolError("couldn't find invokable '" + fnName +
                               "' to execute");
        invokablesToExecute.push_back(invokable);
      }
    }

    // Run each requested invokable as a standalone testcase.  We register
    // different diagnostic handlers with MLIR depending on whether we're in
    // -verify-diagnostics mode or not.
    if (verifyDiagnostics) {
      mlir::SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr,
                                                                &context);
      for (auto *fn : invokablesToExecute)
        runInvokableTestcase(fn, allocatorType, workqueueType, &context);

      hadToolError = failed(sourceMgrHandler.verify());
    } else {
      mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
      for (auto *fn : invokablesToExecute)
        runInvokableTestcase(fn, allocatorType, workqueueType, &context);
    }
  }

  assert(AsyncValue::getNumAllocatedInstances() == 0 &&
         "All async values should be cleaned up at the end");

  LLCL::verifyNoLiveReferenceCountedObjects(
      "All refcounted objects should be deallocated");

  return hadToolError;
}
