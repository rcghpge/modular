//===- bef-executor.cpp ---------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is a command line driver to test GraphRT Primitives.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFExecutor/BEFFile.h"
#include "GraphRT/BEFExecutor/BEFInvokable.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "GraphRT/GRTTestcase.h"
#include "GraphRT/Primitives.h"
#include "LLCL/Runtime/RuntimeCLOptions.h"
#include "Support/CommandLine.h"
#include "Support/CommonCLOptions.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/Support/SourceMgr.h"
#include "LLCL/Runtime/Algorithms.h"

using namespace LLCL;
using namespace M;
using mlir::MLIRContext;

/// This method can be used to print the results of `invokable` in the
/// standard format.
static void printResults(BEFInvokable *invokable,
                         ArrayRef<AnyAsyncValueRef> results,
                         MLIRContext *context, raw_ostream &os) {
  // Check all the results for errors, and if we find any of them, emit them
  // through MLIR error handling.  We use MLIR here just because it makes
  // diagnostic verification easy.  We do this before printing the results
  // so the diagnostics and result printing don't get intermixed.
  for (auto &result : results) {
    if (auto *diag = result->getDiagnosticIfPresent()) {
      auto loc = diag->decodeLocation();
      auto mlirLoc =
          FileLineColLoc::get(context, loc.filename, loc.line, loc.column);
      mlir::emitError(mlirLoc) << diag->getMessage();
    }
  }

  // Print out the invokable results that we know about.
  // Go ahead and print out the invokable results that we know about.
  if (!results.empty()) {
    os << "'" << invokable->getSymbolName() << "' returned ";
    llvm::interleaveComma(results, os, [&](const AnyAsyncValueRef &result) {
      // Handle error value by printing it inline, and emitting it as an error
      // through SourceMgr so we can validate it.
      if (auto *diag = result->getDiagnosticIfPresent()) {
        os << "<error: " << diag->getMessage() << '>';
        return;
      }

      // Otherwise we support a set of standard types for convenience.
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
  }
}

//===----------------------------------------------------------------------===//
// BEFExecutorMain implementation
//===----------------------------------------------------------------------===//

/// Run the specified bef.invokable as a testcase - this means passing in test
/// input and printing results if we can.
static void runTestcase(const GRTTestcase &test,
                        const RuntimeCLOptions &runtimeConfig,
                        MLIRContext *context) {

  // In this test driver, we want to make sure that every invokable completes
  // all execution before moving on to the next one.  This makes the leak
  // checker work better in the face of side effecting kernels that aren't
  // properly chained together (which is useful for testing).  As such, we run
  // each testcase with its own runtime instance and own leak checker.
  runtimeConfig.runWithLeakCheckedRuntime(
      test.getName().str().c_str(), [&](Runtime runtime) {
        auto &os = llvm::outs();

        // Actually run the invokable.
        os << "--- Running '" << test.getName() << "':\n";
        os.flush();

        auto results = test.getBody()->invoke(/*arguments=*/{}, runtime);

        // Block until the invokable results are fully resolved.
        await(results);

        // If anything printed to stdout, force it out to a stream, since we
        // will be printing with raw_ostream.
        fflush(stdout);

        printResults(test.getBody(), results, context, os);
      });
}

namespace {
struct BEFExecutorCLOptions : public CommonCLOptions, public RuntimeCLOptions {
  cl::list<std::string> testcases{
      "tests",
      cl::desc("Specify testcases to run, defaults to all of them. Provide a "
               "comma-separated list like --tests=foo,bar."),
      llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated};

  cl::opt<bool> verbose{"verbose",
                        cl::desc("Print configuration information at startup"),
                        cl::init(false)};
};
} // namespace

int main(int argc, char **argv) {
  // Set up all our primitives.
  PrimitiveRegistry primitives;
  registerAllGraphRTPrimitives(primitives);
  primitives.finalize();

  BEFExecutorCLOptions clOptions;

  llvm::cl::ParseCommandLineOptions(argc, argv, "bef-executor test driver\n");

  LLCL::verifyNoLiveReferenceCountedObjects(
      "No objects should start allocated");

  assert(AsyncValue::getNumAllocatedInstances() == 0 &&
         "We have async values allocated before we started to do anything");

  // Read the input file from disk into a memory buffer.
  auto fileBuffer = clOptions.openInputFileOrExit(argv[0]);

  // Print some informative messages about allocator and threadpool config.
  if (clOptions.verbose) {
    printf("%s: ", argv[0]);
    switch (clOptions.allocatorType) {
    case AllocatorType::kMalloc:
      printf("Using malloc allocator.\n");
      break;
    case AllocatorType::kLeakChecker:
      printf("Using leak check allocator.\n");
      break;
    case AllocatorType::kProfiler:
      printf("Using profiling allocator.\n");
      break;
    }
    printf("%s: ", argv[0]);
    switch (clOptions.numThreads) {
    case 1:
      printf("Using single thread work queue.\n");
      break;
    default:
      printf("Using thread pool work queue.\n");
      break;
    }
    fflush(stdout);
  }

  auto bufferText = fileBuffer->getBuffer();
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(fileBuffer), llvm::SMLoc());

  auto befFileOrError = BEFFile::open(bufferText, primitives);
  if (auto *error = befFileOrError.getError()) {
    llvm::errs() << argv[0] << ": error opening '" << clOptions.inputFilename
                 << "': " << error << '\n';
    return 1;
  }

  auto befFile = befFileOrError.takeValue();

  // Figure out which testcases we want to execute.
  SmallVector<GRTTestcase, 1> testcasesToExecute;

  if (auto err = befFile->walkTopLevelPrimitives<GRTTestcase>(
          [&](GRTTestcase test) -> ErrorOrSuccess {
            // If no testcases are specified, run them all.
            if (clOptions.testcases.empty() ||
                llvm::is_contained(clOptions.testcases, test.getName()))
              testcasesToExecute.push_back(test);
            return success();
          })) {
    llvm::errs() << err.getError() << "\n";
    return {};
  }

  // Run each test with a MLIRContext set up for the right kind of diagnostic
  // emission.
  LogicalResult hadError = clOptions.configureMLIRContextAndExecute(
      sourceMgr, [&](MLIRContext *context) {
        for (auto &fn : testcasesToExecute)
          runTestcase(fn, clOptions, context);
      });

  // We're done with the BEF, we can safely deallocate.
  befFile.reset();

  assert(AsyncValue::getNumAllocatedInstances() == 0 &&
         "All async values should be cleaned up at the end");
  verifyNoLiveReferenceCountedObjects(
      "All refcounted objects should be deallocated");
  return failed(hadError);
}
