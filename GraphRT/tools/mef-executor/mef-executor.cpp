//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is a command line driver to test GraphRT Primitives.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/GRTTestcase.h"
#include "GraphRT/MEFExecutor/MEFFile.h"
#include "GraphRT/MEFExecutor/MEFInvokable.h"
#include "GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "GraphRT/Primitives.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/RuntimeCLOptions.h"
#include "LLCL/Runtime/Utils.h"
#include "Support/CommandLine.h"
#include "Support/CommonCLOptions.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/ML/SizeUtils.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::LLCL;
using mlir::MLIRContext;

/// This method can be used to print the results of `invokable` in the
/// standard format.
static void printResults(MEFInvokable *invokable,
                         ArrayRef<AnyAsyncValueRef> results,
                         MLIRContext *context, raw_ostream &os) {
  // Check all the results for errors, and if we find any of them, emit them
  // through MLIR error handling.  We use MLIR here just because it makes
  // diagnostic verification easy.  We do this before printing the results
  // so the diagnostics and result printing don't get intermixed.
  for (auto &result : results) {
    if (auto *diag = result.getDiagnosticIfPresent()) {
      auto loc = diag->decodeLocation();
      auto mlirLoc =
          FileLineColLoc::get(context, loc.filename, loc.line, loc.column);
      mlir::emitError(mlirLoc) << diag->getMessage();
    }
  }

  // Print out the invokable results that we know about.
  // Go ahead and print out the invokable results that we know about.
  if (!results.empty()) {
    llvm::interleaveComma(results, os, [&](const AnyAsyncValueRef &result) {
      // Handle error value by printing it inline, and emitting it as an error
      // through SourceMgr so we can validate it.
      if (auto *diag = result.getDiagnosticIfPresent()) {
        os << "<error: " << diag->getMessage() << '>';
        return;
      }

      // Otherwise we support a set of standard types for convenience.
#define HANDLE_TYPE(TYPE)                                                      \
  if (result.isType<TYPE>()) {                                                 \
    os << result.get<TYPE>();                                                  \
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
      if (result.isType<Chain>())
        os << "chain";
      else
        os << "<unknown type>";
    });
    os << '\n';
    os.flush();
  }
}

//===----------------------------------------------------------------------===//
// MEFExecutorMain implementation
//===----------------------------------------------------------------------===//

/// Run the specified mef.invokable as a testcase - this means passing in test
/// input and printing results if we can.
static ErrorOrSuccess runTestcase(const GRTTestcase &test,
                                  const RuntimeCLOptions &runtimeConfig,
                                  MLIRContext *context) {
  // In this test driver, we want to make sure that every invokable completes
  // all execution before moving on to the next one.  This makes the leak
  // checker work better in the face of side effecting kernels that aren't
  // properly chained together (which is useful for testing).  As such, we run
  // each testcase with its own runtime instance and own leak checker.
  runWithLeakCheckedRuntime(
      runtimeConfig, test.getName().str().c_str(), [&](Runtime &runtime) {
        // Actually run the invokable.  We use printf here because it has atomic
        // output behavior w.r.t. the printfs in various TestPrimitives.
        printf("--- Running '%.*s':\n", (int)test.getName().size(),
               test.getName().data());
        fflush(stdout);

        auto results = test.getBody()->createResultValuesAndInvoke(
            /*arguments=*/{}, runtime);

        // Block until the invokable results are fully resolved.
        await(results);

        // If anything printed to stdout, force it out to a stream, since we
        // will be printing with raw_ostream.
        fflush(stdout);

        auto &os = llvm::outs();
        if (!results.empty())
          os << "'" << test.getName() << "' returned ";

        printResults(test.getBody(), results, context, os);
      });
  // FIXME: this function currently does not handle errors.
  return success();
}

namespace {
class MEFExecutorCLOptions : public CommonCLOptions, public RuntimeCLOptions {
public:
  using CommonCLOptions::CommonCLOptions;

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
  MEFExecutorCLOptions clOptions(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "mef-executor test driver\n");

  // Set up all our primitives.
  PrimitiveRegistry primitives;
  registerAllGraphRTPrimitives(primitives);
  primitives.finalize();

  LLCL::verifyNoLiveReferenceCountedObjects(
      "No objects should start allocated");

  if constexpr (AsyncValue::isAllocationTrackingEnabled())
    assert(AsyncValue::getNumAllocatedInstances() == 0 &&
           "We have async values allocated before we started to do anything");

  // Read the input file from disk into a memory buffer.
  auto fileBuffer = clOptions.openInputFileOrExit(maxAttributeAlignment());

  // Print some informative messages about allocator and threadpool config.
  if (clOptions.verbose) {
    printf("%s ", argv[0]);
    clOptions.printRuntimeConfig();
    fflush(stdout);
  }

  auto mefFileOrError = MEFFile::open(fileBuffer->getBuffer(), primitives);
  if (failed(mefFileOrError))
    return clOptions.reportError(Twine("error opening '") +
                                 clOptions.inputFilename +
                                 "': " + mefFileOrError.getError());

  auto mefFile = mefFileOrError.takeValue();

  // Figure out which testcases we want to execute.
  SmallVector<GRTTestcase, 1> testcasesToExecute;

  auto *moduleInv = mefFile->getTopLevelInvokable();
  if (auto err = moduleInv->walkPrimitives<GRTTestcase>([&](GRTTestcase test) {
        // If no testcases are specified, run them all.
        if (clOptions.testcases.empty() ||
            llvm::is_contained(clOptions.testcases, test.getName()))
          testcasesToExecute.push_back(test);
      })) {
    return clOptions.reportError(err.getError());
  }

  // Run each test with a MLIRContext set up for the right kind of diagnostic
  // emission.
  LogicalResult hadError = clOptions.configureMLIRContextAndExecute(
      std::move(fileBuffer), [&](MLIRContext *context) -> LogicalResult {
        for (auto &fn : testcasesToExecute) {
          if (auto err = runTestcase(fn, clOptions, context)) {
            clOptions.reportError(err.getError());
            if (clOptions.stopOnFirstError())
              return failure();
          }
        }
        return success();
      });

  // We're done with the MEF, we can safely deallocate.
  mefFile.reset();

  if constexpr (AsyncValue::isAllocationTrackingEnabled())
    assert(AsyncValue::getNumAllocatedInstances() == 0 &&
           "All async values should be cleaned up at the end");
  verifyNoLiveReferenceCountedObjects(
      "All refcounted objects should be deallocated");
  return failed(hadError);
}
