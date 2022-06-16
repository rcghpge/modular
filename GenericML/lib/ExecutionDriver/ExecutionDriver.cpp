//===- ExecutionDriver.cpp ------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "DriverInternals.h"

#include "GenericML/Kernels.h"
#include "GenericML/Support/Model.h"
#include "GenericML/Support/Tensor.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/AllocatorType.h"
#include "Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>

using namespace M;
using llvm::errs;
using llvm::outs;
using mlir::MLIRContext;

/// This method can be used to print the results of a model in the standard
/// format for an executor tool.  It handles converting framework specific
/// tensor types into something we can print if needed, and handles
/// printing/reporting of errors in a way that works with diagnostic
/// verification.
static void printTensorResults(const ExecutableModel *model,
                               MutableArrayRef<AnyAsyncValueRef> results,
                               MLIRContext *context,
                               Kernel::ResultOutputStyle outputStyle,
                               raw_ostream &os) {
  StringRef name = model->getModel()->getName();

  // Check that any tensors we get are generic tensors, not framework
  // specific tensors.  We know that all tensors have been awaited here.
  for (auto &result : results) {
    assert(result->isReady() && "We awaited this, it should be ready");
    // If we have an error or Tensor result, then we're good.
    if (result->isError() || result->isType<Tensor>())
      continue;

    // Otherwise, we must have a framework specific tensor, convert it to a
    // generic one (or get an error).
    result = model->convertToGenericTensor(result);
    await(result);
  }

  // Check all the results for errors, and if we find any of them, emit them
  // through MLIR error handling.  We use MLIR here just because it makes
  // diagnostic verification easy.  We do this before printing the results
  // so the diagnostics and result printing don't get intermixed.
  for (auto &result : results) {
    if (auto *diag = result->getDiagnosticIfPresent()) {
      auto loc = diag->decodeLocation();
      auto mlirLoc = mlir::FileLineColLoc::get(context, loc.filename, loc.line,
                                               loc.column);
      mlir::emitError(mlirLoc) << diag->getMessage();
    }
  }

  // Print out the invokable results that we know about.
  // Go ahead and print out the invokable results that we know about.
  if (!results.empty()) {
    os << "'" << name << "' returned ";
    llvm::interleaveComma(results, os, [&](const AnyAsyncValueRef &result) {
      // Handle error value by printing it inline, and emitting it as an error
      // through SourceMgr so we can validate it.
      if (auto *diag = result->getDiagnosticIfPresent()) {
        os << "<error: " << diag->getMessage() << '>';
        return;
      }

      // Handle tensor types.
      if (auto *tensor = result->dyn_cast<Tensor>()) {
        os.flush();
        if (auto err = Kernel::printTensor(tensor->getPointer(),
                                           tensor->getSpec(), outputStyle))
          os << "<error: " << err.getError() << '>';
        return;
      }

      os << "<unknown type>";
    });
    os << '\n';
    os.flush();
  }
}

static void executeModel(MLIRContext *ctx, const Model *model, Runtime &runtime,
                         const ExecutorDriverCLOptions &options) {

  auto errorOrExecutableModel = model->setup(runtime);
  if (failed(errorOrExecutableModel)) {
    outs() << errorOrExecutableModel.takeError();
    return;
  }

  auto executableModel = errorOrExecutableModel.takeValue();

  auto inputTensorsOr =
      initializeInputMemory(options.inputGenKind, model, runtime);
  if (failed(inputTensorsOr)) {
    outs() << inputTensorsOr.takeError() << "\n";
    return;
  }

  SmallVector<Tensor, 3> inputTensors = inputTensorsOr.takeValue();

  // Measure the time taken for each inference.
  std::vector<double> cpuTimesMillis;
  for (unsigned i = 0, e = options.numRuns; i != e; ++i) {
    auto start = std::chrono::steady_clock::now();

    // Fill the arguments.  If we are running the model once, we pass in the
    // input tensors directly (the model will deallocate them when it is done
    // with them).  If not, we pass in an immortal reference to the tensor data
    // so the input buffers live across all executions.
    SmallVector<AnyAsyncValueRef> args;
    args.reserve(model->getNumInputs());
    for (auto &input : inputTensors) {
      Tensor t = options.numRuns == 1 ? std::move(input) : input.getImmortal();
      args.push_back(AsyncValueRef<Tensor>::createReady(runtime, std::move(t)));
    }

    outs() << "--- Running '" << model->getName() << "':\n";
    outs().flush();

    // Execute the model and insert its results into the results vec.
    auto results = executableModel->execute(args);

    // Wait for the model to finish execution, forcing synchrony.
    await(results);

    // Print the result tensors with the default printer.
    MutableArrayRef resultRef(results.data(), results.size());
    printTensorResults(executableModel.get(), resultRef.drop_front(), ctx,
                       options.resultOutputStyle, outs());

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> time = end - start;
    cpuTimesMillis.push_back(time.count());
  }
}

/// Run `model` using the command line options flags configured by the user.
void M::runModel(MLIRContext *ctx, const Model *model, LLCL::Runtime &runtime,
                 const ExecutorDriverCLOptions &options) {
  if (!options.benchmarkMode)
    return executeModel(ctx, model, runtime, options);

  benchmarkModel(model, runtime, options.benchmarkName, options.inputGenKind,
                 options.benchmarkScenario, options.benchmarkOutputDir,
                 options.benchmarkNoLogPrefix);
}
