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
#include "Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>

using namespace M;
using llvm::outs;
using mlir::MLIRContext;

/// This method can be used to print the results of a model in the standard
/// format for an executor tool.  It handles converting framework specific
/// tensor types into something we can print if needed, and handles
/// printing/reporting of errors in a way that works with diagnostic
/// verification.
static LogicalResult
printTensorResults(const ExecutableModel *model,
                   MutableArrayRef<AnyAsyncValueRef> results,
                   MLIRContext *context, Kernel::ResultOutputStyle outputStyle,
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
  bool hadError = false;
  if (!results.empty()) {
    os << "'" << name << "' returned ";
    llvm::interleaveComma(results, os, [&](const AnyAsyncValueRef &result) {
      // Handle error value by printing it inline, and emitting it as an error
      // through SourceMgr so we can validate it.
      if (auto *diag = result->getDiagnosticIfPresent()) {
        os << "<error: " << diag->getMessage() << '>';
        hadError = true;
        return;
      }

      // Handle tensor types.
      if (auto *tensor = result->dyn_cast<Tensor>()) {
        os.flush();
        if (auto err = Kernel::printTensor(tensor->getPointer(),
                                           tensor->getSpec(), outputStyle)) {
          os << "<error: " << err.getError() << '>';
          hadError = true;
        }
        return;
      }

      os << "<unknown type>";
    });
    os << '\n';
    os.flush();
  }
  return failure(hadError);
}

/// Evaluate the model once and print the output tensor.
static ErrorOrSuccess executeModelOnce(MLIRContext *ctx, const Model *model,
                                       Runtime &runtime,
                                       const ExecutorDriverCLOptions &options) {
  UNWRAP_ERROR(executableModel, model->setup(runtime));
  UNWRAP_ERROR(inputTensors,
               initializeInputMemory(options.inputGenKind, model, runtime));

  // Fill the arguments.  We are running the model once so we pass in the input
  // tensors directly (the model will deallocate them when it is done with
  // them).
  SmallVector<AnyAsyncValueRef> args;
  args.reserve(model->getNumInputs());
  for (auto &input : inputTensors)
    args.push_back(
        AsyncValueRef<Tensor>::createReady(runtime, std::move(input)));

  outs() << "--- Running '" << model->getName() << "':\n";
  outs().flush();

  // Execute the model and insert its results into the results vec.
  auto results = executableModel->execute(args);

  // There is nothing to do if there is no results.
  if (results.empty())
    return success();

  // Wait for the model to finish execution, forcing synchrony.
  await(results);

  // Print the result tensors with the default printer.
  MutableArrayRef resultRef(results.data(), results.size());
  if (failed(printTensorResults(executableModel.get(), resultRef.drop_front(),
                                ctx, options.resultOutputStyle, outs())))
    return Error("model execution failed");
  return success();
}

/// Repeatedly evaluate the model (numRuns to be exact) and capture and print
/// the runtime for each evaluation.
static ErrorOrSuccess
executeModelRepeated(MLIRContext *ctx, const Model *model, Runtime &runtime,
                     const ExecutorDriverCLOptions &options) {
  UNWRAP_ERROR(executableModel, model->setup(runtime));

  UNWRAP_ERROR(inputTensors,
               initializeInputMemory(options.inputGenKind, model, runtime));

  // Run the model numWarmup times to warmup the execution. This is performed to
  // remove any added overhead due to initial setup of the model. The loop is
  // only run if numRuns > 1.
  for (unsigned i = 0, e = options.numWarmup; i != e; ++i) {
    // Fill the arguments with an an immortal reference to the tensor data
    // so the input buffers live across all executions.
    SmallVector<AnyAsyncValueRef> args;
    args.reserve(model->getNumInputs());
    for (auto &input : inputTensors) {
      args.push_back(
          AsyncValueRef<Tensor>::createReady(runtime, input.getImmortal()));
    }

    // Execute the model and insert its results into the results vec.
    auto results = executableModel->execute(args);

    // Wait for the model to finish execution, forcing synchrony.
    await(results);
  }

  outs() << "--- Running '" << model->getName() << "':\n";
  outs().flush();

  SmallVector<AnyAsyncValueRef> results;

  SmallVector<double> cpuTimesMillis;
  cpuTimesMillis.reserve(options.numRuns);

  // Measure the time taken for each inference and print it.
  for (unsigned i = 0, e = options.numRuns; i != e; ++i) {

    // Fill the arguments.  We pass in an immortal
    // reference to the tensor data so the input buffers live across all
    // executions.
    SmallVector<AnyAsyncValueRef> args;
    args.reserve(model->getNumInputs());
    for (auto &input : inputTensors)
      args.push_back(
          AsyncValueRef<Tensor>::createReady(runtime, input.getImmortal()));

    auto start = std::chrono::steady_clock::now();

    // Execute the model and insert its results into the results vec.
    results = executableModel->execute(args);

    // Wait for the model to finish execution, forcing synchrony.
    await(results);

    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> time = end - start;

    // Store the time it took to perform the inference.
    cpuTimesMillis.push_back(time.count());
  }
  outs().flush();
  outs() << "--- Runtime(ms): ";
  TensorSpec cpuTimesMillisShape(TensorShape({options.numRuns.getValue()}),
                                 DType::f64);

  RETURN_ERROR(Kernel::printTensor(cpuTimesMillis.data(), cpuTimesMillisShape,
                                   Kernel::ResultOutputStyle::kFull));
  outs().flush();

  // There is nothing to do if there is no results.
  if (results.empty())
    return success();

  // Print the result tensors with the default printer.
  MutableArrayRef resultRef(results.data(), results.size());
  if (failed(printTensorResults(executableModel.get(), resultRef.drop_front(),
                                ctx, options.resultOutputStyle, outs())))
    return Error("repeated model execution failed");
  return success();
}

/// Run `model` using the command line options flags configured by the user.
LogicalResult M::runModel(MLIRContext *ctx, const Model *model,
                          LLCL::Runtime &runtime,
                          const ExecutorDriverCLOptions &options) {
  if (options.benchmarkMode) {
    // FIXME: benchmarkModel does not handle errors.
    benchmarkModel(model, runtime, options.benchmarkName, options.inputGenKind,
                   options.benchmarkScenario, options.benchmarkOutputDir,
                   options.benchmarkNoLogPrefix);
    return success();
  }

  auto invokeFn =
      options.numRuns == 1 ? executeModelOnce : executeModelRepeated;
  if (auto err = invokeFn(ctx, model, runtime, options))
    return failure(options.reportError(err.getError()));
  return success();
}
