//===- TFLExecution.cpp ---------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "TFLite/Tools/TFLExecution.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/AllocatorType.h"
#include "Support/Error.h"
#include "TFLite/Support/TFLiteType.h"
#include "TFLite/TFLModelFile.h"
#include "TFLite/TFLTensor.h"
#include "TFLite/Tools/TFLInputGeneration.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

#include <chrono>
#include <numeric>

using namespace M;
using llvm::errs;
using llvm::outs;
using mlir::MLIRContext;

static constexpr size_t kExecuteFnNumMinInputs = 2; // chain and context

static bool printAsyncValueResult(const LLCL::AnyAsyncValueRef &value,
                                  llvm::raw_ostream &os) {
  // Handle TFLite tensor types.
  if (auto *tensor = value->dyn_cast<TfLiteTensor>()) {
    os << *tensor;
    return true;
  }

  // Handle TFLite Model descriptors.
  if (auto *modelDesc = value->dyn_cast<TFLRuntimeModel>()) {
    os << "a TFLite model with these members: "
       << modelDesc->getSetupFn()->getName() << ", "
       << modelDesc->getExecuteFn()->getName() << ".\n";
    return true;
  }
  return false;
}

/// This method can be used to print the results of `invokable` in the
/// standard format.  It delegates to printAsyncValueResult and handles
/// standard data types.
static void printResults(StringRef invokableName,
                         ArrayRef<AnyAsyncValueRef> results,
                         MLIRContext *context, raw_ostream &os) {
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
    os << "'" << invokableName << "' returned ";
    llvm::interleaveComma(results, os, [&](const AnyAsyncValueRef &result) {
      // Handle error value by printing it inline, and emitting it as an error
      // through SourceMgr so we can validate it.
      if (auto *diag = result->getDiagnosticIfPresent()) {
        os << "<error: " << diag->getMessage() << '>';
        return;
      }

      // Allow the client bef-executor program to add custom printing hooks.
      if (printAsyncValueResult(result, os))
        return;

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

ErrorOrSuccess
M::initializeInputMemoryForModel(const TFLRuntimeModel &model,
                                 InputGenerationKind inputGenKind,
                                 SmallVectorImpl<void *> &ptrVec) {
  auto dataOr = initializeInputMemory(inputGenKind, model);
  if (failed(dataOr))
    return dataOr.takeError();

  ptrVec.append(*dataOr);

  return success();
}

ErrorOrSuccess
M::initializeInputMemoryForAllModels(const TFLModelFile &models,
                                     InputGenerationKind inputGenKind,
                                     SmallVectorImpl<void *> &ptrVec) {
  for (const auto &model : models) {
    auto dataOr = initializeInputMemory(inputGenKind, model);
    if (failed(dataOr))
      return dataOr.takeError();

    ptrVec.append(*dataOr);
  }

  return success();
}

/// Execute given model, calling `beforeExec` before, and then `afterExec`
/// afterwards. `beforeExec` can abort execution of that model by returning
/// `failure()`. Saves the outputs of `model` into `results`. `afterExec` is
/// responsible for awaiting the AsyncValues passed in.
static ErrorOrSuccess
executeModel(const TFLExecutableModel &executableModel,
             ArrayRef<void *> dataPtrs,
             std::function<LogicalResult()> beforeExec,
             std::function<void(ArrayRef<AnyAsyncValueRef>)> afterExec) {

  const TFLRuntimeModel &model = executableModel.getModel();
  Runtime &runtime = executableModel.getRuntime();

  auto dataPtrStart = dataPtrs.begin();

  size_t totalNumDataPtrsExpected = model.getNumModelInputs();
  if (dataPtrs.size() != totalNumDataPtrsExpected)
    return Error("incorrect number of data pointers supplied, but expected ");

  // Fill the arguments
  SmallVector<AnyAsyncValueRef> args;
  args.reserve(totalNumDataPtrsExpected);
  for (auto [inputType, inputShape] :
       llvm::zip(model.getInputElementTypes(), model.getInputShapes())) {
    TFLiteType eltType(inputType);
    args.push_back(AsyncValueRef<TfLiteTensorRef>::createReady(
        runtime, TfLiteTensorRef::getWithExternalData(inputShape, eltType,
                                                      *dataPtrStart++)));
  }

  if (beforeExec)
    if (failed(beforeExec()))
      return {Twine("Unable to run model: '") + model.getName() + Twine("'.")};

  // Execute the model and insert its results into the results vec.
  auto results = executableModel.execute(args);

  // Pass in the tensor results to `afterExec`.
  if (afterExec)
    afterExec(results);

  return success();
}

LogicalResult M::runModel(const TFLRuntimeModel &model, Runtime &runtime,
                          InputGenerationKind inputGenKind, size_t numRuns,
                          bool benchmarkMode) {
  auto errorOrExecutableModel = model.setup(runtime);
  if (failed(errorOrExecutableModel)) {
    outs() << errorOrExecutableModel.takeError();
    return failure();
  }

  auto executableModel = errorOrExecutableModel.takeValue();

  SmallVector<void *, 0> inputData;
  ErrorOrSuccess error;

  /// This cleans up after a failure and prints the error message in the error
  /// variable.
  auto handleFailure = [&]() {
    for (auto d : inputData)
      free(d);
    outs() << error.takeError() << "\n";
    return failure();
  };

  if ((error = initializeInputMemoryForModel(model, inputGenKind, inputData)))
    return handleFailure();

  auto beforeRun = [&]() -> LogicalResult {
    auto numInputArgs = model.getExecuteFn()->getNumArgs();

    if (inputGenKind == InputGenerationKind::kNone &&
        numInputArgs != kExecuteFnNumMinInputs) {
      errs() << "Cannot execute a model with input tensors (sizes: ";
      llvm::interleaveComma(model.getInputShapes(), errs());
      errs() << ") without input initialization yet! Hint: try setting "
                "`-generate-inputs`.\n";
      errs().flush();
      return failure();
    }

    outs() << "--- Running '" << model.getName() << "':\n";
    outs().flush();
    return success();
  };

  MLIRContext ctx;
  auto afterRun = [&](ArrayRef<AnyAsyncValueRef> results) {
    // Wait for the model to finish execution.
    await(results);

    // Print the result tensors with the default printer.
    if (!benchmarkMode)
      printResults(model.getName(), makeArrayRef(results).drop_front(), &ctx,
                   outs());
  };

  // Measure the time taken for each inference.
  std::vector<double> cpuTimesMillis;
  for (unsigned i = 0, e = numRuns; i != e; ++i) {
    auto start = std::chrono::steady_clock::now();
    if (failed(executeModel(executableModel, inputData, beforeRun, afterRun)))
      return handleFailure();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> time = end - start;
    cpuTimesMillis.push_back(time.count());
  }

  // TODO: Generalize `benchmarkMode` into a more flexible policy around how
  // we control execution of the model.
  if (benchmarkMode) {
    outs() << "Model inference times: ";
    llvm::interleaveComma(cpuTimesMillis, outs());
    outs() << '\n';
    outs().flush();
  }

  for (auto d : inputData)
    free(d);

  return success();
}
