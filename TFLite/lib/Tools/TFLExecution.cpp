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
#include "TFLite/TFLModel/TFLModel.h"
#include "TFLite/TFLTensor.h"
#include "TFLite/Tools/TFLInputGeneration.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include <chrono>

using namespace M;
using llvm::errs;
using llvm::outs;
using mlir::MLIRContext;

/// This method can be used to print the results of `invokable` in the
/// standard format.  It delegates to printAsyncValueResult and handles
/// standard data types.
static void printTensorResults(StringRef invokableName,
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

      // Handle TFLite tensor types.
      if (auto *tensor = result->dyn_cast<TfLiteTensor>()) {
        os << *tensor;
        return;
      }

      os << "<unknown type>";
    });
    os << '\n';
    os.flush();
  }
}

void M::runModel(const TFLRuntimeModel &model, Runtime &runtime,
                 InputGenerationKind inputGenKind, size_t numRuns,
                 bool benchmarkMode) {
  auto errorOrExecutableModel = model.setup(runtime);
  if (failed(errorOrExecutableModel)) {
    outs() << errorOrExecutableModel.takeError();
    return;
  }

  auto executableModel = errorOrExecutableModel.takeValue();

  auto inputTensorsOr = initializeInputMemory(inputGenKind, model, runtime);
  if (failed(inputTensorsOr)) {
    outs() << inputTensorsOr.takeError() << "\n";
    return;
  }

  SmallVector<Tensor, 3> &inputTensors = inputTensorsOr.get();
  MLIRContext ctx;

  // Measure the time taken for each inference.
  std::vector<double> cpuTimesMillis;
  for (unsigned i = 0, e = numRuns; i != e; ++i) {
    auto start = std::chrono::steady_clock::now();

    // Fill the arguments
    SmallVector<AnyAsyncValueRef> args;
    args.reserve(model.getNumModelInputs());
    for (auto &inputTensor : inputTensors) {
      TFLiteType eltType(inputTensor.getEltType());
      assert(eltType != TFLiteType::invalid &&
             "invalid input format for model?");
      args.push_back(AsyncValueRef<TfLiteTensorRef>::createReady(
          runtime,
          TfLiteTensorRef::getWithExternalData(inputTensor.getShape(), eltType,
                                               inputTensor.getPointer())));
    }

    outs() << "--- Running '" << model.getName() << "':\n";
    outs().flush();

    // Execute the model and insert its results into the results vec.
    auto results = executableModel.execute(args);

    // Wait for the model to finish execution, forcing synchrony.
    await(results);

    // Print the result tensors with the default printer.
    if (!benchmarkMode)
      printTensorResults(model.getName(), makeArrayRef(results).drop_front(),
                         &ctx, outs());

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
}
