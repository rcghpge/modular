//===- TFLExecution.cpp ---------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "TFLite/Tools/TFLExecution.h"
#include "GraphRT/BEFExecutor/BEFFile.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "GraphRT/Primitives.h"
#include "LLCL/Runtime/Algorithms.h"
#include "TFLite/Primitives.h"
#include "TFLite/TFLModelFile.h"
#include "TFLite/TFLTensor.h"
#include "TFLite/Tools/TFLInputGeneration.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

#include <chrono>

using namespace M;
using namespace llvm;
using namespace mlir;

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
      auto mlirLoc =
          FileLineColLoc::get(context, loc.filename, loc.line, loc.column);
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

// Everything we need to run a single inference.
SmallVector<AnyAsyncValueRef, 1>
M::runOneInference(const TFLRuntimeModel &model,
                   SmallVectorImpl<AnyAsyncValueRef> &args, Runtime &runtime,
                   MLIRContext *context, SmallVector<void *, 3> inputTensorData,
                   bool print) {
  // Don't malloc, just use the data pointer we already have.
  for (const auto &[shape, elType, dataPtr] :
       llvm::zip(model.getInputShapes(), model.getInputElementTypes(),
                 inputTensorData)) {
    args.push_back(AsyncValueRef<TfLiteTensorRef>::createReady(
        runtime, TfLiteTensorRef::getWithExternalData(shape, elType, dataPtr)));
  }

  auto results = model.execute(args, runtime);

  // Wait for the model to finish execution.
  await(results);

  // Print the result tensors with the default printer, but not the chain.
  if (print)
    printResults(model.getName(), makeArrayRef(results).drop_front(), context,
                 outs());
  return results;
}

LogicalResult M::runModelFile(const TFLModelFile &modelFile, Runtime &runtime,
                              const InputGenerationKind &inputGenKind,
                              unsigned int numRuns, bool benchmarkMode) {
  SmallVector<AnyAsyncValueRef, 0> chains, contexts;
  M::Error err = modelFile.setupAllModels(runtime, chains, contexts);
  if (failed(err)) {
    outs() << err;
    return failure();
  }

  SmallVector<void *, 0> inputData;
  auto fail = [&](M::Error e) {
    for (auto d : inputData)
      free(d);
    outs() << e;
    return failure();
  };

  err = modelFile.initializeInputMemoryForAllModels(inputGenKind, inputData);
  if (failed(err))
    return fail(std::move(err));

  auto beforeRun = [&](const TFLRuntimeModel &model) {
    auto numInputArgs = model.getExecuteFn()->getNumArgs();
    auto inputShapes = model.getInputShapes();

    if (inputGenKind == InputGenerationKind::kNone &&
        numInputArgs != kExecuteFnNumMinInputs) {
      errs() << "Cannot execute a model with input tensors (sizes: ";
      llvm::interleaveComma(inputShapes, errs());
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
  auto afterRun = [&](const TFLRuntimeModel &model,
                      ArrayRef<AnyAsyncValueRef> results) {
    // Wait for the model to finish execution.
    await(results);

    // Print the result tensors with the default printer.
    if (!benchmarkMode)
      printResults(model.getName(), results, &ctx, outs());
  };

  // Run the number of inferences asked for.
  // TODO: Generalize `benchmarkMode` into a more flexible policy around how
  // we control execution of the model.
  if (!benchmarkMode) {
    for (unsigned i = 0, e = numRuns; i != e; ++i) {
      err = modelFile.executeAllModels(runtime, chains, contexts, inputData,
                                       beforeRun, afterRun);
      if (failed(err))
        return fail(std::move(err));
    }
  } else {
    // Measure the time taken for each inference.
    std::vector<double> cpuTimesMillis;
    for (unsigned i = 0, e = numRuns; i != e; ++i) {
      auto start = std::chrono::steady_clock::now();
      err = modelFile.executeAllModels(runtime, chains, contexts, inputData,
                                       beforeRun, afterRun);
      auto end = std::chrono::steady_clock::now();
      std::chrono::duration<double, std::milli> time = end - start;
      cpuTimesMillis.push_back(time.count());
      if (failed(err))
        return fail(std::move(err));
    }

    outs() << "Model inference times: ";
    llvm::interleaveComma(cpuTimesMillis, outs());
    outs() << '\n';
    outs().flush();
  }

  for (auto d : inputData)
    free(d);

  return success();
}

ErrorOr<RCRef<BEFFile>> M::openTFLBEF(StringRef file) {
  // Set up all our primitives.
  PrimitiveRegistry primitives;
  registerGraphRTControlFlowPrimitives(primitives);
  registerTFLitePrimitives(primitives);
  primitives.finalize();

  return BEFFile::open(file, primitives);
}
