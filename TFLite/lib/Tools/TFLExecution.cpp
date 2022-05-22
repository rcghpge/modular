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
#include "Support/Error.h"
#include "TFLite/Primitives.h"
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

/// Run the setup for every model. Puts all the chain outputs into `chains`
/// and all the context outputs into `contexts`. The models are sorted by
/// name, and so the chains and contexts will be too.
Error setupAllModels(const TFLModelFile &models, Runtime &runtime,
                     SmallVectorImpl<AnyAsyncValueRef> &chains,
                     SmallVectorImpl<AnyAsyncValueRef> &contexts) {
  for (const auto &m : models) {
    auto resultsOr = m.setup(runtime);
    if (failed(resultsOr))
      return resultsOr.takeError();

    chains.push_back(std::move(resultsOr->first));
    contexts.push_back(std::move(resultsOr->second));
  }

  return {""};
}

static size_t getNumTensorArgs(const TFLRuntimeModel &model) {
  return model.getExecuteFn()->getNumArgs() - kExecuteFnNumMinInputs;
}

Error M::initializeInputMemoryForAllModels(const TFLModelFile &models,
                                           InputGenerationKind inputGenKind,
                                           SmallVectorImpl<void *> &ptrVec) {
  for (const auto &model : models) {
    auto dataOr = initializeInputMemory(inputGenKind, model);
    if (failed(dataOr))
      return dataOr.takeError();

    ptrVec.append(*dataOr);
  }

  return {""};
}

static Error checkDataPointers(ArrayRef<TFLRuntimeModel> models,
                               ArrayRef<void *> dataPtrs) {
  size_t totalNumDataPtrsExpected = std::accumulate(
      models.begin(), models.end(), 0,
      [](size_t i, const auto &m) { return i + getNumTensorArgs(m); });

  if (dataPtrs.empty() && totalNumDataPtrsExpected != 0)
    return {Twine("no data pointers were supplied, but expected " +
                  Twine(totalNumDataPtrsExpected))};

  if (dataPtrs.size() < totalNumDataPtrsExpected)
    return {Twine(dataPtrs.size()) +
            Twine(" data pointers were supplied, but expected ") +
            Twine(totalNumDataPtrsExpected)};

  return {""};
}

/// Execute every model, calling `beforeExec` before, and then `afterExec`
/// afterwards. `beforeExec` can abort execution of that model by returning
/// `failure()`. Does not save the outputs of any model. `afterExec` is
/// responsible for awaiting the AsyncValues passed in. Updates `chains` with
/// the chain returned from the last execution and does not pass them to
/// `afterExec`.
static Error executeAllModels(
    const TFLModelFile &models, Runtime &runtime,
    MutableArrayRef<AnyAsyncValueRef> chains,
    ArrayRef<AnyAsyncValueRef> contexts, ArrayRef<void *> dataPtrs,
    std::function<LogicalResult(const TFLRuntimeModel &)> beforeExec,
    std::function<void(const TFLRuntimeModel &model,
                       ArrayRef<AnyAsyncValueRef>)>
        afterExec) {
  auto dataPtrStart = dataPtrs.begin();

  Error err = checkDataPointers(models, dataPtrs);
  if (failed(err))
    return err;

  // Fill the arguments
  SmallVector<AnyAsyncValueRef> args;
  for (auto [model, chain, ctx] : llvm::zip(models, chains, contexts)) {
    args.clear();
    args.reserve(model.getExecuteFn()->getNumArgs());
    args.push_back(chain.copy());
    args.push_back(ctx.copy());

    for (size_t i = 0, e = getNumTensorArgs(model); i < e; ++i) {
      TFLiteType eltType(model.getInputElementType(i));
      args.push_back(AsyncValueRef<TfLiteTensorRef>::createReady(
          runtime, TfLiteTensorRef::getWithExternalData(
                       model.getInputShape(i), eltType, *dataPtrStart++)));
    }

    if (beforeExec)
      if (failed(beforeExec(model)))
        continue; // Go to the next model.

    // Execute the model and insert its results into the results vec.
    auto results = model.execute(args, runtime);

    // Pass in the tensor results to `afterExec`.
    if (afterExec)
      afterExec(model, results);

    // The chain is the first result.
    chain = std::move(results.front());
  }

  return {""};
}

LogicalResult M::runModelFile(const TFLModelFile &modelFile, Runtime &runtime,
                              InputGenerationKind inputGenKind,
                              unsigned numRuns, bool benchmarkMode) {

  SmallVector<AnyAsyncValueRef, 4> chains, contexts;
  Error err = setupAllModels(modelFile, runtime, chains, contexts);
  if (failed(err)) {
    outs() << err;
    return failure();
  }

  SmallVector<void *, 0> inputData;
  auto fail = [&](Error e) {
    for (auto d : inputData)
      free(d);
    outs() << e;
    return failure();
  };

  err = initializeInputMemoryForAllModels(modelFile, inputGenKind, inputData);
  if (failed(err))
    return fail(std::move(err));

  auto beforeRun = [&](const TFLRuntimeModel &model) {
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
  auto afterRun = [&](const TFLRuntimeModel &model,
                      ArrayRef<AnyAsyncValueRef> results) {
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
    err = executeAllModels(modelFile, runtime, chains, contexts, inputData,
                           beforeRun, afterRun);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> time = end - start;
    cpuTimesMillis.push_back(time.count());
    if (failed(err))
      return fail(std::move(err));
  }

  // Run the number of inferences asked for.
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

ErrorOr<RCRef<BEFFile>> M::openTFLBEF(StringRef file) {
  // Set up all our primitives.
  PrimitiveRegistry primitives;
  registerGraphRTControlFlowPrimitives(primitives);
  registerTFLitePrimitives(primitives);
  primitives.finalize();

  return BEFFile::open(file, primitives);
}
