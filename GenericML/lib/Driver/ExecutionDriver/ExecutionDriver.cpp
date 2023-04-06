//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/ExecutionDriver.h"
#include "GenericML/Driver/DriverCLOptions.h"
#include "GenericML/Driver/DriverInternals.h"
#include "GenericML/Support/Model.h"
#include "GenericML/Support/Tensor.h"
#include "Kernels/Misc.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Support/Profiling.h"
#include "Support/Error.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>

using namespace M;
using mlir::MLIRContext;

StringLiteral M::stringifyEnum(InputGenerationKind kind) {
  switch (kind) {
  case InputGenerationKind::kRandom:
    return "random";
  case InputGenerationKind::kZeros:
    return "zeros";
  case InputGenerationKind::kOnes:
    return "ones";
  case InputGenerationKind::kNegOnes:
    return "negones";
  }
  llvm_unreachable("unhandled InputGenerationKind");
}

std::optional<InputGenerationKind>
M::symbolizeInputGenerationKind(StringRef s) {
  return llvm::StringSwitch<std::optional<InputGenerationKind>>(s)
      .Case(stringifyEnum(InputGenerationKind::kRandom),
            InputGenerationKind::kRandom)
      .Case(stringifyEnum(InputGenerationKind::kZeros),
            InputGenerationKind::kZeros)
      .Case(stringifyEnum(InputGenerationKind::kOnes),
            InputGenerationKind::kOnes)
      .Case(stringifyEnum(InputGenerationKind::kNegOnes),
            InputGenerationKind::kNegOnes)
      .Default(std::nullopt);
}

std::optional<TensorGeneratorFlag>
M::symbolizeTensorGeneratorFlag(StringRef s) {
  return llvm::StringSwitch<std::optional<TensorGeneratorFlag>>(s)
      .Case("Cycle", TensorGeneratorFlag::kCycle)
      .Case("Truncate", TensorGeneratorFlag::kTruncate)
      .Default(std::nullopt);
}

bool TensorShapeParser::parse(llvm::cl::Option &options, StringRef argName,
                              StringRef argValue, TensorShape &result) {
  SmallVector<int64_t, 5> shape;
  if (!argValue.empty()) {
    // Empty means scalar.
    for (StringRef dimStr : llvm::split(argValue, 'x')) {
      int64_t dim;
      if (!llvm::to_integer(dimStr, dim))
        return options.error("invalid shape dimension '" + dimStr + "'");
      shape.emplace_back(dim);
    }
  }
  result = TensorShape(shape.begin(), shape.end());
  return false;
}

// Parse --input-data arg.
// Format of arg is --input-data=tensor_idx:input_source:input_source_attr_list.
//    - tensor_idx: int representing input tensor index
//    - input_source: can be either:
//        - input generation function: one of [zeros|random|ones|negones], which
//        map to an InputGenerationKind
//        - filename
//    - input_source_attr_list (optional): csv list of [Cycle|Truncate]
//    which map to a TensorGeneratorFlag
ErrorOrSuccess InputDataParser::parseImpl(llvm::cl::Option &options,
                                          StringRef argName, StringRef argValue,
                                          InputDataSpec &result) {
  llvm::SmallVector<llvm::StringRef, 3> args;
  argValue.split(args, ':');
  if (args.size() < 2)
    return Error("--input-data requires `:` separated list of args.");
  if (args.size() > 3)
    return Error("--input-data supports max of 3 args separated by `:`.");

  // parse tensor idx field
  auto tensorIdxStr = args[0];
  auto err = tensorIdxStr.getAsInteger(10, result.tensorIdx);
  if (err)
    return Error(Twine("Could not convert tensorIdx (") + tensorIdxStr +
                 ") to integer.");

  // parse input gen field
  auto filenameOrGenKindStr = args[1];
  std::optional<InputGenerationKind> genKind =
      symbolizeInputGenerationKind(filenameOrGenKindStr.str());
  if (genKind.has_value()) {
    result.inputGen = genKind.value();
    if (args.size() > 2)
      return Error(Twine("No additional --input-data flags (") + args[2] +
                   ") are supported when input_source is an input "
                   "generation function.");
    return success();
  }
  // if string does not match an InputGenerationKind, assume it's a filename
  auto filename = std::filesystem::path(filenameOrGenKindStr.str());
  if (!std::filesystem::exists(filename)) {
    return Error(Twine("--input-data file (") + filename.string() +
                 ") does not exist.");
  }
  result.inputGen = filename;

  // parse gen flags if they exist
  if (args.size() < 3)
    return success();
  for (const auto genFlagStr : llvm::split(args[2], ",")) {
    auto flag = symbolizeTensorGeneratorFlag(genFlagStr);
    if (!flag)
      return Error(Twine("Cannot convert ") + genFlagStr +
                   "to TensorGeneratorFlag.");
    result.genFlags = result.genFlags | flag.value();
  }

  return success();
}

/// Replace any non M::Tensor tensors in results with their M::Tensor
/// equivalents.
static void
convertResultsToGenericTensors(const ExecutableModel *model,
                               MutableArrayRef<AnyAsyncValueRef> results) {
  for (auto &result : results) {
    assert(result.isReady() && "tensor result should be ready");
    if (result.isError() || result.isType<Tensor>())
      continue;
    result = model->convertToGenericTensor(std::move(result));
    await(result);
  }
}

/// Wait for results to be ready and return success if none are marked with an
/// error. Otherwise return first error encountered.
static ErrorOrSuccess awaitAndCheck(MutableArrayRef<AnyAsyncValueRef> results) {
  await(results);
  for (const auto &result : results) {
    if (auto *diag = result.getDiagnosticIfPresent())
      // TODO: Capture location
      return std::move(diag->getMessage());
  }
  return success();
}

/// Helper that prints diagnostics if an AnyAsyncValueRef is an error.
static LogicalResult printDiagIfError(const AnyAsyncValueRef &res) {
  if (auto *diag = res.getDiagnosticIfPresent()) {
    llvm::outs() << "<error: " << diag->getMessage() << '>';
    return failure();
  }
  return success();
}

/// This method can be used to print the results of a model in the standard
/// format for an executor tool. It handles converting framework specific tensor
/// types into something we can print if needed, and handles printing/reporting
/// of errors in a way that works with diagnostic verification. It expects the
/// first argument to be a chain, which is never printed (but it's handled if
/// it's an error).
static LogicalResult
printTensorResults(const ExecutableModel *model,
                   MutableArrayRef<AnyAsyncValueRef> results,
                   Kernels::ResultOutputStyle outputStyle) {
  // We expect the first result to be a chain.
  const AnyAsyncValueRef &ch = results.front();
  assert(ch.isReady() && isa<Chain>(ch) &&
         "first result must be a ready chain");
  if (failed(printDiagIfError(ch)))
    return mlir::failure();
  results = results.drop_front();

  if (results.empty())
    return success();

  convertResultsToGenericTensors(model, results);

  StringRef name = model->getModel()->getName();
  llvm::outs() << "'" << name << "' returned ";

  bool first = true;
  bool hasError = false;
  for (const auto &result : results) {
    assert(result.isReady() && "model result must be ready");
    if (!first)
      llvm::outs() << ", ";
    first = false;
    if (failed(printDiagIfError(result)))
      hasError = true;
    else {
      if (auto *tensor = dyn_cast<Tensor>(result)) {
        if (auto err = Kernels::printTensor(tensor->getBuffer(),
                                            tensor->getSpec(), outputStyle)) {
          llvm::outs() << "<error: " << err.getError() << '>';
          hasError = true;
        }
      } else {
        llvm::outs() << "<unknown type>";
      }
    }
  }
  llvm::outs() << '\n';
  return failure(hasError);
}

/// Copy raw data from results to the pre-allocated memory found in outputs.
static ErrorOrSuccess populateTensorResults(
    const ExecutableModel *model, MutableArrayRef<AnyAsyncValueRef> results,
    MLIRContext *context, llvm::SmallVector<void *> &outputs) {
  // We expect the first result to be a chain
  const AnyAsyncValueRef &ch = results.front();
  assert(ch.isReady() && !ch.isError() && isa<Chain>(ch) &&
         "first result must be a ready non-error chain");
  results = results.drop_front();

  assert(outputs.size() == results.size() &&
         "Wrong number of outputs in the model.");

  if (results.empty())
    return success();

  convertResultsToGenericTensors(model, results);

  size_t outIdx = 0;

  for (const AnyAsyncValueRef &result : results) {
    assert(result.isReady() && !result.isError() &&
           "model result must be ready and non-error");
    auto *tensor = dyn_cast<Tensor>(result);
    if (!tensor)
      return Error("model result is not a Tensor");
    DEBUG_WITH_TYPE("modular_memcpy",
                    llvm::dbgs() << "memcpy " << tensor->getSizeInBytes()
                                 << " bytes from M::Tensor to output buffer\n");
    /// TODO(#5867): Support an API which can avoid copy-on-output.
    memcpy(outputs[outIdx], tensor->getBuffer(), tensor->getSizeInBytes());
    outIdx++;
  }
  return success();
}

// Returns a ready AsyncValueRef<Tensor> for every given generator, ready
// to be passed to the model's execute function. Each tensor will borrow,
// at 'execute' scope, from the underlying generotar's 'next' tensor.
static SmallVector<AnyAsyncValueRef, 3>
buildArgs(Runtime &runtime, ArrayRef<ITensorGeneratorPtr> generators) {
  SmallVector<AnyAsyncValueRef, 3> args;
  args.reserve(generators.size());
  for (const auto &generator : generators) {
    args.emplace_back(AsyncValueRef<Tensor>::createReady(
        runtime, generator->getNextTensor().borrow(GML::BufferRef::kExecute)));
  }
  return args;
}

/// Repeatedly evaluate the model (numRuns to be exact) and capture and print
/// the runtime for each evaluation.
static ErrorOrSuccess executeModel(const Model *model, Runtime &runtime,
                                   const ExecutorDriverCLOptions &options,
                                   unsigned int numRuns,
                                   unsigned int numWarmup) {
  TIME_PROFILER_SCOPE(Trace::kOther, 1, "executeModel");

  // Setup once.
  UNWRAP_ERROR(executableModel, model->setup(runtime));
  auto errorOrSuccess = executableModel->waitForModelSetup();
  if (errorOrSuccess.isError())
    return errorOrSuccess.takeError();

  // Establish a tensor generator for each model input as per command line
  // options and shapes. Each tensor generator holds one or more M::Tensor's
  // with lifetime matching the generator itself.
  UNWRAP_ERROR(inputTensorGenerators,
               createTensorGenerators(
                   options.inputGenKind, options.inputShapes, options.inputData,
                   options.minimumCpuAlignment, model, runtime));

  // Run the model numWarmup times to warmup the execution. This is performed to
  // remove any added overhead due to initial setup of the model.
  if (numWarmup > 0) {
    llvm::outs() << "--- Warming up '" << model->getName().str() << "' "
                 << numWarmup << " times\n";
    llvm::outs().flush();
  }
  for (unsigned i = 0; i < numWarmup; ++i) {
    auto args = buildArgs(runtime, inputTensorGenerators);

    // Execute the model synchronously. Propagate any error back to caller.
    auto results = executableModel->execute(args);
    UNWRAP_ERROR(unit, awaitAndCheck(results));
    (void)unit;
  }

  llvm::outs() << "--- Running '" << model->getName().str() << "'";
  if (numRuns > 1)
    llvm::outs() << " " << numRuns << " times:\n";
  else
    llvm::outs() << ":\n";
  llvm::outs().flush();

  for (unsigned i = 0; i < numRuns; ++i) {
    TIME_PROFILER_SCOPE(Trace::kOther, 1, "runModelOnce");
    auto args = buildArgs(runtime, inputTensorGenerators);

    // Execute the model synchronously.
    auto startTime = std::chrono::high_resolution_clock::now();
    auto results = executableModel->execute(args);
    if (options.resultOutputStyle == Kernels::ResultOutputStyle::kNone) {
      // Though we don't want to print the tensor results, we still want to
      // propagate any error back to the caller.
      UNWRAP_ERROR(unit, awaitAndCheck(results));
      (void)unit;
    } else {
      // Simply wait for the result chain and all tensor results. We'll
      // deal with errors when printing below.
      await(results);
    }
    auto endTime = std::chrono::high_resolution_clock::now();

    if (options.resultOutputStyle != Kernels::ResultOutputStyle::kNone) {
      // Convert and print the result tensors. This is the one place where
      // AsyncValue errors are printed to stdout and a generic
      // "repeated model execution failed" is returned to the caller instead
      // of just propagating the error. This helps unit tests distinguish
      // expected model error output from transient errors running the tool
      // itself.
      if (failed(printTensorResults(executableModel.get(), results,
                                    options.resultOutputStyle)))
        return Error("repeated model execution failed");
    }

    // If requested, print the model execution time in nanoseconds with no
    // adornment for the convenience of scripts. Note that failed executions
    // will have no time reported.
    if (options.printTiming)
      llvm::outs() << std::chrono::nanoseconds(endTime - startTime).count()
                   << "\n";

    llvm::outs().flush();
  }
  return success();
}

/// Run `model` using the command line options flags configured by the user.
LogicalResult M::runModel(MLIRContext *ctx, const Model *model,
                          LLCL::Runtime &runtime,
                          const ExecutorDriverCLOptions &options) {
  if (options.benchmarkMode) {
    if (options.numRuns.getValue() > 2) {
      // If it's not the default 1, the user most likely set it; warn them
      // about that.
      llvm::errs() << "WARNING: num-runs is ignored in benchmarking mode. "
                      "Please use benchmark-min-query-count instead.\n";
    }
    // Execute the model for the warmup runs only.
    if (auto err = executeModel(model, runtime, options, /*numRuns=*/0,
                                options.numWarmup.getValue()))
      return failure(options.reportError(err.getError()));
    // Run the actual benchmark.
    if (auto err = benchmarkModel(
            model, runtime, options.benchmarkName, options,
            options.benchmarkScenario, options.benchmarkOutputDir,
            options.benchmarkNoLogPrefix, options.benchmarkMinDurationMs,
            options.benchmarkMinQueryCount))
      return failure(options.reportError(err.getError()));

  } else {
    if (auto err =
            executeModel(model, runtime, options, options.numRuns.getValue(),
                         options.numWarmup.getValue()))
      return failure(options.reportError(err.getError()));
  }
  return success();
}

/// Run `model` using the command line options flags configured by the user.
/// `inputs` is the input tensor data to the model, while `outputs` is
/// pre-allocated memory to be filled with the model's results.
ErrorOrSuccess
M::runModelWithData(MLIRContext *ctx, const Model *model,
                    LLCL::Runtime &runtime,
                    const ExecutorDriverCLOptions &options,
                    const llvm::SmallVector<const void *> &inputs,
                    llvm::SmallVector<void *> &outputs) {
  assert(inputs.size() == model->getInputSpecs().size() &&
         "Wrong number of inputs to the model.");

  // Setup the model.
  UNWRAP_ERROR(executableModel, model->setup(runtime));
  UNWRAP_ERROR(unit, executableModel->waitForModelSetup());
  (void)unit;

  // Construct input tensors which borrow from the given input buffers. We
  // assume they are held alive for the entire execution.
  SmallVector<AnyAsyncValueRef, 3> args;
  for (const auto &[input, spec] : llvm::zip(inputs, model->getInputSpecs())) {
    if (llvm::any_of(spec, mlir::ShapedType::isDynamic))
      return Error("Dynamic input shapes are not supported.");
    // Nothing is known about the alignment of the inputs.
    // However they will be held alive for the duration of the execute call.
    // TODO(#5867): Support API which allows ownership of inputs to be
    // transferred to execute call.
    args.emplace_back(AsyncValueRef<Tensor>::createReady(
        runtime, Tensor::createBorrowed(input, spec, /*alignment=*/{},
                                        GML::BufferRef::kExecute)));
  }

  llvm::outs() << "--- Running and collecting outputs for '"
               << model->getName().str() << "':\n";
  llvm::outs().flush();

  // Execute the model synchronously.
  auto results = executableModel->execute(args);
  UNWRAP_ERROR(unit2, awaitAndCheck(results));
  (void)unit2;

  // Copy results into pre-allocated outputs, which we trust are of the
  // correct size. Propagate any error back to caller.
  UNWRAP_ERROR(unit3, populateTensorResults(executableModel.get(), results, ctx,
                                            outputs));
  (void)unit3;

  return success();
}
