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
using llvm::outs;
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

/// Wait for results and report their errors
static void waitForResults(const ExecutableModel *model,
                           MutableArrayRef<AnyAsyncValueRef> results,
                           MLIRContext *context) {
  // Check that any tensors we get are generic tensors, not framework
  // specific tensors.  We know that all tensors have been awaited here.
  for (auto &result : results) {
    assert(result.isReady() && "We awaited this, it should be ready");
    // If we have an error or Tensor result, then we're good.
    if (result.isError() || result.isType<Tensor>())
      continue;

    // Otherwise, we must have a framework specific tensor, convert it to a
    // generic one (or get an error).
    result = model->convertToGenericTensor(std::move(result));
    await(result);
  }
}

/// Helper that prints diagnostics if an AnyAsyncValueRef is an error.
static LogicalResult printDiagIfError(const AnyAsyncValueRef &res,
                                      raw_ostream &os) {
  if (auto *diag = res.getDiagnosticIfPresent()) {
    os << "<error: " << diag->getMessage() << '>';
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
                   MLIRContext *context, Kernels::ResultOutputStyle outputStyle,
                   raw_ostream &os) {
  // We expect the first result to be a chaing
  const AnyAsyncValueRef &ch = results.front();
  if (failed(printDiagIfError(ch, os)))
    return failure();
  assert(isa<Chain>(ch) && "first argument is not a chain");
  results = results.drop_front();

  StringRef name = model->getModel()->getName();

  waitForResults(model, results, context);

  // Print out the invokable results that we know about.
  // Go ahead and print out the invokable results that we know about.
  bool hadError = false;
  if (!results.empty()) {
    os << "'" << name << "' returned ";
    llvm::interleaveComma(results, os, [&](const AnyAsyncValueRef &result) {
      // Handle error value by printing it inline, and emitting it as an error
      // through SourceMgr so we can validate it.
      if (failed(printDiagIfError(result, os))) {
        hadError = true;
        return;
      }

      // Handle tensor types.
      if (auto *tensor = dyn_cast<Tensor>(result)) {
        os.flush();
        if (auto err = Kernels::printTensor(tensor->getBuffer(),
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

/// Copy raw data from results to the pre-allocated memory found in outputs.
static LogicalResult populateTensorResults(
    const ExecutableModel *model, MutableArrayRef<AnyAsyncValueRef> results,
    MLIRContext *context, llvm::SmallVector<void *> &outputs) {
  assert(outputs.size() == results.size() &&
         "Wrong number of outputs in the model.");

  waitForResults(model, results, context);

  size_t outIdx = 0;

  for (const AnyAsyncValueRef &result : results) {
    if (auto *diag = result.getDiagnosticIfPresent()) {
      auto loc = diag->decodeLocation();
      auto mlirLoc = mlir::FileLineColLoc::get(context, loc.filename, loc.line,
                                               loc.column);
      return emitError(mlirLoc) << diag->getMessage();
    }
    // Handle tensor types.
    auto *tensor = dyn_cast<Tensor>(result);
    if (!tensor) {
      auto loc = model->getModel()->getLocation().decode();
      auto mlirLoc = mlir::FileLineColLoc::get(context, loc.filename, loc.line,
                                               loc.column);
      return emitError(mlirLoc) << "populateTensorResults: not a Tensor";
    }
    DEBUG_WITH_TYPE("modular_memcpy",
                    llvm::dbgs() << "memcpy " << tensor->getSizeInBytes()
                                 << " bytes from M::Tensor to output buffer\n");
    /// TODO(#5867): Support an API which can avoid copy-on-output.
    memcpy(outputs[outIdx], tensor->getBuffer(), tensor->getSizeInBytes());
    outIdx++;
  }
  return success();
}

/// Repeatedly evaluate the model (numRuns to be exact) and capture and print
/// the runtime for each evaluation.
static ErrorOrSuccess executeModel(MLIRContext *ctx, const Model *model,
                                   Runtime &runtime,
                                   const ExecutorDriverCLOptions &options,
                                   unsigned int numRuns,
                                   unsigned int numWarmup) {
  TIME_PROFILER_SCOPE(Trace::kOther, 1, "executeModel");

  UNWRAP_ERROR(executableModel, model->setup(runtime));
  auto errorOrSuccess = executableModel->waitForModelSetup();
  if (errorOrSuccess.isError())
    return errorOrSuccess.takeError();

  // Note: this may cause the warmup run to consume inputs. If it becomes
  // surprising to users, we won't be able to share buffers between warmup and
  // input.
  UNWRAP_ERROR(inputTensorGenerators,
               createTensorGenerators(
                   options.inputGenKind, options.inputShapes, options.inputData,
                   options.minimumCpuAlignment, model, runtime));

  // Run the model numWarmup times to warmup the execution. This is performed to
  // remove any added overhead due to initial setup of the model.
  for (unsigned i = 0; i < numWarmup; ++i) {
    // Fill the arguments with borrowed references to the tensor data
    // since the input buffers live across all executions.
    SmallVector<AnyAsyncValueRef> args;
    args.reserve(model->getNumInputs());
    for (const auto &inputGen : inputTensorGenerators) {
      args.push_back(AsyncValueRef<Tensor>::createReady(
          runtime, inputGen->getNextTensor().borrow(GML::BufferRef::kExecute)));
    }

    // Execute the model and insert its results into the results vec.
    auto results = executableModel->execute(args);

    // Wait for the model to finish execution, forcing synchrony.
    await(results);
  }

  // We use printf here because it has atomic output behavior w.r.t. the printfs
  // in the model.
  printf("--- Running '%s':\n", model->getName().str().c_str());
  fflush(stdout);

  SmallVector<AnyAsyncValueRef> results;

  for (unsigned i = 0; i < numRuns; ++i) {
    TIME_PROFILER_SCOPE(Trace::kOther, 1, "runModelOnce");
    // Fill the arguments with borrowed references to the tensor data
    // since the input buffers live across all executions.
    SmallVector<AnyAsyncValueRef> args;
    args.reserve(model->getNumInputs());
    for (auto &inputGen : inputTensorGenerators)
      args.push_back(AsyncValueRef<Tensor>::createReady(
          runtime, inputGen->getNextTensor().borrow(GML::BufferRef::kExecute)));

    // Execute the model and insert its results into the results vec.
    results = executableModel->execute(args);

    // Wait for the model to finish execution, forcing synchrony.
    await(results);
    outs().flush();

    // There is nothing to do if there is no results.
    if (results.empty())
      continue;

    // Print the result tensors with the default printer.
    MutableArrayRef<AnyAsyncValueRef> resultRef(results.data(), results.size());
    if (failed(printTensorResults(executableModel.get(), resultRef, ctx,
                                  options.resultOutputStyle, outs())))
      return Error("repeated model execution failed");
  }
  return success();
}

/// Run `model` using the command line options flags configured by the user.
LogicalResult M::runModel(MLIRContext *ctx, const Model *model,
                          LLCL::Runtime &runtime,
                          const ExecutorDriverCLOptions &options) {
  if (options.benchmarkMode) {
    if (options.numRuns.getValue() > 2) {
      // If it's not the default 1, the user most likely set it; warn them about
      // that.
      llvm::errs() << "WARNING: num-runs is ignored in benchmarking mode. "
                      "Please use benchmark-min-query-count instead.\n";
    }
    // Execute the model for the warmup runs only.
    if (auto err = executeModel(ctx, model, runtime, options, 0,
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
    unsigned int numRuns = options.numRuns.getValue();
    unsigned int numWarmup = options.numWarmup.getValue();
    if (numRuns < 2) {
      // Well this sucks. It looks like things barf all over the place when
      // enabling multi runs.
      // So for now, keep the old behavior of resetting num-warmup to 0 for
      // num-runs=1.
      // TODO(#11042): Remove this ugliness.
      if (options.numRuns.getValue() > 2) {
        // If it's not the default 1, the user most likely set it; warn them
        // about that.
        llvm::errs() << "WARNING: Due to #11042, num-warmup is reset to zero "
                        "when num-runs is 1.\n";
      }
      numWarmup = 0;
    }
    if (auto err =
            executeModel(ctx, model, runtime, options, numRuns, numWarmup))
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
  UNWRAP_ERROR(executableModel, model->setup(runtime));
  auto errorOrSuccess = executableModel->waitForModelSetup();
  if (errorOrSuccess.isError())
    return errorOrSuccess.takeError();
  SmallVector<Tensor, 3> inputTensors;
  for (const auto &[input, spec] : llvm::zip(inputs, model->getInputSpecs())) {
    if (llvm::any_of(spec.getDims(), mlir::ShapedType::isDynamic))
      return Error("Dynamic input shapes are not supported.");
    // Nothing is known about the alignment of the inputs.
    // However they will be held alive for the duration of the execute call.
    // TODO(#5867): Support API which allows ownership of inputs to be
    // transferred to execute call.
    inputTensors.emplace_back(Tensor::createBorrowed(
        input, spec, /*alignment=*/{}, GML::BufferRef::kExecute));
  }

  // Fill the arguments.  We are running the model once so we pass in the input
  // tensors directly (the model will deallocate them when it is done with
  // them).
  SmallVector<AnyAsyncValueRef> args;
  args.reserve(model->getNumInputs());
  for (auto &input : inputTensors)
    args.push_back(
        AsyncValueRef<Tensor>::createReady(runtime, std::move(input)));

  // We use printf here because it has atomic output behavior w.r.t. the printfs
  // in the model.
  printf("--- Running and collecting outputs for '%s':\n",
         model->getName().str().c_str());
  fflush(stdout);

  // Execute the model and insert its results into the results vec.
  SmallVector<AnyAsyncValueRef> results = executableModel->execute(args);

  // There is nothing to do if there is no results.
  if (results.empty())
    return success();

  // Wait for the model to finish execution, forcing synchrony.
  await(results);

  MutableArrayRef<AnyAsyncValueRef> resultRef(results);
  // Drop the first chain result and collect the output data.
  if (failed(populateTensorResults(executableModel.get(),
                                   resultRef.drop_front(), ctx, outputs)))
    return Error("model execution failed");
  return success();
}
