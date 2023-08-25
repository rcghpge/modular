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
#include "Support/ML/SizeUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <filesystem>
#include <fstream>

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
  ErrorOr<TensorShape> errorOrTensorShape =
      TensorShape::parseFromString(argValue);
  if (failed(errorOrTensorShape))
    return options.error(errorOrTensorShape.getError());
  result = *errorOrTensorShape;
  return false;
}

bool TensorSpecParser::parse(llvm::cl::Option &options, StringRef argName,
                             StringRef argValue, TensorSpec &result) {
  ErrorOr<TensorSpec> errorOrTensorSpec = TensorSpec::parseFromString(argValue);
  if (failed(errorOrTensorSpec))
    return options.error(errorOrTensorSpec.getError());
  result = *errorOrTensorSpec;
  return false;
}

// Parse --input-data arg.
// Format of arg is
// --input-data=tensor_idx::input_source::input_source_attr_list.
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
  argValue.split(args, "::");
  if (args.size() < 2)
    return Error("--input-data requires `::` separated list of args.");
  if (args.size() > 3)
    return Error("--input-data supports max of 3 args separated by `::`.");

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

/// TensorResultInfo struct and corresponding MappingTraits to help write output
/// information to YAML when --result-output-style=binary.
namespace {
struct TensorResultInfo {
  std::string name;
  std::string shape;
};
} // namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(TensorResultInfo)

template <>
struct llvm::yaml::MappingTraits<TensorResultInfo> {
  static void mapping(IO &io, TensorResultInfo &info) {
    io.mapRequired("name", info.name);
    io.mapRequired("shape", info.shape);
  }
};

/// Helper to wite YAML with tensor result info.
static LogicalResult
writeYamlToFile(const std::filesystem::path &yamlFilePath,
                llvm::SmallVector<TensorResultInfo> &tensorResultsInfo) {
  std::error_code yamlFileErr;
  llvm::raw_fd_ostream yamlFile(llvm::StringRef(yamlFilePath.string()),
                                yamlFileErr);
  if (yamlFileErr) {
    llvm::errs() << "Failed to open output tensor yaml: "
                 << yamlFileErr.message() << "\n";
    return failure();
  }

  llvm::yaml::Output yout(yamlFile);
  yout << tensorResultsInfo;

  return success();
}

/// This method can be used to print the results of a model in the standard
/// format for an executor tool. It handles converting framework specific tensor
/// types into something we can print if needed, and handles printing/reporting
/// of errors in a way that works with diagnostic verification. It expects the
/// first argument to be a chain, which is never printed (but it's handled if
/// it's an error).
///
/// outputDir is only used if --result-output-style=binary, in which case this
/// function will instead create a directory and write one binary file for each
/// tensor result along with a YAML file containing the name and shape of each
/// tensor result. The output directory containing these files must be specified
/// with --binary-output-dir=<output_dir>."
static LogicalResult
outputTensorResults(const ExecutableModel *model,
                    MutableArrayRef<AnyAsyncValueRef> results,
                    Kernels::ResultOutputStyle outputStyle, unsigned precision,
                    StringRef outputDir) {
  // We expect the first result to be a chain.
  assert(!results.empty() && "must have at least one result");
  const AnyAsyncValueRef &ch = results.front();
  if (failed(printDiagIfError(ch)))
    return mlir::failure();
  assert(ch.isValueAvailable() && isa<Chain>(ch) &&
         "first result must be a chain");
  results = results.drop_front();

  if (results.empty())
    return success();

  convertResultsToGenericTensors(model, results);

  // If --result-output-style=binary, write output tensors to files and return.
  if (outputStyle == Kernels::ResultOutputStyle::kBinary) {

    // Holds the TensorResultInfo objects for each output tensor.
    SmallVector<TensorResultInfo> tensorResultsInfo;

    // Create the output directory. If the output directory is not there, then
    // the directory is created along with all of its parents. If the directory
    // does exist, then the following is a no-op.
    if (auto ec =
            llvm::sys::fs::create_directories(outputDir,
                                              /* ignoreExisting */ true)) {
      llvm::errs() << "Failed to create output tensor directories: "
                   << ec.message() << "\n";
      return failure();
    }

    llvm::outs() << "--- Writing Outputs to " << outputDir << "\n";

    bool hasError = false;
    for (size_t i = 0; i < results.size(); ++i) {
      std::string outputName = model->getModel()->getOutputNames()[i];
      const auto &result = results[i];
      assert(result.isReady() && "model result must be ready");
      if (failed(printDiagIfError(result)))
        hasError = true;
      else {
        if (auto *tensor = dyn_cast<Tensor>(result)) {
          tensorResultsInfo.push_back(
              {outputName, tensor->getSpec().getAsString()});

          std::ofstream outputFile(
              (std::filesystem::path(outputDir.str()) / outputName),
              std::ofstream::binary);

          outputFile.write((const char *)tensor->getBuffer(),
                           tensor->getSpec().getSizeInBytes());

          if (!outputFile) {
            llvm::errs() << "Failed to write output tensor file.\n";
            return failure();
          }
        } else {
          tensorResultsInfo.push_back({"unknown result", ""});
        }
      }
    }
    // Write tensor result information to yaml.
    // This can be used by the model tool to map outputs for verification.
    if (writeYamlToFile(
            (std::filesystem::path(outputDir.str()) / "output.yaml"),
            tensorResultsInfo)
            .failed()) {
      llvm::errs() << "Failed to write YAML with result info."
                   << "\n";
    }
    return failure(hasError);
  }

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
        if (auto err =
                Kernels::printTensor(tensor->getBuffer(), tensor->getSpec(),
                                     outputStyle, precision)) {
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
    LLCL::profiledMemcpy(outputs[outIdx], tensor->getBuffer(),
                         tensor->getSizeInBytes());
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

/// Call executableModel->execute(args), await the results, and return them.
/// If checkResults is true, return any error which any result contains.
/// Otherwise, assume the consumer of the results will check for errors.
static ErrorOr<SmallVector<AnyAsyncValueRef>>
executeModelSynchronously(Runtime &runtime, ExecutableModel &executableModel,
                          MutableArrayRef<AnyAsyncValueRef> args,
                          bool checkResults) {
#if MODULAR_PARANOID
  // Create an abstract 'resource' to represent this execution.
  auto resource = LLCL::Resource::allocate(
      Twine("execute(")
          .concat(executableModel.getModel()->getName())
          .concat(")")
          .str());
  // Create a default 'use' of this 'resource', which will be propogated
  // to all work items the call to execute launches.
  runtime.getWorkQueue()->pushDefaultUse(
      resource->beginUse(Twine("task(")
                             .concat(executableModel.getModel()->getName())
                             .concat(")")
                             .str()));
#endif
  auto results = executableModel.execute(args);
#if MODULAR_PARANOID
  // NOTE: Sleeping here would give the model a chance to attempt to
  // emplace it's final output results before we have called await.
  // This is rare, and can trigger bugs since under the common case
  // an emplace on the final output result won't trigger the await
  // until the emplacing task has returned all the way out to the LLCL
  // run items loop.
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif
  if (checkResults) {
    auto unitOr = awaitAndCheck(results);
    if (unitOr.isError())
      return unitOr.takeError();
    // Ignore value.
  } else {
    await(results);
  }
#if MODULAR_PARANOID
  runtime.getWorkQueue()->popDefaultUse();
  // Indicate that the call to execute has completed, and that no tasks launched
  // by it should still be in-flight.
  resource->markFreed();
  // Simulate the caller concurrently accessing all the results. This will
  // assert fail if any of them overlap, which could happen if there's a
  // compiler bug in memory planning or concat-in-place.
  SmallVector<LLCL::ResourceUse> uses;
  for (size_t i = 1, n = results.size(); i < n; ++i) {
    const AnyAsyncValueRef &result = results[i];
    if (auto *tensor = dyn_cast<Tensor>(result)) {
      tensor->collectAccessUses(uses,
                                (Twine("output(") + Twine(i - 1) + ")").str(),
                                LLCL::kReadingResourceUse);
    }
  }
#endif
  return results;
}

/// Repeatedly evaluate the model (numRuns to be exact) and capture and print
/// the runtime for each evaluation.
static ErrorOrSuccess executeModel(const Model *model, Runtime &runtime,
                                   const ExecutorDriverCLOptions &options,
                                   ArrayRef<QuantTensorSpec> inputSpecs,
                                   unsigned int numRuns,
                                   unsigned int numWarmup) {
  std::unique_ptr<llvm::raw_fd_ostream> optTimingOs;
  if (!options.saveTimings.empty()) {
    std::error_code errorCode;
    optTimingOs =
        std::make_unique<llvm::raw_fd_ostream>(options.saveTimings, errorCode);
    if (errorCode)
      return Error(
          Twine("failed to open timings file: " + errorCode.message()));
  }

  TIME_PROFILER_SCOPE(Trace::kOther, 1, "executeModel");

  // Setup once.
  std::unique_ptr<ExecutableModel> executableModel;
  {
    TIME_PROFILER_SCOPE(Trace::kOther, 1, "executeModel.setup");
    auto executableModel1Or = model->setup(runtime);
    if (executableModel1Or.isError())
      return executableModel1Or.takeError();
    auto executableModel1 = std::move(*executableModel1Or);
    auto errorOrSuccess = executableModel1->waitForModelSetup();
    if (errorOrSuccess.isError())
      return errorOrSuccess.takeError();
    executableModel = std::move(executableModel1);
  }

  // Establish a tensor generator for each model input as per command line
  // options and shapes. Each tensor generator holds one or more M::Tensor's
  // with lifetime matching the generator itself.
  auto inputTensorGeneratorsOr = createTensorGenerators(
      options.inputGenKind, inputSpecs, options.inputData,
      options.minimumCpuAlignment, runtime);
  if (inputTensorGeneratorsOr.isError())
    return inputTensorGeneratorsOr.takeError();
  auto inputTensorGenerators = std::move(*inputTensorGeneratorsOr);

  // Run the model numWarmup times to warmup the execution. This is performed to
  // remove any added overhead due to initial setup of the model.
  if (numWarmup > 0) {
    llvm::outs() << "--- Warming up '" << model->getName().str() << "' "
                 << numWarmup << " times\n";
    llvm::outs().flush();
  }
  for (unsigned i = 0; i < numWarmup; ++i) {
    TIME_PROFILER_SCOPE(Trace::kOther, 1, "executeModel.warmup");
    auto args = buildArgs(runtime, inputTensorGenerators);

    // Execute the model synchronously. Propagate any error back to caller.
    auto resultsOr = executeModelSynchronously(runtime, *executableModel, args,
                                               /*checkResults=*/true);
    if (resultsOr.isError())
      return resultsOr.takeError();
    // Ignore results value.
  }

  if (numRuns > 0) {
    llvm::outs() << "--- Running '" << model->getName().str() << "'";
    if (numRuns > 1)
      llvm::outs() << " " << numRuns << " times:\n";
    else
      llvm::outs() << ":\n";
    llvm::outs().flush();
  }

  for (unsigned i = 0; i < numRuns; ++i) {
    auto args = buildArgs(runtime, inputTensorGenerators);

    if (i > 0 && i % 100 == 0)
      llvm::outs() << "--- Run " << i << "...\n";

    // Execute the model synchronously.
    std::chrono::high_resolution_clock::time_point startTime, endTime;
    auto entry = ProfilerEntry<Trace::EnableTrace(Trace::kOther, 1)>::create(
        "executeModel.run");
    startTime = std::chrono::high_resolution_clock::now();
    // If not printing any results we need to still check they are error free.
    // Otherwise, we need only wait for the results and errors will be
    // reported when printing the results below.
    auto resultsOr =
        executeModelSynchronously(runtime, *executableModel, args,
                                  /*checkResults=*/options.resultOutputStyle ==
                                      Kernels::ResultOutputStyle::kNone);
    if (resultsOr.isError())
      return resultsOr.takeError();
    auto results = std::move(*resultsOr);
    endTime = std::chrono::high_resolution_clock::now();
    std::move(entry).record();

    if (options.resultOutputStyle != Kernels::ResultOutputStyle::kNone) {
      // Convert and print the result tensors. This is the one place where
      // AsyncValue errors are printed to stdout and a generic
      // "repeated model execution failed" is returned to the caller instead
      // of just propagating the error. This helps unit tests distinguish
      // expected model error output from transient errors running the tool
      // itself.
      if (failed(outputTensorResults(
              executableModel.get(), results, options.resultOutputStyle,
              options.printPrecision, options.binaryOutputDir)))
        return Error("repeated model execution failed");
    }

    // If requested, print the model execution time in nanoseconds with no
    // adornment for the convenience of scripts. Note that failed executions
    // will have no time reported.
    if (optTimingOs) {
      *optTimingOs << (endTime - startTime).count() << "\n";
      // Let's just flush on every output so you can interrupt the process
      // and still have usable data.
      optTimingOs->flush();
    }

    llvm::outs().flush();
  }

  return success();
}

/// Given an array of compile-time QuantTensorSpecs and an array of run-time
/// override shapes, return an updated array of QuantTensorSpecs.
static ErrorOr<SmallVector<QuantTensorSpec>>
getResolvedInputShapes(ArrayRef<QuantTensorSpec> compiledShapes,
                       ArrayRef<TensorShape> userInputShapes) {
  SmallVector<QuantTensorSpec> ret;
  auto numInputTensors = compiledShapes.size();

  // If no user shapes, just copy input to output
  if (userInputShapes.empty()) {
    ret.append(compiledShapes.begin(), compiledShapes.end());
    return ret;
  }

  // Next, check that array sizes match.
  if (userInputShapes.size() != numInputTensors) {
    return Error(Twine("Number of specified input shapes (") +
                 Twine(userInputShapes.size()) + ") doesn't match " +
                 "number of model inputs (" + Twine(numInputTensors) + ")");
  }

  // Iterate through each input tensor
  for (unsigned i = 0; i < numInputTensors; ++i) {
    auto userShape = userInputShapes[i];
    auto compiledShape = compiledShapes[i];

    // Verify that ranks match.
    if (userShape.getRank() != compiledShape.getRank()) {
      return Error(Twine("Specified shape ") + userShape.getAsString() +
                   " doesn't match the rank of model input shape " +
                   compiledShape.getAsString());
    }

    // Validate given dimensions.
    for (const auto &[userDim, modelDim] :
         llvm::zip(userShape, compiledShape)) {
      if (mlir::ShapedType::isDynamic(modelDim)) {
        if (mlir::ShapedType::isDynamic(userDim)) {
          return Error(
              Twine("Must specify a fixed input shape for model input shape ") +
              compiledShape.getAsString());
        }
      } else if (userDim != modelDim) {
        return Error(Twine("Specified dimension ") + Twine(userDim) +
                     " doesn't match model input dimension " + Twine(modelDim) +
                     " for shape " + compiledShape.getAsString());
      }
    }

    // Build new TensorSpec
    ret.push_back(compiledShape.cloneWithNewShape(userShape));
  }

  return ret;
}

/// Pass resolved input shapes to profiler, if its active.
static void saveInputSpecs(ArrayRef<QuantTensorSpec> specs,
                           std::optional<TimeTraceProfiler> &profiler) {
  if (profiler)
    for (auto &spec : specs)
      profiler->addInputShape(spec.getAsString());
}

/// Run `model` using the command line options flags configured by the user.
LogicalResult M::runModel(MLIRContext *ctx, const Model *model,
                          LLCL::Runtime &runtime,
                          const ExecutorDriverCLOptions &options) {

  // Get fully resolved input shapes
  auto shapesOrErr =
      getResolvedInputShapes(model->getInputSpecs(), options.inputShapes);
  if (shapesOrErr.isError()) {
    return failure(options.reportError(shapesOrErr.getError()));
  }
  const auto &inputSpecs = shapesOrErr.takeValue();
  saveInputSpecs(inputSpecs, runtime.getProfiler());

  if (options.benchmarkMode) {
    int numWarmup = options.numWarmup.getValue();
    if (options.numRuns.getValue() > 2) {
      // If it's not the default 1, the user most likely set it; warn them
      // about that.
      llvm::errs() << "WARNING: num-runs is ignored in benchmarking mode. "
                      "Please use benchmark-min-query-count instead.\n";
    }
    if (numWarmup == 0) {
      // Force at least one warmup.
      llvm::errs() << "WARNING: forcing one warmup run in benchmarking mode. "
                      "You can set an larger value using --num-warmup.\n";
      numWarmup = 1;
    }
    // Execute the model for the warmup runs only.
    if (auto err = executeModel(model, runtime, options, inputSpecs,
                                /*numRuns=*/0, numWarmup))
      return failure(options.reportError(err.getError()));
    // Run the actual benchmark.
    if (auto err = benchmarkModel(
            model, runtime, options.benchmarkName, options, inputSpecs,
            options.benchmarkScenario, options.benchmarkOutputDir,
            options.benchmarkNoLogPrefix, options.benchmarkMinDurationMs,
            options.benchmarkMinQueryCount))
      return failure(options.reportError(err.getError()));

  } else {
    if (auto err = executeModel(model, runtime, options, inputSpecs,
                                options.numRuns.getValue(),
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
  auto executableModelOr = model->setup(runtime);
  if (executableModelOr.isError())
    return executableModelOr.takeError();
  auto executableModel = std::move(*executableModelOr);
  auto unitOr = executableModel->waitForModelSetup();
  if (unitOr.isError())
    return unitOr.takeError();

  // The runModel() entrypoint supports runtime override for input shapes,
  // but that's not available here. Verify that the user isn't trying to do so.
  if (options.inputShapes.size()) {
    return Error("Runtime input shapes are not supported.");
  }

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
  auto resultsOr = executeModelSynchronously(runtime, *executableModel, args,
                                             /*checkResults=*/true);
  if (resultsOr.isError())
    return resultsOr.takeError();
  auto results = std::move(*resultsOr);

  // Copy results into pre-allocated outputs, which we trust are of the
  // correct size. Propagate any error back to caller.
  auto unit2Or =
      populateTensorResults(executableModel.get(), results, ctx, outputs);
  if (unit2Or.isError())
    return unit2Or.takeError();

  return success();
}
