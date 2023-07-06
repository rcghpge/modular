//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/ExecutionDriver.h"
#include "GenericML/Driver/Utils.h"
#include "GenericML/Support/Model.h"
#include "GenericML/Support/ModelSpec.h"
#include "GenericML/Support/Tensor.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Support/FileLocationDecoder.h"
#include "LLCL/Support/Profiling.h"
#include "Support/ML/DType.h"

#include "modular/c/common.h"
#include "modular/c/tensorflow/config.h"
#include "modular/cpp/aliases.h"

#include "GenericML/FormatYAML/FormatYAML.h"

using namespace M;

/// Translates a modular C data type to GenericML DType.
static DType convertToDType(M_Dtype ty) {
  switch (ty) {
  case M_UNKNOWN:
    return DType::invalid;
  case M_INT8:
    return DType::si8;
  case M_INT16:
    return DType::si16;
  case M_INT32:
    return DType::si32;
  case M_INT64:
    return DType::si64;
  case M_UINT8:
    return DType::ui8;
  case M_UINT16:
    return DType::ui16;
  case M_UINT32:
    return DType::ui32;
  case M_UINT64:
    return DType::ui64;
  case M_FLOAT16:
    return DType::f16;
  case M_FLOAT32:
    return DType::f32;
  case M_DOUBLE:
    return DType::f64;
  case M_BOOL:
    return DType::kBool;
  }
}
/// Translates a GeneralML DType to C data type.
static M_Dtype convertToCDataType(DType ty) {
  switch (ty.getValue()) {
  case DType::f32:
    return M_FLOAT32;
  case DType::ui8:
    return M_UINT8;
  case DType::si8:
    return M_INT8;
  case DType::ui16:
    return M_UINT16;
  case DType::si16:
    return M_INT16;
  case DType::ui32:
    return M_UINT32;
  case DType::si32:
    return M_INT32;
  case DType::ui64:
    return M_UINT64;
  case DType::si64:
    return M_INT64;
  case DType::kBool:
    return M_BOOL;
  case DType::f64:
    return M_DOUBLE;
  case DType::invalid:
    [[fallthrough]];
  default:
    return M_UNKNOWN;
  }
}

static M::TensorSpec getTensorSpec(M_AsyncTensor *tensor) {
  M_Dtype dtype = M_getTensorType(tensor);
  TensorSpecPtr spec = {M_getTensorSpec(tensor), &M_freeTensorSpec};
  SmallVector<ssize_t> inputShape;
  size_t numDims = M_getRank(spec.get());
  inputShape.reserve(numDims);
  for (size_t i = 0; i < numDims; ++i)
    inputShape.push_back(M_getDimAt(spec.get(), i));
  return {inputShape, convertToDType(dtype)};
}

/// Get size of a tensor in bytes.
static size_t getSizeOf(M_AsyncTensor *tensor) {
  M_Dtype dtype = M_getTensorType(tensor);
  return M_getTensorNumElements(tensor) * M_sizeOf(dtype);
}

namespace {

/// Contains the core CLI options for modular-api-executor.
class ModularExecutorDriverCLOptions : public ExecutorDriverCLOptions {
public:
  using ExecutorDriverCLOptions::ExecutorDriverCLOptions;

  cl::opt<std::string> exportedName{
      "exported-name",
      cl::desc("Signature name of TF model in model with multiple signature."),
      cl::init("")};

  cl::opt<std::string> modelSpecificationFile{
      "model-specification", cl::desc("Path to the model specification file.")};

  cl::opt<bool> frameworkVersion{
      "framework-version",
      cl::desc("Prints the external framework's version.")};

  /// Reads a model specification from the YAML file set via the
  /// "model-specification" option.
  ModelSpec readModelSpecificationOrExit() {
    if (modelSpecificationFile.empty())
      exit(reportError("the model specification file path must be specified."));

    std::error_code errorCode;
    if (!std::filesystem::exists(
            std::filesystem::path{modelSpecificationFile.getValue()},
            errorCode)) {
      exit(reportError(Twine("the model specification file ") +
                       modelSpecificationFile + " does not exist."));
    }

    std::string errorMsg;
    auto fileOrErr = llvm::MemoryBuffer::getFile(modelSpecificationFile, true);
    if (std::error_code error = fileOrErr.getError()) {
      exit(reportError(Twine("cannot open input file ") +
                       modelSpecificationFile + ": " + error.message()));
    }

    auto modelSpec = loadModelSpecFromYAML(std::move(*fileOrErr));
    if (modelSpec.isError())
      exit(reportError(Twine(modelSpec.getError())));

    return modelSpec.takeValue();
  }
};

class ModularExecutableModel;

class ModularModel final : public Model {
public:
  ModularModel(std::string_view name, std::string_view pathOrId,
               std::string_view exportedName, ModelSpec modelSpec,
               int32_t numThreads)
      : Model(modelSpec),           // create a model from the ModelSpec
        spec(std::move(modelSpec)), // move the model spec
        name(name),                 // the name of the the benchmark
        pathOrId(pathOrId), exportedName(exportedName),
        runtimeContext(nullptr, &M_freeRuntimeContext),
        location(LLCL::FileNameLocationDecoder::getEncodedLocation(pathOrId)) {
    if (numThreads < 0) {
      llvm::errs() << "num-threads must be positive, got " << numThreads;
      exit(EXIT_FAILURE);
    }

    llvm::transform(spec.inputs, std::back_inserter(inputNames),
                    [](const auto &input) { return input.name.c_str(); });
    llvm::transform(spec.outputs, std::back_inserter(outputNames),
                    [](const auto &output) { return output.name.c_str(); });

    StatusPtr status{M_newStatus(), &M_freeStatus};
    RuntimeConfigPtr runtimeConfig{M_newRuntimeConfig(), &M_freeRuntimeConfig};
    M_setNumThreads(runtimeConfig.get(), numThreads);
    runtimeContext.reset(
        M_newRuntimeContext(runtimeConfig.get(), status.get()));

    if (M_isError(status.get())) {
      llvm::errs() << "Error: " << M_getError(status.get()) << "\n";
      exit(EXIT_FAILURE);
    }
  }
  // Gets the name of the model.
  StringRef getName() const override { return name; }

  const EncodedLocation &getLocation() const override { return location; }

  /// Gets the ModelSpec for this model.
  const ModelSpec &getModelSpec() const { return spec; }

  const M_RuntimeContext *getContext() const { return runtimeContext.get(); };

  ArrayRef<std::string> getInputNames() const { return inputNames; }
  ArrayRef<std::string> getOutputNames() const { return outputNames; }

  ErrorOr<std::unique_ptr<ExecutableModel>>
  setup(Runtime &runtime) const override;

  ~ModularModel() override = default;

private:
  ModelSpec spec;
  std::string name;
  std::string pathOrId;
  std::string exportedName;
  SmallVector<std::string> inputNames;
  SmallVector<std::string> outputNames;
  RuntimeContextPtr runtimeContext;
  EncodedLocation location;
};

/// Defines the Modular Executor. This class implements
/// GenericML::ExecutableModel.
class ModularExecutableModel final : public ExecutableModel {
public:
  ModularExecutableModel(const ModularModel *model, Runtime &runtime,
                         CompiledModelPtr compiledModel,
                         ModelPtr initalizedModel)
      : ExecutableModel(model, runtime),
        compiledModel(std::move(compiledModel)),
        initializedModel(std::move(initalizedModel)) {}

  ~ModularExecutableModel() override = default;

  [[nodiscard]] const ModularModel *getModel() const {
    return static_cast<const ModularModel *>(ExecutableModel::getModel());
  }

  // Executes a single instance of the inference process.
  [[nodiscard]] SmallVector<AnyAsyncValueRef>
  execute(MutableArrayRef<AnyAsyncValueRef> args) const override {

    auto model = getModel();
    const auto &modelSpec = model->getModelSpec();
    M_AsyncModel *modularModel = initializedModel.get();

    // Create the resulting async values. The output starts with chain async
    // (so we add 1).
    SmallVector<AnyAsyncValueRef> outputs;
    outputs.reserve(modelSpec.outputs.size() + 1);

    // printTensorResults within the ExecutionDriver.cpp expects the first
    // output to be a chain. We create a dummy chain here to make the output
    // conform to the expectation of printTensorResults in ExecutionDriver.cpp.
    // TODO: remove this once we make printTensorResults more general.
    outputs.emplace_back(AsyncValueRef<Chain>::allocate(getRuntime()));

    // Create the async placeholders for the outputs. The async values will be
    // materialized when the inference is complete.
    for (size_t i = 0, e = modelSpec.outputs.size(); i < e; ++i)
      outputs.emplace_back(AsyncValueRef<Tensor>::allocate(getRuntime()));

    // Hold on to the output async values so that they do not get reclaimed
    // before the inference is complete.
    SmallVector<AnyAsyncValueRef> outputCopy;

    // We keep copy of outputs (one for chain, and another for model output)
    // to hold on to them.
    outputCopy.reserve(outputs.size());
    for (auto &o : outputs)
      outputCopy.push_back(o.copy());

    // Get the runtime value so that we can construct the output tensors (and
    // not depend on the current class).
    Runtime &runtime = getRuntime();

    // When the args are ready, then we call the following lambda.
    andThenSyncMoving(
        args, [modularModel, model, outputs = std::move(outputCopy),
               &runtime](MutableArrayRef<AnyAsyncValueRef> args) mutable {
          const M_RuntimeContext *context = model->getContext();
          TensorMapPtr inputTensors{M_newAsyncTensorMap(context),
                                    &M_freeAsyncTensorMap};

          StatusPtr status{M_newStatus(), &M_freeStatus};
          for (auto &&[inputIndex, input] : llvm::enumerate(args)) {
            auto &inputTensor = input.get<Tensor>();

            SmallVector<int64_t> inputShape;
            for (const auto &dim : inputTensor.getShape())
              inputShape.emplace_back(dim);
            TensorSpecPtr spec{
                M_newTensorSpec(inputShape.data(), inputShape.size(),
                                convertToCDataType(inputTensor.getEltType()),
                                model->getInputNames()[inputIndex].c_str()),
                &M_freeTensorSpec};
            M_borrowTensorInto(inputTensors.get(), inputTensor.getBuffer(),
                               spec.get(), status.get());
            if (M_isError(status.get())) {
              llvm::errs() << "Error: " << M_getError(status.get()) << "\n";
              exit(EXIT_FAILURE);
            }
          }

          TensorMapPtr outputTensors = {nullptr, &M_freeAsyncTensorMap};

          M_AsyncTensorMap *outputTensorMap = M_executeModelSync(
              context, modularModel, inputTensors.get(), status.get());

          if (M_isError(status.get())) {
            llvm::errs() << "Error: " << M_getError(status.get()) << "\n";
            exit(EXIT_FAILURE);
          }

          outputTensors.reset(outputTensorMap);

          // Mark the input chain as ready.
          std::move(outputs[0]).emplace<Chain>();
          // Copy the output tensors to the output async values and then mark
          // the async outputs as ready.
          const auto &outputSpecs = model->getOutputSpecs();
          for (size_t i = 0, e = outputSpecs.size(); i < e; ++i) {
            // Get the output tensor. This tensor will be converted
            // to a GenericML tensor.
            TensorPtr outputTensor = {
                M_getTensorByNameFrom(outputTensors.get(),
                                      model->getOutputNames()[i].c_str(),
                                      status.get()),
                &M_freeTensor};

            if (M_isError(status.get())) {
              llvm::errs() << "Error: " << M_getError(status.get()) << "\n";
              exit(EXIT_FAILURE);
            }
            // Create an uninitialized GenericML Tensor to hold the output.
            auto tensor =
                Tensor::createOwned(getTensorSpec(outputTensor.get()), runtime,
                                    kPreferredMemoryAlignment);

            // Copy the output data to the output GenericML Tensor
            LLCL::profiledMemcpy(tensor.getMutableBuffer(),
                                 M_getTensorData(outputTensor.get()),
                                 getSizeOf(outputTensor.get()));

            // Move the output Tensor to the output async value and mark the
            // value as ready.
            std::move(outputs[i + 1]).emplace<Tensor>(std::move(tensor));
          }
        });
    // Return the output async values. Note that the output async values might
    // not be ready at this point. The outputs are marked as ready after the
    // interpreter is invoked in the above lambda.
    return outputs;
  }

  ErrorOrSuccess waitForModelSetup() override {
    StatusPtr status{M_newStatus(), &M_freeStatus};
    M_waitForModel(initializedModel.get(), status.get());
    if (M_isError(status.get()))
      return Error(M_getError(status.get()));
    return success();
  }

private:
  CompiledModelPtr compiledModel;
  ModelPtr initializedModel;
};

} // namespace

/// Initialize the model using `runtime`, allocating memory, setting up data
/// structures, or whatever else the model needs.  This never blocks.
ErrorOr<std::unique_ptr<ExecutableModel>>
ModularModel::setup(Runtime &runtime) const {
  // Create the executable model.
  CompileConfigPtr compileConfig = {M_newCompileConfig(), &M_freeCompileConfig};
  M_setModelPath(compileConfig.get(), pathOrId.data());
  if (!exportedName.empty())
    M_setExportedName(compileConfig.get(), exportedName.data());
  StatusPtr status{M_newStatus(), &M_freeStatus};
  CompiledModelPtr compiledModel{
      M_compileModel(runtimeContext.get(), compileConfig.get(), status.get()),
      &M_freeCompiledModel};
  ModelPtr initalizedModel = {
      M_initModel(runtimeContext.get(), compiledModel.get(), status.get()),
      &M_freeModel};
  return std::make_unique<ModularExecutableModel>(
      this, runtime, std::move(compiledModel), std::move(initalizedModel));
}

int main(int argc, char **argv) {
  ModularExecutorDriverCLOptions clOptions(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "modular-api-executor driver\n");

  if (clOptions.frameworkVersion) {
    llvm::outs() << "Framework version: " << M_version() << "\n";
    exit(EXIT_SUCCESS);
  }

  // We don't really have a file to load here now.
  std::unique_ptr<llvm::MemoryBuffer> inputFile;

  // Create the ModularModel. Note that the Model object will outlive
  // the execution.
  auto model = std::make_unique<ModularModel>(
      clOptions.benchmarkName, clOptions.inputFilename, clOptions.exportedName,
      clOptions.readModelSpecificationOrExit(), clOptions.getNumThreads());

  // The runtime should run as single threaded. We have already set the
  // thread count in framework runtime.
  clOptions.useSingleThreadedWorkqueue();

  // Run the model with leak detection.
  return setupMLIRAndRunWithLeakCheckedRuntime(
      clOptions, "modular-api-executor", std::move(inputFile),
      [&](Runtime &runtime, MLIRContext *ctx) {
        return runModel(ctx, model.get(), runtime, clOptions);
      });
}
