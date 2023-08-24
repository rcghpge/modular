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

#include "GenericML/FormatYAML/FormatYAML.h"

#include "ModularAPIInterface.h"
#include "mlir/IR/BuiltinTypes.h"

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
  case M_FLOAT64:
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
    return M_FLOAT64;
  case DType::invalid:
    [[fallthrough]];
  default:
    return M_UNKNOWN;
  }
}

// cannot import anything from C api detail files, so redefine constant here.
static constexpr int64_t apiDynamicDimension = -1;

/// Convert TensorSpec from API to GenericML TensorSpec.
static M::TensorSpec getTensorSpec(const ModularAPIInterface &api,
                                   M_TensorSpec *spec) {
  M_Dtype dtype = api.getDtype(spec);
  SmallVector<ssize_t> inputShape;
  size_t numDims = api.getRank(spec);
  inputShape.reserve(numDims);
  for (size_t i = 0; i < numDims; ++i) {
    if (api.getDimAt(spec, i) == apiDynamicDimension) {
      // Shapes in the M::Tensor layer need to follow MLIR convention where
      // dynamic dimensions are represented by constant kDynamic which is equal
      // to std::numeric_limits<int64_t>::min()
      inputShape.push_back(mlir::ShapedType::kDynamic);
    } else {
      inputShape.push_back(api.getDimAt(spec, i));
    }
  }
  return {inputShape, convertToDType(dtype)};
}

/// Get TensorSpec from a API Tensor.
static M::TensorSpec getTensorSpec(const ModularAPIInterface &api,
                                   M_AsyncTensor *tensor) {
  TensorSpecPtr spec = {api.getTensorSpec(tensor), api.freeTensorSpec};
  return getTensorSpec(api, spec.get());
}

/// Get size of a tensor in bytes.
static size_t getSizeOf(const ModularAPIInterface &api, M_AsyncTensor *tensor) {
  M_Dtype dtype = api.getTensorType(tensor);
  return api.getTensorNumElements(tensor) * api.sizeOf(dtype);
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

  cl::list<std::string> inputNames{
      "input-names", cl::desc("Specify input names. This is always required "
                              "and it must match the model's input names")};

  cl::list<std::string> outputNames{
      "output-names", cl::desc("Specify output names. This is always required "
                               "and it must match the model's output names")};

  cl::opt<bool> printMetadata{"print-metadata",
                              cl::desc("Display Model Input and output Specs"),
                              cl::init(false)};

  /// Reads a model specification from the YAML file set via the
  /// "model-specification" option.
  ErrorOr<ModelSpec> readModelSpecification() {
    if (modelSpecificationFile.empty())
      return Error("the model specification file path must be specified.");

    std::error_code errorCode;
    if (!std::filesystem::exists(
            std::filesystem::path{modelSpecificationFile.getValue()},
            errorCode)) {
      return Error(Twine("the model specification file ") +
                   modelSpecificationFile + " does not exist.");
    }

    std::string errorMsg;
    auto fileOrErr = llvm::MemoryBuffer::getFile(modelSpecificationFile, true);
    if (std::error_code error = fileOrErr.getError()) {
      return Error(Twine("cannot open input file ") + modelSpecificationFile +
                   ": " + error.message());
    }

    return loadModelSpecFromYAML(std::move(*fileOrErr));
  }
};

class ModularExecutableModel;

struct ModelState {
  M_RuntimeContext *context;
  M_AsyncCompiledModel *compiledModel;
};

class ModularModel final : public Model {
public:
  ModularModel(ModularAPIInterface &apiInterface, std::string_view name,
               ArrayRef<std::string> inputTensorNames,
               ArrayRef<std::string> outputTensorNames, ModelSpec modelSpec,
               ModelState state, EncodedLocation loc)
      : Model(modelSpec), // create a model from the ModelSpec
        api(apiInterface), spec(std::move(modelSpec)), // move the model spec
        name(name), // the name of the the benchmark
        inputNames(inputTensorNames), outputNames(outputTensorNames),
        modelState(state), location(std::move(loc)) {

    if (inputNames.empty()) {
      llvm::transform(spec.inputs, std::back_inserter(inputNames),
                      [](const auto &input) { return input.name; });
    }

    if (outputNames.empty()) {
      llvm::transform(spec.outputs, std::back_inserter(outputNames),
                      [](const auto &output) { return output.name; });
    }
  }

  const ModularAPIInterface &getApi() const { return api; }
  // Gets the name of the model.
  StringRef getName() const override { return name; }

  const EncodedLocation &getLocation() const override { return location; }

  /// Gets the ModelSpec for this model.
  const ModelSpec &getModelSpec() const { return spec; }

  const M_RuntimeContext *getContext() const { return modelState.context; };

  ArrayRef<std::string> getInputNames() const { return inputNames; }
  ArrayRef<std::string> getOutputNames() const { return outputNames; }

  ErrorOr<std::unique_ptr<ExecutableModel>>
  setup(Runtime &runtime) const override;

  ~ModularModel() override = default;

private:
  ModularAPIInterface &api;
  ModelSpec spec;
  std::string name;
  SmallVector<std::string> inputNames;
  SmallVector<std::string> outputNames;
  ModelState modelState;
  EncodedLocation location;
};

/// Defines the Modular Executor. This class implements
/// GenericML::ExecutableModel.
class ModularExecutableModel final : public ExecutableModel {
public:
  ModularExecutableModel(const ModularModel *model, Runtime &runtime,
                         ModelPtr initalizedModel)
      : ExecutableModel(model, runtime),
        initializedModel(std::move(initalizedModel)) {}

  ~ModularExecutableModel() override = default;

  [[nodiscard]] const ModularModel *getModel() const {
    return static_cast<const ModularModel *>(ExecutableModel::getModel());
  }

  // Executes a single instance of the inference process.
  [[nodiscard]] SmallVector<AnyAsyncValueRef>
  execute(MutableArrayRef<AnyAsyncValueRef> args) const override {
    // Get the runtime value so that we can construct the output tensors (and
    // not depend on the current class).
    Runtime &runtime = getRuntime();

    checkUniqueRuntime(runtime);
    checkKnownCallingThread(runtime);

    auto model = getModel();
    const auto &modelSpec = model->getModelSpec();
    const auto &api = model->getApi();
    M_AsyncModel *modularModel = initializedModel.get();

    // Create the resulting async values. The output starts with chain async
    // (so we add 1).
    SmallVector<AnyAsyncValueRef> outputs;
    outputs.reserve(modelSpec.outputs.size() + 1);

    // printTensorResults within the ExecutionDriver.cpp expects the first
    // output to be a chain. We create a dummy chain here to make the output
    // conform to the expectation of printTensorResults in ExecutionDriver.cpp.
    // TODO: remove this once we make printTensorResults more general.
    outputs.emplace_back(AsyncValueRef<Chain>::allocate(runtime));

    // Create the async placeholders for the outputs. The async values will be
    // materialized when the inference is complete.
    for (size_t i = 0, e = modelSpec.outputs.size(); i < e; ++i)
      outputs.emplace_back(AsyncValueRef<Tensor>::allocate(runtime));

    // Hold on to the output async values so that they do not get reclaimed
    // before the inference is complete.
    SmallVector<AnyAsyncValueRef> outputCopy;

    // We keep copy of outputs (one for chain, and another for model output)
    // to hold on to them.
    outputCopy.reserve(outputs.size());
    for (auto &o : outputs)
      outputCopy.push_back(o.copy());

    // When the args are ready, then we call the following lambda.
    andThenSyncMoving(
        args, [modularModel, model, outputs = std::move(outputCopy), &runtime,
               &api](MutableArrayRef<AnyAsyncValueRef> args) mutable {
          const M_RuntimeContext *context = model->getContext();
          TensorMapPtr inputTensors{api.newAsyncTensorMap(context),
                                    api.freeAsyncTensorMap};

          StatusPtr status{api.newStatus(), api.freeStatus};
          for (auto &&[inputIndex, input] : llvm::enumerate(args)) {
            auto &inputTensor = input.get<Tensor>();

            SmallVector<int64_t> inputShape;
            for (const auto &dim : inputTensor.getShape())
              inputShape.emplace_back(dim);
            TensorSpecPtr spec{
                api.newTensorSpec(inputShape.data(), inputShape.size(),
                                  convertToCDataType(inputTensor.getEltType()),
                                  model->getInputNames()[inputIndex].c_str()),
                api.freeTensorSpec};
            api.borrowTensorInto(inputTensors.get(), inputTensor.getBuffer(),
                                 spec.get(), status.get());
            if (api.isError(status.get())) {
              llvm::errs() << "Error: " << api.getError(status.get()) << "\n";
              exit(EXIT_FAILURE);
            }
          }

          TensorMapPtr outputTensors = {nullptr, api.freeAsyncTensorMap};

          M_AsyncTensorMap *outputTensorMap = api.executeModelSync(
              context, modularModel, inputTensors.get(), status.get());

          if (api.isError(status.get())) {
            llvm::errs() << "Error: " << api.getError(status.get()) << "\n";
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
                api.getTensorByNameFrom(outputTensors.get(),
                                        model->getOutputNames()[i].c_str(),
                                        status.get()),
                api.freeTensor};

            if (api.isError(status.get())) {
              llvm::errs() << "Error: " << api.getError(status.get()) << "\n";
              exit(EXIT_FAILURE);
            }
            // Create an uninitialized GenericML Tensor to hold the output.
            auto tensor =
                Tensor::createOwned(getTensorSpec(api, outputTensor.get()),
                                    runtime, kPreferredMemoryAlignment);

            // Copy the output data to the output GenericML Tensor
            LLCL::profiledMemcpy(tensor.getMutableBuffer(),
                                 api.getTensorData(outputTensor.get()),
                                 getSizeOf(api, outputTensor.get()));

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
    auto model = getModel();
    const auto &api = model->getApi();

    StatusPtr status{api.newStatus(), api.freeStatus};
    api.waitForModel(initializedModel.get(), status.get());
    if (api.isError(status.get()))
      return Error(api.getError(status.get()));
    return success();
  }

private:
  ModelPtr initializedModel;
};

} // namespace

/// Initialize the model using `runtime`, allocating memory, setting up data
/// structures, or whatever else the model needs.  This never blocks.
ErrorOr<std::unique_ptr<ExecutableModel>>
ModularModel::setup(Runtime &runtime) const {
  checkUniqueRuntime(runtime);
  checkKnownCallingThread(runtime);
  // Create the executable model.
  StatusPtr status = {api.newStatus(), api.freeStatus};
  ModelPtr initializedModel = {
      api.initModel(modelState.context, modelState.compiledModel, status.get()),
      api.freeModel};
  return std::make_unique<ModularExecutableModel>(this, runtime,
                                                  std::move(initializedModel));
}

template <typename SpecFunc>
ErrorOr<SmallVector<ModelSpec::ModelIOSpec>>
convertSpecs(M_AsyncCompiledModel *model, TensorNamesPtr specNames,
             size_t specCount, SpecFunc func, ModularAPIInterface &api) {
  SmallVector<ModelSpec::ModelIOSpec> specs;
  StatusPtr status = {api.newStatus(), api.freeStatus};
  for (size_t i = 0; i < specCount; ++i) {
    std::string specName = api.getTensorNameAt(specNames.get(), i);
    TensorSpecPtr spec = {func(model, specName.c_str(), status.get()),
                          api.freeTensorSpec};
    if (api.isError(status.get()))
      return Error(api.getError(status.get()));
    specs.emplace_back(
        ModelSpec::ModelIOSpec{specName, getTensorSpec(api, spec.get())});
  }
  return specs;
}

/// Create ModelSpec from a compiledModel.
static ErrorOr<ModelSpec> getModelSpec(M_AsyncCompiledModel *model,
                                       ModularAPIInterface &api) {
  ModelSpec modelSpec;
  StatusPtr status = {api.newStatus(), api.freeStatus};
  TensorNamesPtr inputNames = {api.getInputNames(model, status.get()),
                               api.freeTensorNameArray};
  if (api.isError(status.get()))
    return Error(api.getError(status.get()));
  size_t inputCount = api.getNumModelInputs(model, status.get());
  if (api.isError(status.get()))
    return Error(api.getError(status.get()));
  auto inputsOr = convertSpecs(model, std::move(inputNames), inputCount,
                               api.getModelInputSpecByName, api);
  if (inputsOr.isError())
    return inputsOr.takeError();
  modelSpec.inputs = std::move(*inputsOr);

  TensorNamesPtr outputNames = {api.getOutputNames(model, status.get()),
                                api.freeTensorNameArray};
  if (api.isError(status.get()))
    return Error(api.getError(status.get()));

  size_t outputCount = api.getNumModelOutputs(model, status.get());
  if (api.isError(status.get()))
    return Error(api.getError(status.get()));
  auto outputsOr = convertSpecs(model, std::move(outputNames), outputCount,
                                api.getModelOutputSpecByName, api);
  if (outputsOr.isError())
    return outputsOr.takeError();
  modelSpec.outputs = std::move(*outputsOr);

  return modelSpec;
}

/// Print model metadata to the console.
static void printSpecs(const ModelSpec &modelSpec) {
  llvm::outs() << "Model Metadata:\n";
  llvm::outs() << "Inputs:\n";

  auto printSpec = [](const ModelSpec::ModelIOSpec &ioSpec) {
    llvm::outs() << "- "
                 << "name: " << ioSpec.name << "\n";
    llvm::outs() << "  "
                 << "spec: ";
    std::string specDim = ioSpec.spec.getAsString();

    // In API variable dimension is represented by -1. So do
    // the conversion.
    for (char c : specDim) {
      if (c == '?')
        llvm::outs() << "-1";
      else
        llvm::outs() << c;
    }
    llvm::outs() << "\n";
  };

  for (const auto &input : modelSpec.inputs)
    printSpec(input);

  llvm::outs() << "Outputs:\n";
  for (const auto &output : modelSpec.outputs)
    printSpec(output);
}

int main(int argc, char **argv) {
  ModularExecutorDriverCLOptions clOptions(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "modular-api-executor driver\n");

  M::ModularAPIInterface api;
  if (auto err = api.loadModularAndInitSymbols())
    exit(clOptions.reportError(err.getError()));

  if (clOptions.frameworkVersion) {
    llvm::outs() << "Framework version: " << api.version() << "\n";
    exit(EXIT_SUCCESS);
  }

  // We don't really have a file to load here now.
  std::unique_ptr<llvm::MemoryBuffer> inputFile;

  StatusPtr status{api.newStatus(), api.freeStatus};
  RuntimeConfigPtr runtimeConfig{api.newRuntimeConfig(), api.freeRuntimeConfig};
  api.setNumThreads(runtimeConfig.get(), clOptions.getNumThreads());
  RuntimeContextPtr context = {
      api.newRuntimeContext(runtimeConfig.get(), status.get()),
      api.freeRuntimeContext};

  if (api.isError(status.get())) {
    llvm::errs() << "Error: " << api.getError(status.get()) << "\n";
    exit(EXIT_FAILURE);
  }

  CompileConfigPtr compileConfig = {api.newCompileConfig(),
                                    api.freeCompileConfig};
  api.setModelPath(compileConfig.get(), clOptions.inputFilename.c_str());
  if (!clOptions.exportedName.empty())
    api.setExportedName(compileConfig.get(), clOptions.exportedName.c_str());
  CompiledModelPtr compiledModel{
      api.compileModel(context.get(), compileConfig.get(), status.get()),
      api.freeCompiledModel};

  api.waitForCompilation(compiledModel.get(), status.get());

  if (api.isError(status.get())) {
    llvm::errs() << "Error: " << api.getError(status.get()) << "\n";
    exit(EXIT_FAILURE);
  }

  ModelSpec modelSpec;
  if (clOptions.modelSpecificationFile.empty() || clOptions.printMetadata) {
    auto modelSpecOr = getModelSpec(compiledModel.get(), api);
    if (modelSpecOr.isError())
      exit(clOptions.reportError(modelSpecOr.getError()));
    else
      modelSpec = modelSpecOr.takeValue();
    if (clOptions.printMetadata) {
      printSpecs(modelSpec);
      return EXIT_SUCCESS;
    }
  } else {
    auto modelSpecOr = clOptions.readModelSpecification();
    if (modelSpecOr.isError())
      exit(clOptions.reportError(modelSpecOr.getError()));
    else
      modelSpec = modelSpecOr.takeValue();
  }

  ModelState state = {context.get(), compiledModel.get()};
  EncodedLocation loc = LLCL::FileNameLocationDecoder::getEncodedLocation(
      clOptions.inputFilename);

  // Create the ModularModel. Note that the Model object will outlive
  // the execution.
  auto model = std::make_unique<ModularModel>(
      api, clOptions.benchmarkName, clOptions.inputNames, clOptions.outputNames,
      modelSpec, state, std::move(loc));

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
