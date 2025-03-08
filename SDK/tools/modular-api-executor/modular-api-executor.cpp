//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/Utils.h"
#include "AsyncRT/Support/FileLocationDecoder.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "GenericML/API/c/detail/helpers.h"
#include "GenericML/Driver/DriverCLOptions.h"
#include "GenericML/Driver/ExecutionDriver.h"
#include "GenericML/Driver/Utils.h"
#include "GenericML/Support/FormatYAML.h"
#include "GenericML/Support/Model.h"
#include "GenericML/Support/ModelSpec.h"
#include "Init/Init.h"
#include "Kernels/WeightsRegistry.h"
#include "Runtime/Tensor/Tensor.h"
#include "Support/DynamicLibrary.h"
#include "Support/ML/DType.h"
#include "max/c/common.h"
#include "max/c/custom_ops.h"
#include "max/c/internal/context.h"
#include "max/cpp/aliases.h"
#include "max/cpp/pytorch/aliases.h"
#include "llvm/ADT/StringExtras.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace M;

/// Convert TensorSpec from API to GenericML TensorSpec.
static M::TensorSpec getTensorSpec(M_TensorSpec *spec) {
  M_Dtype dtype = M_getDtype(spec);
  SmallVector<ssize_t> inputShape;
  if (M_isDynamicRanked(spec))
    return M::TensorSpec(kDynamicallyRanked, M::Detail::mapDType(dtype));

  int64_t numDims = M_getRank(spec);
  assert(numDims >= 0 && "for this usage this should always be positive");
  inputShape.reserve(numDims);
  for (size_t i = 0, e = (size_t)numDims; i < e; ++i) {
    if (M_getDimAt(spec, i) == M_getDynamicDimensionValue()) {
      // Shapes in the Tensor layer need to follow MLIR convention
      // where dynamic dimensions are represented by constant kDynamic which is
      // equal to std::numeric_limits<int64_t>::min()
      inputShape.push_back(mlir::ShapedType::kDynamic);
    } else {
      inputShape.push_back(M_getDimAt(spec, i));
    }
  }
  return {inputShape, M::Detail::mapDType(dtype)};
}

/// Get TensorSpec from a API Tensor.
static M::TensorSpec getTensorSpec(M_AsyncTensor *tensor) {
  TensorSpecPtr spec{M_getTensorSpec(tensor)};
  return getTensorSpec(spec.get());
}

/// Get size of a tensor in bytes.
static size_t getSizeOf(M_AsyncTensor *tensor) {
  M_Dtype dtype = M_getTensorType(tensor);
  return M_getTensorNumElements(tensor) * M_sizeOf(dtype);
}

namespace {

class ModularExecutorDriverOptions : public ExecutorDriverOptions {
public:
  std::string exportedName;
  std::string modelSpecificationFile;
  bool frameworkVersion{false};
  SmallVector<std::string> inputNames;
  SmallVector<std::string> outputNames;
  SmallVector<TensorSpec> compileSpecs;
  SmallVector<std::string> inputDimNames;
  SmallVector<std::string> extraTorchLibs;
  bool printMetadata{false};
  SmallVector<std::string> kernelLibraryPaths;

  /// Reads a model specification from the YAML file set via the
  /// "model-specification" option.
  ErrorOr<ModelSpec> readModelSpecification() {
    if (modelSpecificationFile.empty())
      return Error("the model specification file path must be specified.");

    std::error_code errorCode;
    if (!std::filesystem::exists(std::filesystem::path{modelSpecificationFile},
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

/// Contains the core CLI options for modular-api-executor.
class ModularExecutorDriverCLOptions : public ExecutorDriverCLOptions {
public:
  ModularExecutorDriverOptions &options;
  ModularExecutorDriverCLOptions(int argc, char **argv,
                                 ModularExecutorDriverOptions &o,
                                 bool skipInitLLVM = false)
      : ExecutorDriverCLOptions(argc, argv, o, skipInitLLVM), options(o) {}

private:
  llvm::cl::OptionCategory ModularExecutorDriverCLOptionsCategory{
      "Modular executor driver command line options"};

  M::cl::MOpt<std::string, true> exportedName{
      "exported-name",
      cl::desc("Signature name of TF model in model with multiple signature."),
      llvm::cl::location(options.exportedName),
      llvm::cl::cat(ModularExecutorDriverCLOptionsCategory)};

  M::cl::MOpt<std::string, true> modelSpecificationFile{
      "model-specification", cl::desc("Path to the model specification file."),
      llvm::cl::location(options.modelSpecificationFile),
      llvm::cl::cat(ModularExecutorDriverCLOptionsCategory)};

  M::cl::MOpt<bool, true> frameworkVersion{
      "framework-version", cl::desc("Prints the external framework's version."),
      llvm::cl::location(options.frameworkVersion),
      llvm::cl::cat(ModularExecutorDriverCLOptionsCategory)};

  M::cl::MListOpt<std::string, SmallVector<std::string>> inputNames{
      "input-names",
      cl::desc("Specify input names. This is always required "
               "and it must match the model's input names"),
      llvm::cl::location(options.inputNames),
      llvm::cl::cat(ModularExecutorDriverCLOptionsCategory)};

  M::cl::MListOpt<std::string, SmallVector<std::string>> outputNames{
      "output-names",
      cl::desc("Specify output names. This is always required "
               "and it must match the model's output names"),
      llvm::cl::location(options.outputNames),
      llvm::cl::cat(ModularExecutorDriverCLOptionsCategory)};

  M::cl::MListOpt<std::string, llvm::SmallVector<std::string>> extraTorchLibs{
      "torch-library-paths",
      cl::desc("Path to a custom library built for PyTorch "
               "(for example a custom operator library)."),
      llvm::cl::location(options.extraTorchLibs),
      llvm::cl::cat(ModularExecutorDriverCLOptionsCategory)};
  M::cl::MListOpt<TensorSpec, SmallVector<TensorSpec>, TensorSpecParser>
      compileSpecs{
          "compile-specs",
          cl::desc(
              "Specify the input shapes to compile model with. "
              "This is only required for PyTorch models. If both are "
              "given compile-specs will be preferred over input-specs at model "
              "compile time and input-specs will used at runtime."),
          llvm::cl::location(options.compileSpecs),
          llvm::cl::cat(ModularExecutorDriverCLOptionsCategory)};
  M::cl::MListOpt<std::string, SmallVector<std::string>> inputDimNames{
      "input-dim-names",
      cl::desc("Specify names of input dimensions to compile model with. "
               "This is only used with PyTorch models. "
               "Inputs should be separated by commas. "
               "Dimensions in each input should be separated by semcolons. "
               "Unnamed dimensions should be marked with a hyphen."),
      llvm::cl::location(options.inputDimNames),
      llvm::cl::cat(ModularExecutorDriverCLOptionsCategory)};

  M::cl::MOpt<bool, true> printMetadata{
      "print-metadata", cl::desc("Display Model Input and output Specs"),
      llvm::cl::location(options.printMetadata),
      llvm::cl::cat(ModularExecutorDriverCLOptionsCategory)};

  M::cl::MListOpt<std::string, SmallVector<std::string>> kernelLibraryPaths{
      "kernel-library-paths",
      cl::desc("Paths to mojo packages containing mojo kernel definitions used "
               "to replace MO and Framework ops."),
      llvm::cl::location(options.kernelLibraryPaths),
      llvm::cl::cat(ModularExecutorDriverCLOptionsCategory)};
};

class ModularExecutableModel;

struct ModelState {
  M_RuntimeContext *context;
  M_AsyncCompiledModel *compiledModel;
};

class ModularModel final : public Model {
public:
  ModularModel(std::string_view name, ArrayRef<std::string> inputTensorNames,
               ArrayRef<std::string> outputTensorNames, ModelSpec modelSpec,
               ModelState state, EncodedLocation loc)
      : Model(modelSpec),           // create a model from the ModelSpec
        spec(std::move(modelSpec)), // move the model spec
        name(name),                 // the name of the the benchmark
        modelState(state), location(std::move(loc)) {
    inputNames = SmallVector<std::string>(inputTensorNames);
    outputNames = SmallVector<std::string>(outputTensorNames);

    if (inputNames.empty())
      llvm::transform(spec.inputs, std::back_inserter(inputNames),
                      [](const auto &input) { return input.name; });

    if (outputNames.empty())
      llvm::transform(spec.outputs, std::back_inserter(outputNames),
                      [](const auto &output) { return output.name; });
  }

  // Gets the name of the model.
  StringRef getName() const override { return name; }

  const EncodedLocation &getLocation() const override { return location; }

  /// Gets the ModelSpec for this model.
  const ModelSpec &getModelSpec() const { return spec; }

  const M_RuntimeContext *getContext() const { return modelState.context; }

  AsyncRT::ErrorDiagnosticOr<std::unique_ptr<ExecutableModel>> setup(
      Runtime &runtime, const WeightsRegistry *weightsRegistry = {},
      const Driver::DeviceMap &deviceMap = Driver::DeviceMap()) const override;

  ~ModularModel() override = default;

  /// Return the devices used by the model.
  ArrayRef<M::DeviceRef> getDevices() const override {
    llvm_unreachable("unimplemented");
  }
  /// Return an array of devices that map 1:1 to the devices used by each input.
  ArrayRef<M::DeviceRef> getInputDevices() const override {
    llvm_unreachable("unimplemented");
  }
  /// Return an array of devices that map 1:1 to the devices used by each
  /// output.
  ArrayRef<M::DeviceRef> getOutputDevices() const override {
    llvm_unreachable("unimplemented");
  }

private:
  ModelSpec spec;
  std::string name;
  ModelState modelState;
  EncodedLocation location;
};

static Tensor getTensorFromValue(M_AsyncValue *outputValue,
                                 AsyncRT::Runtime &runtime) {
  TensorPtr outputTensor{M_getTensorFromValue(outputValue)};

  // Create an uninitialized GenericML Tensor to hold the output.
  auto tensor = Tensor::createOwned(getTensorSpec(outputTensor.get()), runtime,
                                    kPreferredMemoryAlignment);

  // Copy the output data to the output GenericML Tensor
  AsyncRT::profiledMemcpy(tensor.getMutableBuffer(),
                          M_getTensorData(outputTensor.get()),
                          getSizeOf(outputTensor.get()));
  return tensor;
}

static std::string getStringFromValue(M_AsyncValue *outputValue) {
  return M_getStringFromValue(outputValue);
}

static double getDoubleFromValue(M_AsyncValue *outputValue) {
  return M_getDoubleFromValue(outputValue);
}

static int64_t getLongFromValue(M_AsyncValue *outputValue) {
  return M_getLongFromValue(outputValue);
}

static bool getBoolFromValue(M_AsyncValue *outputValue) {
  return M_getBoolFromValue(outputValue);
}

static AnyAsyncValueRef getAsyncValue(ValuePtr value,

                                      AsyncRT::Runtime &runtime);

static GML::Dict getDictFromValue(M_AsyncValue *outputValue,
                                  AsyncRT::Runtime &runtime) {
  DictPtr outputDict{M_getDictFromValue(outputValue)};
  size_t dictSize = M_getDictSize(outputDict.get());
  GML::Dict dict;
  for (size_t i = 0; i < dictSize; ++i) {
    ValuePtr key{M_getDictKey(outputDict.get(), i)};
    ValuePtr value{M_getDictValue(outputDict.get(), i)};
    dict.getElements().insert(
        {GML::Variant(getAsyncValue(std::move(key), runtime)),
         GML::Variant(getAsyncValue(std::move(value), runtime))});
  }
  return dict;
}

static GML::List getListFromValue(M_AsyncValue *outputValue,
                                  AsyncRT::Runtime &runtime) {
  ListPtr outputList{M_getListFromValue(outputValue)};
  size_t listSize = M_getListSize(outputList.get());
  SmallVector<AnyAsyncValueRef> list;
  list.reserve(listSize);
  for (size_t i = 0; i < listSize; ++i) {
    ValuePtr value{M_getListValue(outputList.get(), i)};
    list.push_back(getAsyncValue(std::move(value), runtime));
  }
  return GML::List(list);
}

static GML::Tuple getTupleFromValue(M_AsyncValue *outputValue,
                                    AsyncRT::Runtime &runtime) {
  TuplePtr outputTuple{M_getTupleFromValue(outputValue)};
  size_t tupleSize = M_getTupleSize(outputTuple.get());
  GML::Tuple tuple;
  for (size_t i = 0; i < tupleSize; ++i) {
    ValuePtr value{M_getTupleValue(outputTuple.get(), i)};
    tuple.getElements().push_back(
        GML::Variant(getAsyncValue(std::move(value), runtime)));
  }
  return tuple;
}

AnyAsyncValueRef getAsyncValue(ValuePtr value,

                               AsyncRT::Runtime &runtime) {
  M_ValueType valueType = M_getValueType(value.get());
  switch (valueType) {
  case M_STRING_VALUE:
    return AnyAsyncValueRef::createReady<std::string>(
        runtime, getStringFromValue(value.get()));
  case M_DOUBLE_VALUE:
    return AnyAsyncValueRef::createReady<double>(
        runtime, getDoubleFromValue(value.get()));
  case M_LONG_VALUE:
    return AnyAsyncValueRef::createReady<int64_t>(
        runtime, getLongFromValue(value.get()));
  case M_BOOL_VALUE:
    return AnyAsyncValueRef::createReady<bool>(runtime,
                                               getBoolFromValue(value.get()));
  case M_TENSOR_VALUE:
    return AnyAsyncValueRef::createReady<Tensor>(
        runtime, getTensorFromValue(value.get(), runtime));
  case M_NONE_VALUE:
    return AnyAsyncValueRef::createReady<GML::None>(runtime);
  case M_DICT_VALUE:
    return AnyAsyncValueRef::createReady<GML::Dict>(
        runtime, getDictFromValue(value.get(), runtime));
  case M_LIST_VALUE:
    return AnyAsyncValueRef::createReady<GML::List>(
        runtime, getListFromValue(value.get(), runtime));
  case M_TUPLE_VALUE:
    return AnyAsyncValueRef::createReady<GML::Tuple>(
        runtime, getTupleFromValue(value.get(), runtime));
  case M_UNKNOWN_VALUE:
  case M_MOJO_VALUE:
  case M_PYTHON_MOJO_VALUE:
    break;
  }
  AsyncRT::EncodedDiagnostic diagnostic =
      AsyncRT::UnknownLocationDecoder::getDiagnostic(
          Error("unsupported value"));
  return AnyAsyncValueRef::createError(runtime, std::move(diagnostic));
}

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

  AnyAsyncValueRef
  convertToGenericValue(AnyAsyncValueRef unknownValue) const final {
    if (unknownValue.isType<Tensor>())
      return convertToGenericTensor(std::move(unknownValue));

    return unknownValue;
  }
  // Executes a single instance of the inference process.
  [[nodiscard]] SmallVector<AnyAsyncValueRef>
  execute(MutableArrayRef<AnyAsyncValueRef> args) const override {
    // Get the runtime value so that we can construct the output tensors (and
    // not depend on the current class).
    Runtime &runtime = getRuntime();

    checkUniqueRuntime(runtime);

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
    outputs.emplace_back(AsyncValueRef<Chain>::allocate(runtime));

    // Create the async placeholders for the outputs. The async values will be
    // materialized when the inference is complete.
    for (size_t i = 0, e = modelSpec.outputs.size(); i < e; ++i)
      outputs.emplace_back(AnyAsyncValueRef::createIndirect(runtime));

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
        args, [modularModel, model, outputs = std::move(outputCopy),
               &runtime](MutableArrayRef<AnyAsyncValueRef> args) mutable {
          const M_RuntimeContext *context = model->getContext();
          TensorMapPtr inputTensors{M_newAsyncTensorMap(context)};

          StatusPtr status{M_newStatus()};
          for (auto &&[inputIndex, input] : llvm::enumerate(args)) {
            auto &inputTensor = input.get<Tensor>();

            SmallVector<int64_t> inputShape;
            for (const auto &dim : inputTensor.getShape())
              inputShape.emplace_back(dim);
            TensorSpecPtr spec{
                M_newTensorSpec(inputShape.data(), inputShape.size(),
                                M::Detail::mapDType(inputTensor.getEltType()),
                                model->getInputNames()[inputIndex].c_str())};
            M_borrowTensorInto(inputTensors.get(), inputTensor.getBuffer(),
                               spec.get(), status.get());
            if (M_isError(status.get())) {
              llvm::errs() << "Error: " << M_getError(status.get()) << "\n";
              exit(EXIT_FAILURE);
            }
          }

          TensorMapPtr outputTensors{nullptr};

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
            ValuePtr outputValue{M_getValueByNameFrom(
                outputTensors.get(), model->getOutputNames()[i].c_str(),
                status.get())};

            if (M_isError(status.get())) {
              llvm::errs() << "Error: " << M_getError(status.get()) << "\n";
              exit(EXIT_FAILURE);
            }

            std::move(outputs[i + 1])
                .resolveIndirect(
                    getAsyncValue(std::move(outputValue), runtime));
          }
        });
    // Return the output async values. Note that the output async values might
    // not be ready at this point. The outputs are marked as ready after the
    // interpreter is invoked in the above lambda.
    return outputs;
  }

  ErrorOrSuccess waitForModelSetup() override {
    StatusPtr status{M_newStatus()};
    M_waitForModel(initializedModel.get(), status.get());
    if (M_isError(status.get()))
      return Error(M_getError(status.get()));
    return success();
  }

private:
  ModelPtr initializedModel;
};

} // namespace

/// Initialize the model using `runtime`, allocating memory, setting up data
/// structures, or whatever else the model needs.  This never blocks.
AsyncRT::ErrorDiagnosticOr<std::unique_ptr<ExecutableModel>>
ModularModel::setup(Runtime &runtime, const WeightsRegistry *weightsRegistry,
                    const Driver::DeviceMap &deviceMap) const {
  checkUniqueRuntime(runtime);
  // Create the executable model.
  StatusPtr status{M_newStatus()};
  ModelPtr initializedModel{
      M_initModel(modelState.context, modelState.compiledModel,
                  /*weightsRegistry=*/nullptr, status.get())};
  return {std::make_unique<ModularExecutableModel>(
      this, runtime, std::move(initializedModel))};
}

template <typename SpecFunc>
ErrorOr<SmallVector<ModelSpec::ModelIOSpec>>
convertSpecs(M_AsyncCompiledModel *model, TensorNamesPtr specNames,
             size_t specCount, SpecFunc func) {
  SmallVector<ModelSpec::ModelIOSpec> specs;
  StatusPtr status{M_newStatus()};
  for (size_t i = 0; i < specCount; ++i) {
    std::string specName = M_getTensorNameAt(specNames.get(), i);
    TensorSpecPtr spec{func(model, specName.c_str(), status.get())};
    if (M_isError(status.get()))
      return Error(M_getError(status.get()));
    specs.emplace_back(
        ModelSpec::ModelIOSpec{specName, getTensorSpec(spec.get())});
  }
  return specs;
}

/// Create ModelSpec from a compiledModel.
static ErrorOr<ModelSpec> getModelSpec(M_AsyncCompiledModel *model) {
  ModelSpec modelSpec;
  StatusPtr status{M_newStatus()};
  TensorNamesPtr inputNames{M_getInputNames(model, status.get())};
  if (M_isError(status.get()))
    return Error(M_getError(status.get()));
  size_t inputCount = M_getNumModelInputs(model, status.get());
  if (M_isError(status.get()))
    return Error(M_getError(status.get()));
  auto inputsOr = convertSpecs(model, std::move(inputNames), inputCount,
                               M_getModelInputSpecByName);
  if (inputsOr.isError())
    return inputsOr.takeError();
  modelSpec.inputs = std::move(*inputsOr);

  TensorNamesPtr outputNames{M_getOutputNames(model, status.get())};
  if (M_isError(status.get()))
    return Error(M_getError(status.get()));

  size_t outputCount = M_getNumModelOutputs(model, status.get());
  if (M_isError(status.get()))
    return Error(M_getError(status.get()));
  auto outputsOr = convertSpecs(model, std::move(outputNames), outputCount,
                                M_getModelOutputSpecByName);
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
  ModularExecutorDriverOptions clOptions;
  ModularExecutorDriverCLOptions parser(argc, argv, clOptions);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "modular-api-executor driver\n");

  if (clOptions.frameworkVersion) {
    llvm::outs() << "Framework version: " << M_version() << "\n";
    exit(EXIT_SUCCESS);
  }

  // We don't really have a file to load here now.
  std::unique_ptr<llvm::MemoryBuffer> inputFile;

  // FIXME: We really shouldn't be mixed and matching internal and external
  // APIs in this tool. Unfortunately there's a lot of implementation that
  // consumes underlying details and breaks these abstractions. For now, work
  // around the double runtime problem by at least setting the maxContext in
  // the created runtime to be the top-level context.
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "modular-api-executor",
      Init::Options().withRuntimeOptions(AsyncRT::RuntimeOptions()));
  if (ctxOr.isError()) {
    llvm::errs() << "Error: " << ctxOr.getError() << "\n";
    exit(EXIT_FAILURE);
  }
  StatusPtr status{M_newStatus()};
  RuntimeConfigPtr runtimeConfig{M_newRuntimeConfig()};
  M_setMaxContext(runtimeConfig.get(),
                  reinterpret_cast<M_MaxContext *>(ctxOr->getPointer()));

  RuntimeContextPtr context{
      M_newRuntimeContext(runtimeConfig.get(), status.get())};

  if (M_isError(status.get())) {
    llvm::errs() << "Error: " << M_getError(status.get()) << "\n";
    exit(EXIT_FAILURE);
  }

  CompileConfigPtr compileConfig{M_newCompileConfig()};
  M_setModelPath(compileConfig.get(), clOptions.inputFilename.c_str());

  if (!clOptions.kernelLibraryPaths.empty()) {
    for (auto &path : clOptions.kernelLibraryPaths)
      M_useKernelsFrom(compileConfig.get(), path.data(), status.get());
  }

  if (M_isError(status.get())) {
    llvm::errs() << "Error: " << M_getError(status.get()) << "\n";
    exit(EXIT_FAILURE);
  }
  M_CompileConfig *configPtr = compileConfig.release();

  // We need to set torch input specs on models in case we're actually running
  // a torch model. For this, we prefer to use `input-specs/compile-specs` over
  // `input-shapes` as the CLI argument input. This allows for us to directly
  // get the dtype of the input tensors.
  if (!clOptions.inputSpecs.empty() || !clOptions.compileSpecs.empty()) {
    std::vector<TorchInputSpecPtr> specOwners;
    std::vector<M_TorchInputSpec *> specPointers;
    SmallVector<TensorSpec> specSource = clOptions.compileSpecs.empty()
                                             ? clOptions.inputSpecs
                                             : clOptions.compileSpecs;
    for (auto &&[inputIndex, inputSpec] : llvm::enumerate(specSource)) {
      TorchInputSpecPtr spec{nullptr};
      if (inputSpec.hasRank()) {
        SmallVector<int64_t> inputShape;
        for (auto dim : inputSpec) {
          if (mlir::ShapedType::isDynamic(dim))
            inputShape.push_back(M_getDynamicDimensionValue());
          else
            inputShape.emplace_back((int64_t)dim);
        }
        std::vector<std::string> dimNamesStorage;
        std::vector<const char *> dimNamesVec;
        const char *const *dimNames = nullptr;
        if (inputIndex < clOptions.inputDimNames.size()) {
          StringRef dimNamesStrRef = clOptions.inputDimNames[inputIndex];
          SmallVector<StringRef> elemRefs;
          dimNamesStrRef.split(elemRefs, ';');
          if (elemRefs.size() != inputShape.size()) {
            llvm::errs() << "Error: Input " << inputIndex << " names setting '"
                         << dimNamesStrRef
                         << "' has different number of elements ("
                         << elemRefs.size() << ") than rank ("
                         << inputShape.size() << ") of shape (" << inputSpec
                         << ")\n";
            exit(EXIT_FAILURE);
          }
          dimNamesStorage.reserve(inputShape.size());
          dimNamesVec.reserve(inputShape.size());
          for (StringRef elemRef : elemRefs) {
            if (elemRef == "-") {
              dimNamesVec.push_back(nullptr);
            } else {
              dimNamesStorage.push_back(elemRef.str());
              dimNamesVec.push_back(dimNamesStorage.back().c_str());
            }
          }
          dimNames = dimNamesVec.data();
        }
        spec.reset(
            M_newTorchInputSpec(inputShape.data(), dimNames, inputShape.size(),
                                M::Detail::mapDType(inputSpec.getEltType()),
                                nullptr, status.get()));
      } else {
        spec.reset(
            M_newTorchInputSpec(nullptr, nullptr, M_getDynamicRankValue(),
                                M::Detail::mapDType(inputSpec.getEltType()),
                                nullptr, status.get()));
      }
      if (M_isError(status.get())) {
        llvm::errs() << "Error: " << M_getError(status.get()) << "\n";
        exit(EXIT_FAILURE);
      }

      assert(spec && "expected spec to have value by now");
      specPointers.push_back(spec.get());
      specOwners.push_back(std::move(spec));
    }
    M_setTorchInputSpecs(configPtr, specPointers.data(), specPointers.size());
  }
  if (!clOptions.extraTorchLibs.empty()) {
    for (std::string &extraTorchLib : clOptions.extraTorchLibs) {
      std::filesystem::path path(extraTorchLib);
      if (!path.empty() && path.has_extension() &&
          (path.filename().string().find(".so.") != std::string::npos ||
           path.extension() == ".so" || path.extension() == ".dylib" ||
           path.extension() == ".dll")) {
        // PyTorch custom ops library.
        M_setTorchLibraryPath(configPtr, extraTorchLib.c_str());
      } else if (!path.empty() && path.has_extension() &&
                 (path.extension() == ".pt" ||
                  path.extension() == ".torchscript")) {
        // PyTorch metadata library.
        M_setTorchMetadataLibraryPath(configPtr, extraTorchLib.c_str());
      } else {
        llvm::errs() << "Error: Unsupported torch library '" << path << "'\n";
        exit(EXIT_FAILURE);
      }
    }
  }
  CompiledModelPtr compiledModel{
      M_compileModel(context.get(), &configPtr, status.get())};

  M_waitForCompilation(compiledModel.get(), status.get());

  if (M_isError(status.get())) {
    llvm::errs() << "Error: " << M_getError(status.get()) << "\n";
    exit(EXIT_FAILURE);
  }

  ModelSpec modelSpec;
  if (clOptions.modelSpecificationFile.empty() || clOptions.printMetadata) {
    auto modelSpecOr = getModelSpec(compiledModel.get());
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
  EncodedLocation loc = AsyncRT::FileNameLocationDecoder::getEncodedLocation(
      clOptions.inputFilename);

  // Create the ModularModel. Note that the Model object will outlive
  // the execution.
  auto model = std::make_unique<ModularModel>(
      clOptions.benchmarkName, clOptions.inputNames, clOptions.outputNames,
      modelSpec, state, std::move(loc));

  // Run with leak checking.
  if (failed(clOptions.configureMLIRContextAndExecute(
          std::move(inputFile), [&](MLIRContext *ctx) {
            return AsyncRT::runWithLeakCheckedRuntime(
                "modular-api-executor", [&]() {
                  return runModel(ctx, model.get(),
                                  *(*ctxOr)->get<AsyncRT::Runtime>(),
                                  clOptions);
                });
          })))
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
