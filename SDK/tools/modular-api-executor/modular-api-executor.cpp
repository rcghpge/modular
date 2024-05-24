//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ArraySupport/Tensor.h"
#include "GenericML/Driver/ExecutionDriver.h"
#include "GenericML/Driver/Utils.h"
#include "GenericML/Support/FormatYAML.h"
#include "GenericML/Support/Model.h"
#include "GenericML/Support/ModelSpec.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Support/FileLocationDecoder.h"
#include "LLCL/Support/UnknownLocationDecoder.h"
#include "Support/DynamicLibrary.h"
#include "Support/ML/DType.h"
#include "llvm/ADT/StringExtras.h"

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
  case M_BFLOAT16:
    return DType::bf16;
  case M_BOOL:
    return DType::kBool;
  }
  llvm_unreachable("invalid data type");
}
/// Translates a GeneralML DType to C data type.
/// TODO
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
  case DType::bf16:
    return M_BFLOAT16;
  case DType::invalid:
    [[fallthrough]];
  default:
    return M_UNKNOWN;
  }
}

/// Convert TensorSpec from API to GenericML TensorSpec.
static M::TensorSpec getTensorSpec(const ModularAPIInterface &api,
                                   M_TensorSpec *spec) {
  M_Dtype dtype = api.getDtype(spec);
  SmallVector<ssize_t> inputShape;
  if (api.isDynamicRanked(spec))
    return M::TensorSpec(kDynamicallyRanked, convertToDType(dtype));

  int64_t numDims = api.getRank(spec);
  assert(numDims >= 0 && "for this usage this should always be positive");
  inputShape.reserve(numDims);
  for (size_t i = 0, e = (size_t)numDims; i < e; ++i) {
    if (api.getDimAt(spec, i) == api.getDynamicDimensionValue()) {
      // Shapes in the Tensor layer need to follow MLIR convention
      // where dynamic dimensions are represented by constant kDynamic which is
      // equal to std::numeric_limits<int64_t>::min()
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
  bool gpu{false};

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
      "torch-library",
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

  M::cl::MOpt<bool, true> gpu{
      "gpu", cl::desc("Enable execution using a GPU device."),
      llvm::cl::location(options.gpu),
      llvm::cl::cat(ModularExecutorDriverCLOptionsCategory)};
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

  const ModularAPIInterface &getApi() const { return api; }
  // Gets the name of the model.
  StringRef getName() const override { return name; }

  const EncodedLocation &getLocation() const override { return location; }

  /// Gets the ModelSpec for this model.
  const ModelSpec &getModelSpec() const { return spec; }

  const M_RuntimeContext *getContext() const { return modelState.context; };

  LLCL::ErrorDiagnosticOr<std::unique_ptr<ExecutableModel>>
  setup(Runtime &runtime) const override;

  ~ModularModel() override = default;

private:
  ModularAPIInterface &api;
  ModelSpec spec;
  std::string name;
  ModelState modelState;
  EncodedLocation location;
};

static Tensor getTensorFromValue(const M::ModularAPIInterface &api,
                                 M_AsyncValue *outputValue,
                                 LLCL::Runtime &runtime) {
  TensorPtr outputTensor = {api.getTensorFromValue(outputValue),
                            api.freeTensor};

  // Create an uninitialized GenericML Tensor to hold the output.
  auto tensor = Tensor::createOwned(getTensorSpec(api, outputTensor.get()),
                                    runtime, kPreferredMemoryAlignment);

  // Copy the output data to the output GenericML Tensor
  LLCL::profiledMemcpy(tensor.getMutableBuffer(),
                       api.getTensorData(outputTensor.get()),
                       getSizeOf(api, outputTensor.get()));
  return tensor;
}

static std::string getStringFromValue(const M::ModularAPIInterface &api,
                                      M_AsyncValue *outputValue) {
  return api.getStringFromValue(outputValue);
}

static double getDoubleFromValue(const M::ModularAPIInterface &api,
                                 M_AsyncValue *outputValue) {
  return api.getDoubleFromValue(outputValue);
}

static int64_t getLongFromValue(const M::ModularAPIInterface &api,
                                M_AsyncValue *outputValue) {
  return api.getLongFromValue(outputValue);
}

static bool getBoolFromValue(const M::ModularAPIInterface &api,
                             M_AsyncValue *outputValue) {
  return api.getBoolFromValue(outputValue);
}

static AnyAsyncValueRef getAsyncValue(ValuePtr value,
                                      const M::ModularAPIInterface &api,
                                      LLCL::Runtime &runtime);

static GML::Dict getDictFromValue(const M::ModularAPIInterface &api,
                                  M_AsyncValue *outputValue,
                                  LLCL::Runtime &runtime) {
  DictPtr outputDict = {api.getDictFromValue(outputValue), api.freeDict};
  size_t dictSize = api.getDictSize(outputDict.get());
  GML::Dict dict;
  for (size_t i = 0; i < dictSize; ++i) {
    ValuePtr key = {api.getDictKey(outputDict.get(), i), api.freeValue};
    ValuePtr value = {api.getDictValue(outputDict.get(), i), api.freeValue};
    dict.getElements().insert(
        {GML::Variant(getAsyncValue(std::move(key), api, runtime)),
         GML::Variant(getAsyncValue(std::move(value), api, runtime))});
  }
  return dict;
}

static GML::List getListFromValue(const M::ModularAPIInterface &api,
                                  M_AsyncValue *outputValue,
                                  LLCL::Runtime &runtime) {
  ListPtr outputList = {api.getListFromValue(outputValue), api.freeList};
  size_t listSize = api.getListSize(outputList.get());
  SmallVector<AnyAsyncValueRef> list;
  list.reserve(listSize);
  for (size_t i = 0; i < listSize; ++i) {
    ValuePtr value = {api.getListValue(outputList.get(), i), api.freeValue};
    list.push_back(getAsyncValue(std::move(value), api, runtime));
  }
  return GML::List(list);
}

static GML::Tuple getTupleFromValue(const M::ModularAPIInterface &api,
                                    M_AsyncValue *outputValue,
                                    LLCL::Runtime &runtime) {
  TuplePtr outputTuple = {api.getTupleFromValue(outputValue), api.freeTuple};
  size_t tupleSize = api.getTupleSize(outputTuple.get());
  GML::Tuple tuple;
  for (size_t i = 0; i < tupleSize; ++i) {
    ValuePtr value = {api.getTupleValue(outputTuple.get(), i), api.freeValue};
    tuple.getElements().push_back(
        GML::Variant(getAsyncValue(std::move(value), api, runtime)));
  }
  return tuple;
}

AnyAsyncValueRef getAsyncValue(ValuePtr value,
                               const M::ModularAPIInterface &api,
                               LLCL::Runtime &runtime) {
  M_ValueType valueType = api.getValueType(value.get());
  switch (valueType) {
  case M_STRING_VALUE:
    return AnyAsyncValueRef::createReady<std::string>(
        runtime, getStringFromValue(api, value.get()));
  case M_DOUBLE_VALUE:
    return AnyAsyncValueRef::createReady<double>(
        runtime, getDoubleFromValue(api, value.get()));
  case M_LONG_VALUE:
    return AnyAsyncValueRef::createReady<int64_t>(
        runtime, getLongFromValue(api, value.get()));
  case M_BOOL_VALUE:
    return AnyAsyncValueRef::createReady<bool>(
        runtime, getBoolFromValue(api, value.get()));
  case M_TENSOR_VALUE:
    return AnyAsyncValueRef::createReady<Tensor>(
        runtime, getTensorFromValue(api, value.get(), runtime));
  case M_NONE_VALUE:
    return AnyAsyncValueRef::createReady<GML::None>(runtime);
  case M_DICT_VALUE:
    return AnyAsyncValueRef::createReady<GML::Dict>(
        runtime, getDictFromValue(api, value.get(), runtime));
  case M_LIST_VALUE:
    return AnyAsyncValueRef::createReady<GML::List>(
        runtime, getListFromValue(api, value.get(), runtime));
  case M_TUPLE_VALUE:
    return AnyAsyncValueRef::createReady<GML::Tuple>(
        runtime, getTupleFromValue(api, value.get(), runtime));
  case M_UNKNOWN_VALUE:
    break;
  }
  LLCL::EncodedDiagnostic diagnostic =
      LLCL::UnknownLocationDecoder::getDiagnostic(Error("unsupported value"));
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
            ValuePtr outputValue = {
                api.getValueByNameFrom(outputTensors.get(),
                                       model->getOutputNames()[i].c_str(),
                                       status.get()),
                api.freeValue};

            if (api.isError(status.get())) {
              llvm::errs() << "Error: " << api.getError(status.get()) << "\n";
              exit(EXIT_FAILURE);
            }

            std::move(outputs[i + 1])
                .resolveIndirect(
                    getAsyncValue(std::move(outputValue), api, runtime));
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
LLCL::ErrorDiagnosticOr<std::unique_ptr<ExecutableModel>>
ModularModel::setup(Runtime &runtime) const {
  checkUniqueRuntime(runtime);
  checkKnownCallingThread(runtime);
  // Create the executable model.
  StatusPtr status = {api.newStatus(), api.freeStatus};
  ModelPtr initializedModel = {
      api.initModel(modelState.context, modelState.compiledModel, status.get()),
      api.freeModel};
  return {std::make_unique<ModularExecutableModel>(
      this, runtime, std::move(initializedModel))};
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
  ModularExecutorDriverOptions clOptions;
  ModularExecutorDriverCLOptions parser(argc, argv, clOptions);
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
  std::unique_ptr<Runtime> runtime =
      clOptions.withMainWillNotDonate().createRuntime();

  if (clOptions.gpu)
    api.setDevice(runtimeConfig.get(), "cuda", /*deviceID=*/0);
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

  if (api.useKernelsFrom && !clOptions.kernelLibraryPaths.empty()) {
    for (auto &path : clOptions.kernelLibraryPaths)
      api.useKernelsFrom(compileConfig.get(), path.data(), status.get());
  }

  if (api.isError(status.get())) {
    llvm::errs() << "Error: " << api.getError(status.get()) << "\n";
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
      TorchInputSpecPtr spec{nullptr, api.freeTorchInputSpec};
      if (inputSpec.hasRank()) {
        SmallVector<int64_t> inputShape;
        for (auto dim : inputSpec) {
          if (mlir::ShapedType::isDynamic(dim))
            inputShape.push_back(api.getDynamicDimensionValue());
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
        spec.reset(api.newTorchInputSpec(
            inputShape.data(), dimNames, inputShape.size(),
            convertToCDataType(inputSpec.getEltType()), status.get()));
      } else {
        spec.reset(api.newTorchInputSpec(
            nullptr, nullptr, api.getDynamicRankValue(),
            convertToCDataType(inputSpec.getEltType()), status.get()));
      }
      if (api.isError(status.get())) {
        llvm::errs() << "Error: " << api.getError(status.get()) << "\n";
        exit(EXIT_FAILURE);
      }

      assert(spec && "expected spec to have value by now");
      specPointers.push_back(spec.get());
      specOwners.push_back(std::move(spec));
    }
    api.setTorchInputSpecs(configPtr, specPointers.data(), specPointers.size());
  }
  if (!clOptions.extraTorchLibs.empty()) {
    size_t size = clOptions.extraTorchLibs.size();
    auto array = std::make_unique<char *[]>(size);
    for (size_t i = 0; i < size; ++i) {
      array[i] = const_cast<char *>(clOptions.extraTorchLibs[i].c_str());
    }
    api.setTorchLibraries(configPtr, array.get(), size);
  }
  CompiledModelPtr compiledModel{
      api.compileModel(context.get(), &configPtr, status.get()),
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

  // Run with leak checking.
  if (failed(clOptions.configureMLIRContextAndExecute(
          std::move(inputFile), [&](MLIRContext *ctx) {
            return LLCL::runWithLeakCheckedRuntime(
                "modular-api-executor", [&]() {
                  return runModel(ctx, model.get(), *runtime, clOptions);
                });
          })))
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
