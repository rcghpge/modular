//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MEFModel/MEFModel.h"
#include "GenericML/DeviceDriver/DeviceExecution.h"
#include "GenericML/GraphCompiler/MGPDialect/Primitives/MGPPrimitives.h"
#include "GenericML/Support/FormatYAML.h"
#include "GenericML/Support/MEFTypeUtils.h"
#include "GenericML/Support/ModelSpec.h"
#include "GenericML/Support/QuantSpec.h"
#include "GenericML/Support/ResultInfo.h"
#include "GenericML/Support/Value.h"
#include "MLRT/AsyncRT/Runtime/Algorithms.h"
#include "MLRT/AsyncRT/Runtime/AsyncValue.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "MLRT/Core/MojoValue.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "MLRT/GraphRT/MEF/MEFAttr.h"
#include "MLRT/GraphRT/MEF/MEFTypes.h"
#include "MLRT/GraphRT/MEFExecutor/MEFInvokable.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveDescriptor.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "MLRT/GraphRT/Primitives.h"
#include "Support/AssertStream.h"
#include "Support/DeviceSpecs.h"
#include "Support/HashUtils.h"
#include "Support/ML/CommonAttributes.h"
#include "Support/Profiling/Tracy.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <vector>

using namespace M;

static SmallVector<QuantTensorSpec>
getSpecsAsVector(ArrayRef<ModelSpec::ModelIOSpec> modelIOSpec) {
  SmallVector<QuantTensorSpec> result(modelIOSpec.size());
  llvm::transform(
      modelIOSpec, result.begin(),
      [](const ModelSpec::ModelIOSpec &ioSpec) { return ioSpec.spec; });
  return result;
}

static SmallVector<StringRef>
getNamesAsVector(ArrayRef<ModelSpec::ModelIOSpec> modelIOSpec) {
  SmallVector<StringRef> result(modelIOSpec.size());
  llvm::transform(modelIOSpec, result.begin(),
                  [](const ModelSpec::ModelIOSpec &ioSpec) {
                    return StringRef(ioSpec.name);
                  });
  return result;
}

//===----------------------------------------------------------------------===//
// Model
//===----------------------------------------------------------------------===//

Model::Model(const ModelSpec &spec)
    : name(spec.name),
      location(AsyncRT::UnknownLocationDecoder::getEncodedLocation()) {
  auto inputSpecs = getSpecsAsVector(spec.inputs);
  auto outputSpecs = getSpecsAsVector(spec.outputs);
  auto inputNames = getNamesAsVector(spec.inputs);
  auto outputNames = getNamesAsVector(spec.outputs);
  initializeSignature(inputSpecs, outputSpecs, inputNames, outputNames);
}

void Model::initializeSignature(ArrayRef<QuantTensorSpec> inputSpecs,
                                ArrayRef<QuantTensorSpec> outputSpecs,
                                ArrayRef<StringRef> inputNamesRefs,
                                ArrayRef<StringRef> outputNamesRefs) {
  inputNames.clear();
  outputNames.clear();
  inputTensorNameToIndex.clear();
  outputTensorNameToIndex.clear();

  if (inputNamesRefs.empty()) {
    for (size_t i = 0; i < inputSpecs.size(); ++i)
      inputNames.push_back("input" + std::to_string(i));
  } else {
    inputNames.reserve(inputNamesRefs.size());
    for (StringRef nameRef : inputNamesRefs)
      inputNames.emplace_back(nameRef.str());
  }

  if (outputNamesRefs.empty()) {
    for (size_t i = 0; i < outputSpecs.size(); ++i)
      outputNames.push_back("output" + std::to_string(i));
  } else {
    outputNames.reserve(outputNamesRefs.size());
    for (StringRef nameRef : outputNamesRefs)
      outputNames.emplace_back(nameRef.str());
  }

  this->inputSpecs.assign(inputSpecs.begin(), inputSpecs.end());
  this->outputSpecs.assign(outputSpecs.begin(), outputSpecs.end());

  for (size_t i = 0, e = getNumInputs(); i < e; ++i)
    inputTensorNameToIndex.insert({inputNames[i], i});

  for (size_t i = 0, e = getNumOutputs(); i < e; ++i)
    outputTensorNameToIndex.insert({outputNames[i], i});
}

Model::Model(std::string name, SmallVector<DeviceRef> devices,
             CompiledFrameworkLabel frameworkLabel, const MEFInvokable *initFn,
             const MEFInvokable *executeFn,
             ArrayRef<QuantTensorSpec> inputSpecs,
             ArrayRef<QuantTensorSpec> outputSpecs,
             SmallVector<DeviceRef> inputDevices,
             SmallVector<DeviceRef> outputDevices, EncodedLocation location)
    : name(std::move(name)), initFn(initFn), executeFn(executeFn),
      devices(std::move(devices)), inputDevices(std::move(inputDevices)),
      outputDevices(std::move(outputDevices)), location(std::move(location)) {
  ASSERT_STREAM(frameworkLabel == kFrameworkLabel,
                << "Expected '" << kFrameworkLabel.getAsString()
                << "' label for model, but got '"
                << frameworkLabel.getAsString() << "'");
  initializeSignature(inputSpecs, outputSpecs, {}, {});
}

Model::Model(std::string name, SmallVector<DeviceRef> devices,
             CompiledFrameworkLabel frameworkLabel, const MEFInvokable *initFn,
             const MEFInvokable *executeFn,
             ArrayRef<QuantTensorSpec> inputSpecs,
             ArrayRef<QuantTensorSpec> outputSpecs,
             ArrayRef<StringRef> inputNames, ArrayRef<StringRef> outputNames,
             SmallVector<DeviceRef> inputDevices,
             SmallVector<DeviceRef> outputDevices, EncodedLocation location)
    : name(std::move(name)), initFn(initFn), executeFn(executeFn),
      devices(std::move(devices)), inputDevices(std::move(inputDevices)),
      outputDevices(std::move(outputDevices)), location(std::move(location)) {
  ASSERT_STREAM(frameworkLabel == kFrameworkLabel,
                << "Expected " << kFrameworkLabel.getAsString()
                << " label for model, but got "
                << frameworkLabel.getAsString());
  initializeSignature(inputSpecs, outputSpecs, inputNames, outputNames);
}

AsyncRT::ErrorDiagnosticOr<std::unique_ptr<ExecutableModel>>
Model::setup(Runtime &runtime, const WeightsRegistry *weightsRegistry,
             const M::GenericML::Driver::DeviceMap &deviceMap) const {
  checkUniqueRuntime(runtime);

  SmallVector<AnyAsyncValueRef> initArgs;
  initArgs.reserve(kMgpModelNumFixedImplicitInitArgs + devices.size());

  // Populate init block args with a borrowed ref to the weights registry.
  initArgs.push_back(AnyAsyncValueRef::createReady<MojoValue>(
      runtime, MojoValue::createRef(weightsRegistry ? weightsRegistry->opaque()
                                                    : nullptr)));

  // Set device contexts from the device ref to device mapping.
  auto deviceContextsOr = GenericML::Driver::mapDeviceRefsToDevices(
      getDevices(), deviceMap, runtime);
  if (deviceContextsOr.isError())
    return AsyncRT::EncodedDiagnostic(
        Twine("setup for model ") + getName() +
            " failed to map DeviceRefs to Devices: " +
            Twine(deviceContextsOr.getError()),
        location.copy());
  initArgs.append(std::make_move_iterator(deviceContextsOr->begin()),
                  std::make_move_iterator(deviceContextsOr->end()));

  // Run the init block.
  auto results = initFn->invoke(initArgs, runtime);

  for (AnyAsyncValueRef &result : results) {
    if (result.isError()) {
      auto diagnostic = result.takeDiagnostic();
      return AsyncRT::EncodedDiagnostic(
          Twine("setup for model ") + getName() +
              " returned error: " + std::string(diagnostic.getMessage().get()),
          diagnostic.getLocation().copy());
    }
  }

  SmallVector<AnyAsyncValueRef> extraArgs;
  extraArgs.reserve(results.size() - kMgpModelNumFixedImplicitExecArgs);
  for (size_t i = kMgpModelNumFixedImplicitExecArgs; i < results.size(); ++i)
    extraArgs.push_back(std::move(results[i]));

  return {std::unique_ptr<ExecutableModel>(new ExecutableModel(
      /*chain=*/std::move(results[0]),
      /*mgpContext=*/std::move(results[1]),
      /*weightsRegistry=*/std::move(results[2]),
      /*extraArgs*/ std::move(extraArgs), this, runtime))};
}

ErrorOr<std::unique_ptr<Model>>
Model::getFromDescriptor(const PrimitiveDescriptor &desc) {
  constexpr size_t devicesAttrIdx = 0;
  constexpr size_t frameworkNameAttrIdx = 1;
  constexpr size_t modelFnTypeAttrIdx = 4;
  constexpr size_t inputDeviceIndicesIdx = 5;
  constexpr size_t resultDeviceIndicesIdx = 6;
  constexpr size_t argumentNamesIdx = 7;
  constexpr size_t resultNamesIdx = 8;
  constexpr size_t modelNameAttrIdx = 9;

  constexpr size_t initFnIdx = 0;
  constexpr size_t executeFnIdx = 1;

  StringRef modelName = *desc.getAttr<StringRef>(modelNameAttrIdx);
  StringRef frameworkName = *desc.getAttr<StringRef>(frameworkNameAttrIdx);

  CompiledFrameworkLabel frameworkLabel =
      CompiledFrameworkLabel::getLabelForOpName(getEntityName(), frameworkName);

  MEFFunctionType signature = desc.getAttr<MEFFunctionType>(modelFnTypeAttrIdx);

  auto devicesJson = *desc.getAttr<StringRef>(devicesAttrIdx);
  auto valueOr = llvm::json::parse(devicesJson);
  if (!valueOr)
    return toModularError(valueOr.takeError());

  llvm::SmallVector<DeviceRef> devices(valueOr->getAsArray()->size());
  for (auto [v, dev] : llvm::zip(*valueOr->getAsArray(), devices)) {
    auto deviceOr = DeviceRef::deserializeFromJSON(&v);
    if (deviceOr.isError())
      return deviceOr.takeError();
    dev = deviceOr.takeValue();
  }

  const MEFInvokable *init = desc.getInvokable(initFnIdx);
  const MEFInvokable *execute = desc.getInvokable(executeFnIdx);

  SmallVector<QuantTensorSpec> inputSpecs, outputSpecs;
  if (auto err = decodeQuantizedSignature(signature, inputSpecs, outputSpecs))
    return err.takeError();

  auto getIODevices = [&](const PrimitiveDescriptor &desc,
                          size_t attrIdx) -> llvm::SmallVector<DeviceRef> {
    auto deviceList = desc.getAttr<MEFListAttr>(attrIdx);
    SmallVector<DeviceRef> ioDevices;
    for (MEFDynamicAttr attr : deviceList)
      ioDevices.push_back(devices[attr.cast<MEFAttr<uint32_t>>().get()]);

    return ioDevices;
  };

  auto getNames = [&](const PrimitiveDescriptor &desc,
                      size_t attrIdx) -> llvm::SmallVector<StringRef> {
    auto namesAttr = desc.getAttr<MEFListAttr>(attrIdx);
    SmallVector<StringRef> namesList;
    for (MEFDynamicAttr attr : namesAttr) {
      namesList.push_back(attr.cast<MEFAttr<StringRef>>().get());
    }
    return namesList;
  };

  SmallVector<DeviceRef> inputDevices =
      getIODevices(desc, inputDeviceIndicesIdx);
  SmallVector<DeviceRef> outputDevices =
      getIODevices(desc, resultDeviceIndicesIdx);

  SmallVector<StringRef> inputNames = getNames(desc, argumentNamesIdx);
  SmallVector<StringRef> outputNames = getNames(desc, resultNamesIdx);

  return std::make_unique<Model>(modelName.str(), devices, frameworkLabel, init,
                                 execute, inputSpecs, outputSpecs, inputNames,
                                 outputNames, inputDevices, outputDevices,
                                 desc.copyLocation());
}

//===----------------------------------------------------------------------===//
// ExecutableModel
//===----------------------------------------------------------------------===//

ExecutableModel::ExecutableModel(AnyAsyncValueRef chain,
                                 AnyAsyncValueRef mgpContext,
                                 AnyAsyncValueRef weightsRegistry,
                                 SmallVector<AnyAsyncValueRef> &&extraArgs,
                                 const Model *model, Runtime &runtime)
    : model(model), runtime(runtime), chain(std::move(chain)),
      mgpContext(std::move(mgpContext)),
      weightsRegistry(std::move(weightsRegistry)),
      extraArgs(std::move(extraArgs)) {}

ExecutableModel::~ExecutableModel() = default;

SmallVector<AnyAsyncValueRef>
ExecutableModel::execute(MutableArrayRef<AnyAsyncValueRef> args) const {
  TRACY_ZONE_SCOPED_NC("ExecutableModel::execute", TRACY_COLOR_TEAL);

  checkUniqueRuntime(getRuntime());

  // Pass fixed execute args (chain, context, and weights registry), devices,
  // init->exec context bundles, and inputs to the forward pass.
  // Number of devices and model inputs come from the graph op definition and
  // vary independently in general.
  // See ModelOp in MGPOps.td for a detailed spec of mgp.model.
  SmallVector<AnyAsyncValueRef> passedArgs;
  passedArgs.reserve(kMgpModelNumFixedImplicitExecArgs + extraArgs.size() +
                     args.size());
  passedArgs.emplace_back(chain.copy());
  passedArgs.emplace_back(mgpContext.copy());
  passedArgs.emplace_back(weightsRegistry.copy());
  for (const AnyAsyncValueRef &dev : extraArgs)
    passedArgs.emplace_back(dev.copy());
  for (AnyAsyncValueRef &arg : args)
    passedArgs.emplace_back(std::move(arg));

  // Set the thread local singleton runtime to the provided runtime. This is
  // referenced internally by some kernels.
  AsyncRT::CompactRuntimePtr::setCurrentRuntime(getRuntime());

  return getModel()->getExecuteFn()->invoke(passedArgs, getRuntime());
}

ErrorOrSuccess ExecutableModel::waitForModelSetup() {
  AsyncRT::await({chain.copy(), mgpContext.copy(), weightsRegistry.copy()});
  AsyncRT::await(extraArgs);
  if (chain.isError())
    return Error(chain.takeDiagnostic().getMessage().get());
  if (mgpContext.isError())
    return Error(mgpContext.takeDiagnostic().getMessage().get());
  if (weightsRegistry.isError())
    return Error(weightsRegistry.takeDiagnostic().getMessage().get());
  for (auto &arg : extraArgs) {
    if (arg.isError())
      return Error(arg.takeDiagnostic().getMessage().get());
  }
  return success();
}

// Compute a stable key from input shapes and dtypes using xxhash. This is used
// to look up captured graphs and ensures we only replay graphs for matching
// signatures.
static uint64_t
computeInputKey(ArrayRef<GenericML::Driver::DeviceMemory *> inputs) {
  raw_xxhash64_stream hasher;
  for (const GenericML::Driver::DeviceMemory *memory : inputs) {
    void *bufferAddress = memory->getBuffer().getPointer()->ptr();
    hasher.write(reinterpret_cast<char *>(&bufferAddress),
                 sizeof(bufferAddress));

    const TensorSpec &spec = memory->getSpec();
    uint64_t rank = spec.getRank();
    hasher.write(reinterpret_cast<const char *>(&rank), sizeof(rank));
    for (int64_t dim : spec)
      hasher.write(reinterpret_cast<const char *>(&dim), sizeof(dim));

    uint64_t dtype = static_cast<uint64_t>(spec.getEltType().getValue());
    hasher.write(reinterpret_cast<const char *>(&dtype), sizeof(dtype));
  }
  return hasher.hash();
}

// Return all CUDA streams from the model's device contexts. This keeps
// capture/replay tied to the same runtime devices as execution.
static ErrorOr<SmallVector<Driver::DeviceStreamRef>>
captureStreams(MutableArrayRef<AnyAsyncValueRef> extraArgs) {
  if (auto err = AsyncRT::awaitOrError(extraArgs))
    return err.takeError();

  SmallVector<Driver::DeviceStreamRef> results;
  for (const AnyAsyncValueRef &arg : extraArgs) {
    M::RCRef<const M::Driver::DeviceContext> *context =
        dyn_cast<Driver::DeviceContextRef>(arg);
    if (!context ||
        context->getPointer()->device().api() != Driver::Device::cudaAPI)
      continue;

    ErrorOr<Driver::DeviceStreamRef> streamOr = context->getPointer()->stream();
    if (streamOr.isError())
      return streamOr.takeError();

    results.push_back(streamOr.takeValue());
  }

  return results;
}

ErrorOr<SmallVector<AnyAsyncValueRef>>
ExecutableModel::executeWithDeviceMemoryInputs(
    ArrayRef<GenericML::Driver::DeviceMemory *> inputs) const {
  SmallVector<AnyAsyncValueRef> inputValues;
  inputValues.reserve(inputs.size());
  for (GenericML::Driver::DeviceMemory *memory : inputs) {
    ErrorOr<GenericML::Driver::DeviceMemory *> viewOrErr =
        memory->createView(memory->getSpec());
    if (viewOrErr.isError())
      return viewOrErr.takeError();
    GenericML::Driver::DeviceMemory *view = viewOrErr.takeValue();
    inputValues.push_back(
        GenericML::Driver::prepareInput(getRuntime(), view, view->getSpec()));
  }

  SmallVector<AnyAsyncValueRef> results = execute(inputValues);
  AsyncRT::await(results);
  for (AnyAsyncValueRef &result : results)
    if (result.isError())
      return result.getDiagnostic().getMessage().copy();

  return results;
}

ErrorOr<SmallVector<AnyAsyncValueRef>>
ExecutableModel::capture(ArrayRef<GenericML::Driver::DeviceMemory *> inputs) {
  uint64_t key = computeInputKey(inputs);

  if (auto it = capturedGraphs.find(key); it != capturedGraphs.end()) {
    const Driver::DeviceGraphCollection &graph = it->second;
    if (graph.numGraphs() == 0)
      return Error("no device graphs associated with an existing capture");
    return graph.getOutputs();
  }

  ErrorOr<SmallVector<Driver::DeviceStreamRef>> streamsOr =
      captureStreams(extraArgs);

  if (streamsOr.isError())
    return streamsOr.takeError();

  SmallVector<AnyAsyncValueRef> graphResults;

  ErrorOr<Driver::DeviceGraphCollection> graphOrErr =
      Driver::DeviceGraph::capture(*streamsOr, [&]() -> ErrorOrSuccess {
        auto maybeResult = executeWithDeviceMemoryInputs(inputs);
        if (maybeResult.isError())
          return maybeResult.takeError();

        graphResults = maybeResult.takeValue();
        return success();
      });

  if (graphOrErr.isError())
    return Error(Twine("Failed to capture graph: ") + graphOrErr.getError());

  Driver::DeviceGraphCollection graphs = graphOrErr.takeValue();
  graphs.setOutputs(graphResults);

  capturedGraphs.try_emplace(key, std::move(graphs));
  return graphResults;
}

ErrorOrSuccess
ExecutableModel::replay(ArrayRef<GenericML::Driver::DeviceMemory *> inputs) {
  uint64_t key = computeInputKey(inputs);
  auto it = capturedGraphs.find(key);
  if (it == capturedGraphs.end())
    return Error("No captured graph for these inputs");

  // A forward pass of the model may correspond to multiple graphs (one per
  // device), we launch all graphs in a single replay call.
  const Driver::DeviceGraphCollection &graphs = it->second;
  return graphs.replay();
}

AnyAsyncValueRef
ExecutableModel::convertToSpecificValue(AnyAsyncValueRef genericValue) const {
  return genericValue;
}

AnyAsyncValueRef
ExecutableModel::convertToGenericTensor(AnyAsyncValueRef unknownTensor) const {
  return unknownTensor;
}

AnyAsyncValueRef
ExecutableModel::convertToGenericValue(AnyAsyncValueRef unknownValue) const {
  AnyAsyncValueRef tensor = convertToGenericTensor(std::move(unknownValue));

  if (!tensor.isError())
    return tensor;

  return AnyAsyncValueRef::createError(
      tensor.getRuntime(),
      {"cannot convert unknown value", getModel()->getLocation().copy()});
}

ErrorOrSuccess ExecutableModel::printValue(raw_ostream &os,
                                           AnyAsyncValueRef const &value,
                                           ResultOutputStyle style,
                                           unsigned precision) const {
  return GML::printGenericValue(os, value, style, precision);
}

ErrorOrSuccess
ExecutableModel::processResultForBinaryOutput(const AnyAsyncValueRef &result,
                                              const std::string &outputName,
                                              ProcessedResults &results) const {
  if (auto *tensor = dyn_cast<Tensor>(result)) {
    results.insertResultInfo(
        std::make_shared<TensorResultInfo>(outputName, *tensor));

    const std::string illegalChars = "\\/:?\"<>|";
    std::string sanitizedName = outputName;
    std::replace_if(
        sanitizedName.begin(), sanitizedName.end(),
        [&](char c) { return illegalChars.find(c) != std::string::npos; }, '_');

    results.insertTensor(sanitizedName, tensor);
  } else {
    ErrorOr<AsyncRT::AnyAsyncValueRef> outOrError =
        M::getResultForBinaryOutput(result, results);
    if (outOrError.isError())
      results.insertResultInfo(
          std::make_shared<TensorResultInfo>(outOrError.getError(), ""));
    else
      results.insertResultInfo(std::make_shared<M::VariantResultInfo>(
          outputName, outOrError.takeValue()));
  }
  return success();
}

ErrorOr<ModelList> M::loadMGPMEFFile(RCRef<MEFFile> mefFile) {
  PrimitiveRegistry primitives;
  registerAllGraphRTPrimitives(primitives);
  registerMGPPrimitives(primitives);
  primitives.finalize();

  if (auto err = mefFile->installPrimitives(primitives))
    return err.takeError();

  ModelList models;
  auto *moduleInv = mefFile->getTopLevelInvokable();
  if (auto err = moduleInv->walkPrimitives<Model>(
          [&](auto &&m) { models.push_back(std::forward<decltype(m)>(m)); }))
    return err.takeError();

  return std::move(models);
}
