//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/DeviceDriver/DeviceExecution.h"
#include "AsyncRT/Runtime/Algorithms.h"

using namespace M;
namespace {

// A managed memory block used to transfer device memory to graph execution.
// Once all the uses are complete the graph will call destructor of this and
// free memory.
struct OwnedDeviceMemoryBlock {
  OwnedDeviceMemoryBlock() = delete;
  OwnedDeviceMemoryBlock(Driver::DeviceMemory *memory) : deviceMemory(memory) {}
  ~OwnedDeviceMemoryBlock() { delete deviceMemory; }

  Driver::DeviceMemory *deviceMemory;
};
} // namespace

static M::Tensor getTensorFromDeviceMemory(AsyncRT::Runtime &runtime,
                                           Driver::DeviceMemory *memory,
                                           TensorSpec spec) {
  AnyAsyncValueRef storageRef =
      storageRef.createReady<OwnedDeviceMemoryBlock>(runtime, memory);
  // FIXME: Get alignment of device memory.
  return Tensor(TensorBufferRef::create(memory->data(), memory->bytecount(),
                                        std::move(storageRef),
                                        llvm::MaybeAlign(std::nullopt)),
                TensorSpec{std::move(spec)});
}

static Driver::DeviceMemory *getDeviceMemoryFromTensor(M::ContextRef context,
                                                       Driver::Device *device,
                                                       M::Tensor tensor) {

  // Take ownership of underlying tensor storage.
  TensorBufferRef bufferRef = tensor.releaseBufferRef();
  AnyAsyncValueRef storageRef = bufferRef.getMemStorageHandle();

  AsyncRT::await(storageRef);

  auto destructor = [context = std::move(context),
                     storageRef = storageRef.copy()](
                        void *ptr, const TensorSpec &spec) mutable {
    // When the last reference goes out of scope the data will be deallocated.
    (void)storageRef;

    // When graph deallocates CPU memory it is done through AsyncRT Runtime. So
    // we need to keep runtime alive until the usage of output is complete.
    // During `mojo run` this happens automatically since `mojo` maintains
    // context. However for `mojo build` we need to make sure of that.
    // TODO: Ideally this should be done by graph and not driver.
    (void)context;
  };

  void *outputData = bufferRef.getBuffer();

  // Create a device memory from the pointer.
  return device->createDeviceMemory(outputData, tensor.getSpec(),
                                    std::move(destructor));
}

// Given an AsyncValue that contains a MojoValue convert to a type that
// we can pass to mojo. We assume the outputValue is already awaited.
static Driver::DriverMojoValue
getMojoValueFromAsyncValue(AnyAsyncValueRef outputValue) {
  // Take ownership of underlying mojo value.
  auto mojoValue = std::move(outputValue.get<MojoValue>());
  Driver::DriverMojoValue value;
  value.destructor = mojoValue.getDestructor();
  value.data = mojoValue.take();
  return value;
}

SmallVector<AnyAsyncValueRef>
M::Driver::prepareInputs(Runtime &runtime,
                         M::Driver::DeviceMemory **inputMemories,
                         TensorSpec *inputSpecs, size_t inputCount,
                         DriverMojoValue *mojoValues, size_t mojoValuesCount) {
  SmallVector<AnyAsyncValueRef> inputs;
  inputs.reserve(inputCount + mojoValuesCount);
  for (size_t i = 0; i < inputCount; ++i, ++inputMemories, ++inputSpecs) {
    // Convert inputs from device memory we received from mojo to AsyncValue
    // of tensor as expected by graph. We take ownership of inputs here.
    inputs.push_back(AnyAsyncValueRef::createReady<Tensor>(
        runtime,
        getTensorFromDeviceMemory(runtime, *inputMemories, *inputSpecs)));
  }
  for (size_t i = 0; i < mojoValuesCount; ++i, ++mojoValues) {
    // Take mojo values. Here everything is completely opaque to
    // us. Directly pass this to mgp and we rely on op author to do the
    // right thing with it.
    inputs.push_back(AnyAsyncValueRef::createReady<M::MojoValue>(
        runtime, mojoValues->data, mojoValues->destructor));
  }

  return inputs;
}

Driver::GraphExecutionResultsTy
M::Driver::processResults(llvm::MutableArrayRef<AnyAsyncValueRef> results,
                          ExecutableModel *model, ContextRef maxContext,
                          RCRef<Driver::Device> device) {
  // First result is a chain.
  results = results.drop_front();
  llvm::SmallVector<AnyAsyncValueRef> outputRefs;
  outputRefs.reserve(results.size());
  for (auto &result : results) {
    // Convert only outputs that are of tensor types.
    if (result.isType<Tensor>()) {
      // Each framework has their own way to convert to tensor. So call this
      // function to get the generic form.
      outputRefs.push_back(model->convertToGenericTensor(std::move(result)));
    } else {
      outputRefs.push_back(std::move(result));
    }
  }
  AsyncRT::await(outputRefs);

  Driver::GraphExecutionResultsTy outputs;

  for (auto &&outputRef : outputRefs) {
    if (outputRef.isType<Tensor>()) {
      outputs.push_back(std::move(outputRef.get<Tensor>()));
    } else if (outputRef.isType<MojoValue>()) {
      // Pass through opaque mojo types. We don't have much info here and
      // rely on caller to do the right thing.
      Driver::DriverMojoValue value =
          getMojoValueFromAsyncValue(std::move(outputRef));
      outputs.push_back(value);
    }
  }

  // Return back the output count. This is mostly useful of verification on
  // mojo side.
  return outputs;
}

void M::Driver::passResultsToMojo(
    MutableArrayRef<std::variant<M::Tensor, M::Driver::DriverMojoValue>>
        results,
    decltype(addTensorToOutputList) addTensorToOutputListFn,
    decltype(addValueToOutputList) addValueToOutputListFn, void *mojoContext,
    void *outputList, ContextRef maxContext, RCRef<Driver::Device> device) {
  for (auto &output : results) {
    if (std::holds_alternative<Tensor>(output)) {
      // Convert tensor back to device memory and call the callback we got from
      // mojo. Here we take ownership of output.
      Driver::DeviceMemory *outputMemory =
          getDeviceMemoryFromTensor(std::move(maxContext), device.getPointer(),
                                    std::move(std::get<M::Tensor>(output)));
      auto spec = outputMemory->getSpec();
      addTensorToOutputListFn(mojoContext, outputList, outputMemory, &spec);
    } else {
      // Pass through opaque mojo types. We don't have much info here and
      // rely on caller to do the right thing.
      addValueToOutputListFn(mojoContext, outputList,
                             std::get<DriverMojoValue>(output));
    }
  }
}

size_t M::Driver::processResults(
    llvm::MutableArrayRef<AnyAsyncValueRef> results, M::ExecutableModel *model,
    decltype(addTensorToOutputList) addTensorToOutputListFn,
    decltype(addValueToOutputList) addValueToOutputListFn,
    ContextRef maxContext, RCRef<M::Driver::Device> device, void *mojoContext,
    void *outputList) {

  auto outputsResolved =
      processResults(results, model, maxContext.copy(), device.copy());

  passResultsToMojo(outputsResolved, addTensorToOutputListFn,
                    addValueToOutputListFn, mojoContext, outputList, maxContext,
                    device);

  // Return back the output count. This is mostly useful of verification on
  // mojo side.
  return outputsResolved.size();
}
