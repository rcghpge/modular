//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/DeviceDriver/DeviceExecution.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "Runtime/MojoValue.h"
#include "Support/AssertStream.h"

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

// Construct on-device memory to hold a provided tensor.
ErrorOr<Driver::DeviceMemory *> M::Driver::constructDeviceMemoryFromTensor(
    M::ContextRef context, Driver::Device *device, M::Tensor tensor) {

  // Take ownership of underlying tensor storage.
  TensorBufferRef bufferRef = tensor.releaseBufferRef();
  AnyAsyncValueRef storageRef = bufferRef.getMemStorageHandle();

  // We can have situations where the graph is attempting to return a value that
  // will not outlive it. In these cases, we need to explicitly copy this value
  // to a DeviceMemory object.
  if (!storageRef) {
    // If we hit a situation where we're returning a null storage ref on a GPU,
    // we should just error out, since this situation should be impossible.
    if (!device->isHost())
      return Error("Execution returned a null result on a GPU.");

    ErrorOr<Driver::DeviceMemory *> memoryOr =
        device->createDeviceMemory(tensor.getSpec());
    if (memoryOr.isError())
      return memoryOr.takeError();
    Driver::DeviceMemory *memory = memoryOr.get();

    void *tensorBuffer = bufferRef.getBuffer();
    std::memcpy(memory->data(), tensorBuffer,
                tensor.getSpec().getSizeInBytes());

    return memory;
  }

  AsyncRT::await(storageRef);

  auto destructor = [context = std::move(context),
                     storageRef = storageRef.copy()](
                        void *ptr, const TensorSpec &spec) mutable {
    // When the last reference goes out of scope, the data will be deallocated.
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
  ASSERT_STREAM(mojoValue.isOwned() && !mojoValue.destructorWillFree(),
                "MojoValue must be owned and the destructor should not free");
  value.destructor = mojoValue.getDestructor();
  value.data = mojoValue.take();
  // TODO: this is unsafe, need to copy owning too.
  return value;
}

AnyAsyncValueRef M::Driver::prepareInput(Runtime &runtime,
                                         M::Driver::DeviceMemory *inputMemory,
                                         TensorSpec inputSpec) {
  // Convert inputs from device memory we received to AsyncValue of tensor as
  // expected by graph. We take ownership of inputs here.
  AnyAsyncValueRef storageRef =
      storageRef.createReady<OwnedDeviceMemoryBlock>(runtime, inputMemory);
  // FIXME: Get alignment of device memory.
  Tensor tensor(TensorBufferRef::create(
                    inputMemory->data(), inputMemory->bytecount(),
                    std::move(storageRef), llvm::MaybeAlign(std::nullopt)),
                std::move(inputSpec));

  return AnyAsyncValueRef::createReady<Tensor>(runtime, std::move(tensor));
}

Driver::GraphExecutionResultsTy
M::Driver::processResults(llvm::MutableArrayRef<AnyAsyncValueRef> results,
                          ExecutableModel *model) {
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

  return outputs;
}

ErrorOrSuccess M::Driver::passResultsToMojo(
    MutableArrayRef<std::variant<M::Tensor, M::Driver::DriverMojoValue>>
        results,
    decltype(addTensorToOutputList) addTensorToOutputListFn,
    decltype(addValueToOutputList) addValueToOutputListFn, void *mojoContext,
    void *outputList, ContextRef maxContext,
    ArrayRef<Driver::Device *> resultDevices) {
  // We only count non-opaque or Tensor values as a result.
  size_t resultDeviceIndex = 0;
  for (auto &output : results) {
    if (M::Tensor *tensor = std::get_if<M::Tensor>(&output)) {
      auto device = resultDevices[resultDeviceIndex];
      // Convert tensor back to device memory and call the callback we got from
      // mojo. Here we take ownership of output.
      ErrorOr<Driver::DeviceMemory *> outputMemoryOr =
          M::Driver::constructDeviceMemoryFromTensor(
              std::move(maxContext), device, std::move(*tensor));

      if (outputMemoryOr.isError())
        return outputMemoryOr.takeError();

      Driver::DeviceMemory *outputMemory = outputMemoryOr.get();
      auto spec = outputMemory->getSpec();
      addTensorToOutputListFn(mojoContext, outputList, outputMemory, &spec);
      resultDeviceIndex++;
    } else {
      // Pass through opaque mojo types. We don't have much info here and
      // rely on caller to do the right thing.
      addValueToOutputListFn(mojoContext, outputList,
                             std::get<DriverMojoValue>(output));
    }
  }
  return success();
}
