//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/DeviceDriver/DeviceExecution.h"
#include "LLCL/Runtime/Algorithms.h"

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

static M::Tensor getTensorFromDeviceMemory(LLCL::Runtime &runtime,
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

static Driver::DeviceMemory *
getDeviceMemoryFromAsyncTensor(M::ContextRef context, Driver::Device *device,
                               AnyAsyncValueRef outputTensor) {

  // Take ownership of underlying tensor storage.
  Tensor &tensor = outputTensor.get<Tensor>();
  TensorBufferRef bufferRef = tensor.releaseBufferRef();
  AnyAsyncValueRef storageRef = bufferRef.getMemStorageHandle();

  LLCL::await(storageRef);

  auto destructor = [context = std::move(context),
                     storageRef = storageRef.copy()](
                        void *ptr, const TensorSpec &spec) mutable {
    // When the last reference goes out of scope the data will be deallocated.
    (void)storageRef;

    // When graph deallocates CPU memory it is done through LLCL Runtime. So
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

SmallVector<AnyAsyncValueRef>
M::Driver::prepareInputs(Runtime &runtime,
                         M::Driver::DeviceMemory **inputMemories,
                         TensorSpec *inputSpecs, size_t inputCount) {
  SmallVector<AnyAsyncValueRef> inputs;
  inputs.reserve(inputCount);
  for (size_t i = 0; i < inputCount; ++i, ++inputMemories, ++inputSpecs) {
    // Convert inputs from device memory we received from mojo to AsyncValue
    // of tensor as expected by graph. We take ownership of inputs here.
    inputs.push_back(AnyAsyncValueRef::createReady<Tensor>(
        runtime,
        getTensorFromDeviceMemory(runtime, *inputMemories, *inputSpecs)));
  }

  return inputs;
}

size_t M::Driver::processResults(
    llvm::MutableArrayRef<AnyAsyncValueRef> results, M::ExecutableModel *model,
    decltype(M::Driver::addToOutputList) addToOutputListFn,
    const ContextRef &maxContext, const RCRef<M::Driver::Device> &device,
    void *mojoContext, void *outputList) {
  // First result is a chain.
  results = results.drop_front();
  llvm::SmallVector<AnyAsyncValueRef> outputTensorRefs;
  outputTensorRefs.reserve(results.size());
  for (auto &result : results) {
    // Each framework has their own way to convert to tensor. So call this
    // function to get the generic form.
    outputTensorRefs.push_back(
        model->convertToGenericTensor(std::move(result)));
  }
  LLCL::await(outputTensorRefs);
  for (auto &&outputTensorRef : outputTensorRefs) {
    // Convert tensor back to device memory and call the callback we got from
    // mojo. Here we take ownership of output.
    Driver::DeviceMemory *outputMemory = getDeviceMemoryFromAsyncTensor(
        maxContext, device.getPointer(), std::move(outputTensorRef));
    const TensorSpec &spec = outputMemory->getSpec();
    addToOutputListFn(mojoContext, outputList, outputMemory,
                      const_cast<TensorSpec *>(&spec));
  }

  // Return back the output count. This is mostly useful of verification on
  // mojo side.
  return outputTensorRefs.size();
}
