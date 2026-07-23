//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Driver/Device/DeviceExecution.h"
#include "AsyncRT/Runtime/Algorithms.h"

using M::AnyAsyncValueRef;
using M::ErrorOr;
using M::GraphCompiler::Driver::Device;
using M::GraphCompiler::Driver::DeviceMemory;
using M::MLRT::Tensor;
using M::MLRT::TensorBufferRef;
namespace {

// A managed memory block used to transfer device memory to graph execution.
// Once all the uses are complete the graph will call destructor of this and
// free memory.
struct OwnedDeviceMemoryBlock {
  OwnedDeviceMemoryBlock() = delete;
  OwnedDeviceMemoryBlock(DeviceMemory *memory) : deviceMemory(memory) {}
  ~OwnedDeviceMemoryBlock() { delete deviceMemory; }

  DeviceMemory *deviceMemory;
};
} // namespace

// Construct on-device memory to hold a provided tensor.
ErrorOr<DeviceMemory *>
M::GraphCompiler::Driver::constructDeviceMemoryFromTensor(M::ContextRef context,
                                                          Device *device,
                                                          Tensor tensor) {

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

    ErrorOr<DeviceMemory *> memoryOr =
        device->createDeviceMemory(tensor.getSpec());
    if (memoryOr.isError())
      return memoryOr.takeError();
    DeviceMemory *memory = memoryOr.get();

    void *tensorBuffer = bufferRef.getBuffer();
    std::memcpy(memory->data(), tensorBuffer,
                tensor.getSpec().getSizeInBytes());

    return memory;
  }

  AsyncRT::await(storageRef);

  auto destructor = [context = std::move(context),
                     storageRef = storageRef.copy()]() mutable {
    // When the last reference goes out of scope, the data will be deallocated.
    storageRef.reset();

    // When graph deallocates CPU memory it is done through AsyncRT Runtime. So
    // we need to keep cpuDevice alive until the usage of output is complete.
    // During `mojo run` this happens automatically since `mojo` maintains
    // context. However for `mojo build` we need to make sure of that.
    // TODO: Ideally this should be done by graph and not driver.
    context.reset();
  };

  void *outputData = bufferRef.getBuffer();

  // Create a device memory from the pointer.
  return device->createDeviceMemory(outputData, tensor.getSpec(),
                                    std::move(destructor));
}

AnyAsyncValueRef M::GraphCompiler::Driver::prepareInput(
    CPUDevice &cpuDevice, DeviceMemory *inputMemory, TensorSpec inputSpec) {
  // Convert inputs from device memory we received to AsyncValue of tensor as
  // expected by graph. We take ownership of inputs here.
  AnyAsyncValueRef storageRef =
      storageRef.createReady<OwnedDeviceMemoryBlock>(cpuDevice, inputMemory);
  // FIXME: Get alignment of device memory.
  Tensor tensor(TensorBufferRef::create(
                    inputMemory->data(), inputMemory->bytecount(),
                    std::move(storageRef), llvm::MaybeAlign(std::nullopt)),
                std::move(inputSpec));

  return AnyAsyncValueRef::createReady<Tensor>(cpuDevice, std::move(tensor));
}
