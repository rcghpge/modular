//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Driver/Device/DeviceExecution.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "GraphCompiler/Driver/Device/DebugAllocPoison.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using M::AnyAsyncValueRef;
using M::ErrorOr;
using M::Driver::DeviceBufferRef;
using M::Driver::DeviceContextRef;
using M::MLRT::Tensor;
using M::MLRT::TensorBufferRef;

ErrorOr<DeviceBufferRef> M::GraphCompiler::Driver::allocateDeviceBuffer(
    const DeviceContextRef &device, const M::TensorSpec &spec, bool hostAlloc) {
  ErrorOr<DeviceBufferRef> bufOrErr =
      hostAlloc ? device->createHostBuffer(spec.getSizeInBytes())
                : device->enqueueCreateBuffer(spec.getSizeInBytes());
  if (bufOrErr.isError())
    return bufOrErr.takeError();
  DeviceBufferRef buf = bufOrErr.takeValue();

  // Debug allocation poisoning: fills newly allocated memory with default
  // patterns to help detect use of uninitialized memory. Host and device
  // fills use the same per-type pattern (see DebugAllocPoison.h) so the Mojo
  // stdlib uninit-read check has a single value to look for.
  // Enable via: MODULAR_DEBUG_DEVICE_ALLOCATOR=uninitialized-poison
  if (device->isDebugAllocatorEnabled("uninitialized-poison")) {
    if (hostAlloc) {
      M::GraphCompiler::Driver::poisonUninitializedMemory(buf->ptr(), spec);
    } else if (std::optional<M::GraphCompiler::Driver::PoisonPattern> pattern =
                   M::GraphCompiler::Driver::getPoisonPattern(spec)) {
      llvm::errs() << "DEBUG_ALLOC_POISON: Filling device " << spec.getEltType()
                   << " allocation (" << spec.getSizeInBytes()
                   << " bytes) with " << pattern->patternKind << " pattern ";
      (llvm::errs().changeColor(llvm::raw_ostream::RED)
       << "0x"
       << llvm::format_hex_no_prefix(pattern->bits, pattern->patternBytes * 2))
          .resetColor();
      llvm::errs() << "\n";

      if (M::ErrorOrSuccess err = device->enqueueSetMemory(
              *buf, pattern->bits, pattern->patternBytes))
        return err.takeError();
      if (M::ErrorOrSuccess err = device->synchronize())
        return err.takeError();
    }
  }

  return buf;
}

M::Driver::DeviceBufferRef
M::GraphCompiler::Driver::borrowBuffer(const DeviceContextRef &device,
                                       void *ptr, size_t byteSize) {
  return device->createBufferOwning(ptr, byteSize, /*owning=*/false);
}

// Construct an on-device buffer to hold a provided tensor.
ErrorOr<DeviceBufferRef>
M::GraphCompiler::Driver::constructDeviceMemoryFromTensor(
    M::ContextRef context, DeviceContextRef device, Tensor tensor) {

  // Take ownership of underlying tensor storage.
  TensorBufferRef bufferRef = tensor.releaseBufferRef();
  AnyAsyncValueRef storageRef = bufferRef.getMemStorageHandle();

  // We can have situations where the graph is attempting to return a value that
  // will not outlive it. In these cases, we need to explicitly copy this value
  // to a device buffer.
  if (!storageRef) {
    // If we hit a situation where we're returning a null storage ref on a GPU,
    // we should just error out, since this situation should be impossible.
    if (device->device().api() != M::Driver::Device::cpuAPI)
      return Error("Execution returned a null result on a GPU.");

    ErrorOr<DeviceBufferRef> bufferOr =
        allocateDeviceBuffer(device, tensor.getSpec());
    if (bufferOr.isError())
      return bufferOr.takeError();
    DeviceBufferRef buffer = bufferOr.takeValue();

    void *tensorBuffer = bufferRef.getBuffer();
    std::memcpy(buffer->ptr(), tensorBuffer, tensor.getSpec().getSizeInBytes());

    return buffer;
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

  // Create a device buffer from the pointer.
  DeviceBufferRef buffer = device->createBufferOwning(
      outputData, tensor.getSpec().getSizeInBytes(), /*owning=*/true);
  buffer->setDestructorFn(std::move(destructor));
  return buffer;
}

AnyAsyncValueRef M::GraphCompiler::Driver::prepareInput(
    CPUDevice &cpuDevice, DeviceBufferRef inputBuffer, TensorSpec inputSpec) {
  // Convert inputs from the device buffer we received to AsyncValue of tensor
  // as expected by graph. Wrapping the refcounted buffer itself as the
  // storage handle (rather than a manually-`delete`d wrapper) means it stays
  // alive for exactly as long as this AsyncValueRef needs it.
  void *ptr = inputBuffer->ptr();
  size_t byteCount = inputBuffer->byteSize();
  AnyAsyncValueRef storageRef = AnyAsyncValueRef::createReady<DeviceBufferRef>(
      cpuDevice, std::move(inputBuffer));
  // FIXME: Get alignment of device memory.
  Tensor tensor(TensorBufferRef::create(ptr, byteCount, std::move(storageRef),
                                        llvm::MaybeAlign(std::nullopt)),
                std::move(inputSpec));

  return AnyAsyncValueRef::createReady<Tensor>(cpuDevice, std::move(tensor));
}
