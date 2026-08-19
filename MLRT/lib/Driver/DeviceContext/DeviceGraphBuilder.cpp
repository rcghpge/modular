//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the base DeviceGraphBuilder interface.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceGraphBuilder.h"

#include <cassert>
#include <utility>

namespace M::Driver {

DeviceGraphBuilder::DeviceGraphBuilder(DeviceContextRef context,
                                       DeviceGraphMemoryPoolRef pool) {
  assert((!pool || &pool->device() == &context->device()) &&
         "shared device graph memory pool belongs to a different device");
  memoryPool = pool ? std::move(pool)
                    : DeviceGraphMemoryPoolRef::create(std::move(context));
}

DeviceGraphBuilder::~DeviceGraphBuilder() = default;

void DeviceGraphBuilder::addOutput(AsyncRT::AnyAsyncValueRef output) {
  outputs.push_back(std::move(output));
}

void DeviceGraphBuilder::addInput(DeviceBufferRef input) {
  inputs.push_back(std::move(input));
}

ErrorOr<DeviceContextRef>
DeviceGraphBuilder::recordingContext(DeviceGraphNode initialDep) {
  return DeviceContextRef(
      RCRef<RecordingDeviceContext>::create(*this, initialDep));
}

ErrorOr<DeviceBufferRef> DeviceGraphBuilder::createBuffer(size_t numBytes,
                                                          bool isHost) {
  if (!memoryPool)
    return Error("cannot allocate from a device graph builder that has already "
                 "been instantiated");

  ErrorOr<void *> alloc = memoryPool->allocate(numBytes, isHost);
  if (alloc)
    return alloc.takeError();

  void *ptr = alloc.takeValue();
  auto result =
      DeviceBufferRef::take(new DeviceBuffer(context(), ptr, numBytes, isHost));

  // Ensure the pool lives at least as long as the buffer. `isHost` selects the
  // pool the block came from; freeing it to the other one would reach the
  // driver with the wrong host/device flag.
  result->setDestructorFn([pool = memoryPool.copy(), ptr, isHost]() mutable {
    pool->free(ptr, isHost);
    pool.reset();
  });

  return result;
}

// ===== RecordingDeviceContext ===== //

RecordingDeviceContext::RecordingDeviceContext(DeviceGraphBuilder &builder,
                                               DeviceGraphNode initialDep)
    : DeviceContext(builder.context().device()),
      builder_(DeviceGraphBuilderRef::copy(&builder)),
      // Recorded operations chain after `initialDep`, which the Mojo layer
      // seeds with the ambient/region dependencies (this builder does not
      // model dependencies itself).
      lastNode_(initialDep) {}

DeviceContext &RecordingDeviceContext::backendContext() {
  return builder_->context();
}

const DeviceContext &RecordingDeviceContext::backendContext() const {
  return builder_->context();
}

RecordingDeviceContext::~RecordingDeviceContext() = default;

llvm::SmallVector<DeviceGraphNode, 1>
RecordingDeviceContext::currentDeps() const {
  llvm::SmallVector<DeviceGraphNode, 1> deps;
  deps.push_back(lastNode_);
  return deps;
}

ErrorOrSuccess RecordingDeviceContext::chain(ErrorOr<DeviceGraphNode> node) {
  if (node.isError())
    return node.takeError();
  lastNode_ = node.takeValue();
  return success();
}

// ----- Recorded ----- //

ErrorOrSuccess RecordingDeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args, uint32_t argCount,
    const size_t * /*argSizes*/) {
  // `argSizes` is dropped: it is not part of `DeviceFunctionLaunchParams` and
  // the graph node path does not use it (matching
  // AsyncRT_DeviceGraphBuilder_addFunctionDirect).
  DeviceFunctionLaunchParams params;
  params.gridX = gridX;
  params.gridY = gridY;
  params.gridZ = gridZ;
  params.blockX = blockX;
  params.blockY = blockY;
  params.blockZ = blockZ;
  params.sharedMemBytes = sharedMemBytes;
  params.attrs = attrs;
  params.numAttrs = numAttrs;
  params.args = args;
  params.argCount = argCount;
  return chain(builder_->addFunctionDirect(func, params, currentDeps()));
}

ErrorOrSuccess RecordingDeviceContext::enqueueCopyToDevice(void *dst,
                                                           const void *src,
                                                           size_t byteSize) {
  return chain(
      builder_->addCopyHostToDevice(dst, src, byteSize, currentDeps()));
}

ErrorOrSuccess RecordingDeviceContext::enqueueCopyFromDevice(void *dst,
                                                             const void *src,
                                                             size_t byteSize) {
  return chain(
      builder_->addCopyDeviceToHost(dst, src, byteSize, currentDeps()));
}

ErrorOrSuccess RecordingDeviceContext::enqueueSetMemory(void *dst,
                                                        size_t byteSize,
                                                        uint64_t val,
                                                        size_t valSize) {
  return chain(
      builder_->addSetMemory(dst, byteSize, val, valSize, currentDeps()));
}

ErrorOrSuccess RecordingDeviceContext::memCopyDtoD(DeviceBuffer &dst,
                                                   DeviceBuffer &src,
                                                   size_t srcByteOffset,
                                                   size_t srcByteSize,
                                                   bool /*crossStreamSync*/) {
  // Cross-stream sync is irrelevant here: ordering is expressed by the node's
  // dependency edge, not by stream semaphores.
  return chain(builder_->addCopyDeviceToDevice(dst, src, srcByteOffset,
                                               srcByteSize, currentDeps()));
}

// ----- Buffer creation: delegated to the builder's memory pool ----- //

ErrorOr<DeviceBufferRef>
RecordingDeviceContext::enqueueCreateBuffer(size_t byteSize) {
  return builder_->createBuffer(byteSize, /*isHost=*/false);
}

ErrorOr<DeviceBufferRef>
RecordingDeviceContext::createHostBuffer(size_t byteSize) {
  // Recorded nodes bake in the host address just as they do a device one, so
  // these go through the pool too. Note this trades the backing context's
  // event-ordered host free for an immediate return to the pool: the block
  // stays reserved for the graph, but a later allocation from the same pool can
  // reuse the address, so it is only safe while recording order matches the
  // graph's execution order.
  return builder_->createBuffer(byteSize, /*isHost=*/true);
}

// ----- Forwarded to the backing context ----- //

ErrorOr<DeviceFunctionRef> RecordingDeviceContext::loadFunction(
    StringRef moduleName, StringRef functionName, StringRef code,
    int32_t maxDynamicSharedBytes, StringRef debugLevel,
    int32_t optimizationLevel) {
  return builder_->context().loadFunction(moduleName, functionName, code,
                                          maxDynamicSharedBytes, debugLevel,
                                          optimizationLevel);
}

ErrorOr<DeviceFunctionRef> RecordingDeviceContext::loadFunctionWithDebugName(
    StringRef moduleName, StringRef functionName, StringRef code,
    int32_t maxDynamicSharedBytes, StringRef debugLevel,
    int32_t optimizationLevel, StringRef debugName) {
  return builder_->context().loadFunctionWithDebugName(
      moduleName, functionName, code, maxDynamicSharedBytes, debugLevel,
      optimizationLevel, debugName);
}

ErrorOr<int> RecordingDeviceContext::getAttribute(Attribute attr) const {
  return builder_->context().getAttribute(attr);
}

ErrorOr<DeviceStreamRef> RecordingDeviceContext::stream() const {
  return builder_->context().stream();
}

ErrorOr<Device::MemInfo> RecordingDeviceContext::getDeviceMemoryInfo() const {
  return builder_->context().getMemoryInfo();
}

ErrorOrSuccess RecordingDeviceContext::memAlloc(void **ptr, size_t byteSize,
                                                Device::AllocMode mode) {
  return builder_->context().memAlloc(ptr, byteSize, mode);
}

ErrorOrSuccess RecordingDeviceContext::memFree(void *ptr, bool isHost) {
  return builder_->context().memFree(ptr, isHost);
}

RecordingDeviceContext::MMapFileImplResult
RecordingDeviceContext::memmapFileImpl(
    const std::filesystem::path &filename) const {
  return builder_->context().memmapFileImpl(filename);
}

void *RecordingDeviceContext::driverContext() const {
  return builder_->context().driverContext();
}

// ----- Hard-error (non-recordable host-visible waits) ----- //

namespace {
Error notRecordable(const char *op) {
  return Error(std::string(op) +
               " cannot be recorded into a device graph: it observes a "
               "device-side result on the host, which does not exist until "
               "replay. Hoist this operation out of the recorded region.");
}
} // namespace

ErrorOrSuccess RecordingDeviceContext::synchronize() {
  return notRecordable("RecordingDeviceContext::synchronize");
}

ErrorOr<DeviceEventRef> RecordingDeviceContext::enqueueEvent() {
  return notRecordable("RecordingDeviceContext::enqueueEvent");
}

ErrorOrSuccess RecordingDeviceContext::enqueueEvent(const DeviceEvent &) {
  return notRecordable("RecordingDeviceContext::enqueueEvent");
}

ErrorOrSuccess RecordingDeviceContext::enqueueWaitFor(DeviceEvent &) {
  return notRecordable("RecordingDeviceContext::enqueueWaitFor");
}

ErrorOrSuccess RecordingDeviceContext::enqueueWaitFor(DeviceContext &) {
  return notRecordable("RecordingDeviceContext::enqueueWaitFor");
}

ErrorOr<DeviceTimerRef> RecordingDeviceContext::startTimer() {
  return notRecordable("RecordingDeviceContext::startTimer");
}

ErrorOr<std::chrono::nanoseconds>
RecordingDeviceContext::stopTimer(const DeviceTimer &) {
  return notRecordable("RecordingDeviceContext::stopTimer");
}

} // namespace M::Driver
