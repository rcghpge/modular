//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Context.h"
#include "MLRT/Driver/HAL/Buffer.h"
#include "MLRT/Driver/HAL/Driver.h"
#include "MLRT/Driver/HAL/Event.h"
#include "MLRT/Driver/HAL/Queue.h"

namespace M::Driver::HAL {

Context::~Context() {
  // Functions must be unloaded while the context is still valid.
  funcCache.clear();
  device_->driver->plugin.context_destroy(handle);
}

ErrorOr<BufferRef> Context::allocBuffer(size_t byteSize,
                                        Buffer::AllocMode mode) {
  void *memory;
  if (M_driver_status_t status = device_->driver->plugin.memory_alloc(
          handle, static_cast<M_driver_alloc_mode_t>(mode), byteSize, &memory))
    return Error(device_->driver->getStatusMessage(status));
  Buffer *buffer = new (std::nothrow) Buffer(this, memory);
  if (!buffer)
    return Error("Failed to allocate memory for Buffer object");
  return BufferRef::take(buffer);
}

ErrorOr<QueueRef> Context::createQueue() {
  M_driver_queue_t queueHandle = nullptr;
  if (M_driver_status_t status =
          device_->driver->plugin.queue_create(handle, &queueHandle))
    return Error(device_->driver->getStatusMessage(status));
  std::unique_ptr<Queue> queue(new (std::nothrow) Queue(this, queueHandle));
  if (!queue)
    return Error("Failed to allocate memory for Queue object");
  return QueueRef::take(queue.release());
}

ErrorOr<EventRef> Context::createEvent(bool enableTiming) {
  M_driver_event_t eventHandle = nullptr;
  if (M_driver_status_t status = device_->driver->plugin.event_create(
          handle, &eventHandle, enableTiming))
    return Error(device_->driver->getStatusMessage(status));
  std::unique_ptr<Event> event(new (std::nothrow)
                                   Event(this, eventHandle, enableTiming));
  if (!event)
    return Error("Failed to allocate memory for Event object");
  return EventRef::take(event.release());
}

ErrorOr<float> Context::eventElapsedTime(Event &start, Event &end) {
  if (!start.enableTiming || !end.enableTiming)
    return Error("Both events must be created with timing enabled to "
                 "compute elapsed time");
  float elapsed_ms = 0.0f;
  if (M_driver_status_t status = device_->driver->plugin.event_elapsed_time(
          handle, start.handle, end.handle, &elapsed_ms))
    return Error(device_->driver->getStatusMessage(status));
  return elapsed_ms;
}

ErrorOr<FunctionRef>
Context::loadFunction(llvm::StringRef moduleName, llvm::StringRef functionName,
                      llvm::StringRef code, int32_t maxDynamicSharedBytes,
                      llvm::StringRef debugLevel, int32_t optimizationLevel) {
  FunctionCacheKey key{moduleName.empty() ? code.str() : moduleName.str(),
                       debugLevel.str(), maxDynamicSharedBytes,
                       optimizationLevel};

  {
    std::shared_lock<std::shared_mutex> lock(funcCacheMutex);
    if (FunctionRef cached = funcCache.lookup(key))
      return cached;
  }

  M_driver_function_t funcHandle = nullptr;
  if (M_driver_status_t status = device_->driver->plugin.function_load(
          device_->driver->handle, handle, code.data(), code.size(),
          functionName.data(), maxDynamicSharedBytes,
          debugLevel.empty() ? nullptr : debugLevel.data(), optimizationLevel,
          &funcHandle))
    return Error(device_->driver->getStatusMessage(status));

  FunctionRef ref = FunctionRef::take(new Function(this, funcHandle));
  {
    std::unique_lock<std::shared_mutex> lock(funcCacheMutex);
    funcCache[std::move(key)] = ref.copy();
  }
  return ref;
}

} // namespace M::Driver::HAL
