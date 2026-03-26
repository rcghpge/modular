//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Context.h"
#include "HAL/M_driver_bundle.h"
#include "HAL/M_driver_common.h"
#include "MLRT/Driver/HAL/Buffer.h"
#include "MLRT/Driver/HAL/Bundle.h"
#include "MLRT/Driver/HAL/Driver.h"
#include "MLRT/Driver/HAL/Event.h"
#include "MLRT/Driver/HAL/Queue.h"
#include <mutex>

namespace M::Driver::HAL {

/// NOTE: assumes `s` is null terminated
M_driver_slice M_driver_slice_from(const char *s) {
  uint64_t size = strlen(s);
  return M_driver_slice{s, size};
}

M_driver_slice M_driver_slice_from(std::string &s) {
  return M_driver_slice{s.data(), s.size()};
}

M_driver_slice M_driver_slice_from(llvm::StringRef &s) {
  return M_driver_slice{s.data(), s.size()};
}

Context::Context(DeviceRef device, M_driver_context_t handle)
    : device(std::move(device)), handle(handle) {}

Context::~Context() {
  // Functions must be unloaded while the context is still valid.
  funcCache.clear();
  device->driver->plugin.context_destroy(handle);
}

DeviceRef Context::getDevice() const { return device.copy(); }

ErrorOr<BufferRef> Context::allocBufferSync(size_t byteSize) {
  M_driver_memory_t memory;
  if (M_driver_status_t status =
          device->driver->plugin.memory_alloc_sync(handle, byteSize, &memory))
    return Error(device->driver->getStatusMessage(status));
  Buffer *buffer = new (std::nothrow)
      Buffer(copyRCRef(this), Buffer::AllocMode::kSync, memory, byteSize);
  if (!buffer)
    return Error("Failed to allocate memory for Buffer object");
  return BufferRef::take(buffer);
}

ErrorOr<BufferRef> Context::allocBufferPinned(size_t byteSize) {
  M_driver_memory_t memory;
  if (M_driver_status_t status =
          device->driver->plugin.memory_alloc_pinned(handle, byteSize, &memory))
    return Error(device->driver->getStatusMessage(status));
  Buffer *buffer = new (std::nothrow)
      Buffer(copyRCRef(this), Buffer::AllocMode::kPinned, memory, byteSize);
  if (!buffer)
    return Error("Failed to allocate memory for Buffer object");
  return BufferRef::take(buffer);
}

ErrorOr<QueueRef> Context::createQueue() {
  M_driver_queue_t queueHandle = nullptr;
  if (M_driver_status_t status =
          device->driver->plugin.queue_create(handle, &queueHandle))
    return Error(device->driver->getStatusMessage(status));
  std::unique_ptr<Queue> queue(new (std::nothrow)
                                   Queue(copyRCRef(this), queueHandle));
  if (!queue)
    return Error("Failed to allocate memory for Queue object");
  return QueueRef::take(queue.release());
}

ErrorOr<EventRef> Context::createEvent() {
  M_driver_event_t eventHandle = nullptr;
  if (M_driver_status_t status =
          device->driver->plugin.event_create(handle, &eventHandle))
    return Error(device->driver->getStatusMessage(status));
  std::unique_ptr<Event> event(new (std::nothrow)
                                   Event(copyRCRef(this), eventHandle));
  if (!event)
    return Error("Failed to allocate memory for Event object");
  return EventRef::take(event.release());
}

ErrorOr<BundleRef> Context::loadBundle(llvm::StringRef bundleName,
                                       llvm::StringRef code,
                                       llvm::StringRef codeType,
                                       llvm::StringRef debugLevel,
                                       int32_t optimizationLevel) {

  // TODO(Sawyer/Konrad): REVISIT: is keying on the text of the module
  // reasonable?
  BundleCacheKey key{bundleName.empty() ? code.str() : bundleName.str(),
                     debugLevel.str(), optimizationLevel};
  {
    std::shared_lock<std::shared_mutex> lock(bundleCacheMutex);
    if (BundleRef cached = bundleCache.lookup(key)) {
      return cached;
    }
  }

  M_driver_runtime_bundle_t bundleHandle = nullptr;
  M_driver_static_bundle staticBundle{.mapped_data = M_driver_slice_from(code),
                                      .file_type = codeType.data(),
                                      .file_type_len = codeType.size()};

  M_driver_bundle_compilation_options opts{
      .debug_level = debugLevel.data(),
      .debug_level_len = debugLevel.size(),
      .optimization_level = optimizationLevel,
  };

  M_driver_status_t status = device->driver->plugin.bundle_load(
      handle, &staticBundle, &opts, &bundleHandle);

  if (status) {
    return Error(device->driver->getStatusMessage(status));
  }

  BundleRef ref = BundleRef::take(new Bundle(this, bundleHandle));
  {
    std::unique_lock<std::shared_mutex> lock(bundleCacheMutex);
    bundleCache[std::move(key)] = ref.copy();
  }

  return ref;
}

ErrorOr<FunctionRef> Context::loadFunction(BundleRef bundle,
                                           llvm::StringRef functionName) {
  FuncCacheKey key{.bundle = bundle->handle, .name = std::string(functionName)};

  {
    std::shared_lock<std::shared_mutex> lock(funcCacheMutex);
    if (FunctionRef cached = funcCache.lookup(key))
      return cached;
  }

  M_driver_function_t funcHandle = nullptr;
  if (M_driver_status_t status = device->driver->plugin.function_load(
          handle, bundle->handle, functionName.data(), functionName.size(),
          &funcHandle))
    return Error(device->driver->getStatusMessage(status));

  FunctionRef ref =
      FunctionRef::take(new Function(copyRCRef(this), funcHandle));
  {
    std::unique_lock<std::shared_mutex> lock(funcCacheMutex);
    funcCache[std::move(key)] = ref.copy();
  }
  return ref;
}

} // namespace M::Driver::HAL
