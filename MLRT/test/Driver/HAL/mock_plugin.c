/*===----------------------------------------------------------------------===*\
|*                                                                            *|
|* This file is Modular Inc proprietary.                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Minimal mock HAL plugin for testing the Mojo HAL bindings without GPU
/// hardware. Implements all M_driver_* symbols expected by the Mojo Plugin
/// struct with deterministic stub behaviour:
///
///   - 2 mock devices
///   - status_message returns standard strings for all predefined codes
///   - memory alloc returns malloc'd blocks; free calls free()
///   - queue/event/bundle/function operations succeed silently

#include "HAL/M_driver_api.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Sentinel objects used as opaque handles.
// NOTE: global state — not safe for concurrent test processes sharing
// the same .so instance.
// ---------------------------------------------------------------------------
static int g_driver_sentinel;
static int g_device_sentinels[2];
static int g_context_sentinel;
static int g_queue_sentinel;
static int g_event_sentinel;
static int g_bundle_sentinel;
static int g_function_sentinel;

#define MOCK_DEVICE_COUNT 2

// ---------------------------------------------------------------------------
// status_message
// ---------------------------------------------------------------------------
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_status_message(M_driver_driver_t driver,
                                          M_driver_status_t status,
                                          char *message_buffer,
                                          int64_t message_buffer_size) {
  (void)driver;
  if (!message_buffer || message_buffer_size <= 0)
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;

  const char *msg = NULL;
  switch (status) {
  case M_DRIVER_STATUS_SUCCESS:
    msg = "success";
    break;
  case M_DRIVER_STATUS_INCOMPATIBLE_VERSION:
    msg = "incompatible version";
    break;
  case M_DRIVER_STATUS_UNKNOWN_ERROR:
    msg = "unknown error";
    break;
  case M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE:
    msg = "argument out of range";
    break;
  case M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME:
    msg = "unknown property name";
    break;
  case M_DRIVER_STATUS_NO_DEVICES:
    msg = "no devices";
    break;
  case M_DRIVER_STATUS_BAD_DEVICE:
    msg = "bad device";
    break;
  case M_DRIVER_STATUS_BAD_CONTEXT:
    msg = "bad context";
    break;
  case M_DRIVER_STATUS_BAD_MEMORY:
    msg = "bad memory";
    break;
  case M_DRIVER_STATUS_BAD_QUEUE:
    msg = "bad queue";
    break;
  case M_DRIVER_STATUS_BAD_EVENT:
    msg = "bad event";
    break;
  case M_DRIVER_STATUS_BAD_FUNCTION:
    msg = "bad function";
    break;
  case M_DRIVER_STATUS_OUT_OF_MEMORY:
    msg = "out of memory";
    break;
  case M_DRIVER_STATUS_INVALID_ARG:
    msg = "invalid argument";
    break;
  case M_DRIVER_STATUS_INCOMPATIBLE:
    msg = "incompatible";
    break;
  case M_DRIVER_STATUS_BAD_DRIVER:
    msg = "bad driver";
    break;
  case M_DRIVER_STATUS_BAD_BUNDLE:
    msg = "bad bundle";
    break;
  case M_DRIVER_STATUS_EXECUTE_MODE_NOT_SUPPORTED:
    msg = "execute mode not supported";
    break;
  default:
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;
  }

  strncpy(message_buffer, msg, (size_t)message_buffer_size - 1);
  message_buffer[message_buffer_size - 1] = '\0';
  return M_DRIVER_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Driver lifecycle
// ---------------------------------------------------------------------------
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_create(M_driver_version *version,
                                  M_driver_driver_t *driver) {
  if (!version || !driver)
    return M_DRIVER_STATUS_INVALID_ARG;

  if (version->major != M_DRIVER_INTERFACE_VERSION_MAJOR)
    return M_DRIVER_STATUS_INCOMPATIBLE_VERSION;

  // Write back the mock plugin's version.
  version->major = M_DRIVER_INTERFACE_VERSION_MAJOR;
  version->minor = M_DRIVER_INTERFACE_VERSION_MINOR;
  version->patch = M_DRIVER_INTERFACE_VERSION_PATCH;

  *driver = (M_driver_driver_t)&g_driver_sentinel;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_destroy(M_driver_driver_t driver) {
  (void)driver;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_property(M_driver_driver_t handle,
                                    const char *property_name, void *value) {
  if (!property_name || !value)
    return M_DRIVER_STATUS_INVALID_ARG;

  if (strcmp(property_name, M_DRIVER_PROPERTY_API) == 0) {
    *(const char **)value = "mock";
    return M_DRIVER_STATUS_SUCCESS;
  }

  if (strcmp(property_name, M_DRIVER_PROPERTY_VERSION) == 0) {
    M_driver_version *v = (M_driver_version *)value;
    v->major = M_DRIVER_INTERFACE_VERSION_MAJOR;
    v->minor = M_DRIVER_INTERFACE_VERSION_MINOR;
    v->patch = M_DRIVER_INTERFACE_VERSION_PATCH;
    return M_DRIVER_STATUS_SUCCESS;
  }

  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_device_count(M_driver_driver_t driver,
                                        int64_t *count) {
  (void)driver;
  if (!count)
    return M_DRIVER_STATUS_INVALID_ARG;
  *count = MOCK_DEVICE_COUNT;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_device_get(M_driver_driver_t driver,
                                      int64_t device_id,
                                      M_driver_device_t *device) {
  (void)driver;
  if (!device)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (device_id < 0 || device_id >= MOCK_DEVICE_COUNT)
    return M_DRIVER_STATUS_BAD_DEVICE;
  *device = (M_driver_device_t)&g_device_sentinels[device_id];
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_device_property(M_driver_device_t device,
                                           const char *property_name,
                                           void *value) {
  (void)device;
  if (!property_name || !value)
    return M_DRIVER_STATUS_INVALID_ARG;
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_context_create(M_driver_device_t device,
                                          M_driver_context_t *context) {
  (void)device;
  if (!context)
    return M_DRIVER_STATUS_INVALID_ARG;
  *context = (M_driver_context_t)&g_context_sentinel;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_context_destroy(M_driver_context_t context) {
  (void)context;
  return M_DRIVER_STATUS_SUCCESS;
}

// Mock implements a single global "current driver context" so tests can
// observe set/get round-trips.
static M_driver_native_handle_t g_mock_current_driver_ctx = 0;

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_context_set_current(M_driver_context_t context) {
  if (!context)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  g_mock_current_driver_ctx = (M_driver_native_handle_t)context;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_context_get_current_driver_context(
    M_driver_context_t context, M_driver_native_handle_t *driver_context) {
  if (!context)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (!driver_context)
    return M_DRIVER_STATUS_INVALID_ARG;
  *driver_context = g_mock_current_driver_ctx;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_context_set_current_driver_context(
    M_driver_context_t context, M_driver_native_handle_t driver_context) {
  if (!context)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  g_mock_current_driver_ctx = driver_context;
  return M_DRIVER_STATUS_SUCCESS;
}

// Fixed sentinel values so tests can assert the plumbing end to end.
#define MOCK_MEMORY_INFO_FREE (3ull << 30)
#define MOCK_MEMORY_INFO_TOTAL (4ull << 30)

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_context_memory_info(M_driver_context_t context,
                                               uint64_t *free,
                                               uint64_t *total) {
  if (!context)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (!free || !total)
    return M_DRIVER_STATUS_INVALID_ARG;
  *free = MOCK_MEMORY_INFO_FREE;
  *total = MOCK_MEMORY_INFO_TOTAL;
  return M_DRIVER_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Memory
//
// A handle is a small heap struct holding the allocation pointer and whether
// the mock owns it. `owning` is true for alloc'd blocks and owning wraps, and
// false for non-owning wraps and any handle after unwrap; freeing a
// non-owning handle reclaims only the struct, never the underlying pointer.
// ---------------------------------------------------------------------------
struct M_driver_memory {
  void *ptr;
  bool owning;
  // True only for pinned host allocations, whose `ptr` is host-dereferenceable.
  bool host_accessible;
};

static M_driver_status_t mock_alloc(uint64_t size, bool host_accessible,
                                    M_driver_memory_t *memory) {
  if (!memory)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (size == 0)
    return M_DRIVER_STATUS_INVALID_ARG;
  void *ptr = malloc((size_t)size);
  if (!ptr)
    return M_DRIVER_STATUS_OUT_OF_MEMORY;
  M_driver_memory_t m = malloc(sizeof(*m));
  if (!m) {
    free(ptr);
    return M_DRIVER_STATUS_OUT_OF_MEMORY;
  }
  m->ptr = ptr;
  m->owning = true;
  m->host_accessible = host_accessible;
  *memory = m;
  return M_DRIVER_STATUS_SUCCESS;
}

// Frees the handle, reclaiming the underlying allocation only when owned.
static M_driver_status_t mock_free(M_driver_memory_t memory) {
  if (!memory)
    return M_DRIVER_STATUS_SUCCESS;
  if (memory->owning)
    free(memory->ptr);
  free(memory);
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_alloc_pinned(M_driver_context_t context,
                                               uint64_t size,
                                               M_driver_memory_t *memory) {
  (void)context;
  return mock_alloc(size, /*host_accessible=*/true, memory);
}

M_DRIVER_API_EXPORT
M_driver_status_t
M_driver_memory_alloc_pinned_mapped(M_driver_context_t context, uint64_t size,
                                    M_driver_memory_t *memory, void **host_ptr,
                                    uint64_t *device_address) {
  (void)context;
  if (!host_ptr || !device_address)
    return M_DRIVER_STATUS_INVALID_ARG;
  M_driver_status_t status = mock_alloc(size, /*host_accessible=*/true, memory);
  if (status != M_DRIVER_STATUS_SUCCESS)
    return status;
  // Host memory is its own device address on the mock.
  *host_ptr = (void *)*memory;
  *device_address = (uint64_t)(uintptr_t)*memory;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_free_pinned(M_driver_context_t context,
                                              M_driver_memory_t memory) {
  (void)context;
  return mock_free(memory);
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_alloc_sync(M_driver_context_t context,
                                             uint64_t size,
                                             M_driver_memory_t *memory) {
  (void)context;
  return mock_alloc(size, /*host_accessible=*/false, memory);
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_free_sync(M_driver_context_t context,
                                            M_driver_memory_t memory) {
  (void)context;
  return mock_free(memory);
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_wrap(M_driver_context_t context,
                                       uint64_t address, uint64_t size,
                                       bool owning, M_driver_memory_t *memory) {
  (void)context;
  (void)size;
  if (!memory || address == 0)
    return M_DRIVER_STATUS_INVALID_ARG;
  M_driver_memory_t m = malloc(sizeof(*m));
  if (!m)
    return M_DRIVER_STATUS_OUT_OF_MEMORY;
  m->ptr = (void *)(uintptr_t)address;
  m->owning = owning;
  // Wrapped memory has unknown residency; treat it as not host-accessible.
  m->host_accessible = false;
  *memory = m;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_unwrap(M_driver_context_t context,
                                         M_driver_memory_t memory,
                                         uint64_t *address) {
  (void)context;
  if (!memory || !address)
    return M_DRIVER_STATUS_INVALID_ARG;
  memory->owning = false;
  *address = (uint64_t)(uintptr_t)memory->ptr;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_alloc_async(M_driver_queue_t queue,
                                              uint64_t size,
                                              M_driver_memory_t *memory) {
  (void)queue;
  return mock_alloc(size, /*host_accessible=*/false, memory);
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_free_async(M_driver_queue_t queue,
                                             M_driver_memory_t memory) {
  (void)queue;
  return mock_free(memory);
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_property(M_driver_memory_t memory,
                                           const char *property_name,
                                           void *value) {
  if (!property_name || !value)
    return M_DRIVER_STATUS_INVALID_ARG;

  if (strcmp(property_name, M_DRIVER_MEMORY_PROPERTY_ADDRESS) == 0) {
    *(void **)value = memory->ptr;
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (strcmp(property_name, M_DRIVER_MEMORY_PROPERTY_HOST_ADDRESS) == 0) {
    // Refuse a host address for memory that is not host-accessible, rather
    // than hand back a device pointer that lies about its residency.
    if (!memory->host_accessible)
      return M_DRIVER_STATUS_INVALID_ARG;
    *(void **)value = memory->ptr;
    return M_DRIVER_STATUS_SUCCESS;
  }

  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

// ---------------------------------------------------------------------------
// Queue
// ---------------------------------------------------------------------------
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_create(M_driver_context_t context,
                                        M_driver_queue_t *queue) {
  (void)context;
  if (!queue)
    return M_DRIVER_STATUS_INVALID_ARG;
  *queue = (M_driver_queue_t)&g_queue_sentinel;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_destroy(M_driver_context_t context,
                                         M_driver_queue_t queue) {
  (void)context;
  (void)queue;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_property(M_driver_queue_t queue,
                                          const char *property_name,
                                          void *value) {
  if (!queue)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (!property_name || !value)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (strcmp(property_name, M_DRIVER_QUEUE_PROPERTY_NATIVE_HANDLE) == 0) {
    *(uint64_t *)value = (uint64_t)(uintptr_t)queue;
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_copy_to_device(M_driver_queue_t queue,
                                                M_driver_memory_view_t dst,
                                                const void *src) {
  (void)queue;
  if (dst && dst->memory && src && dst->size > 0)
    memcpy((char *)dst->memory->ptr + dst->offset, src, (size_t)dst->size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_copy_from_device(M_driver_queue_t queue,
                                                  void *dst,
                                                  M_driver_memory_view_t src) {
  (void)queue;
  if (dst && src && src->memory && src->size > 0)
    memcpy(dst, (char *)src->memory->ptr + src->offset, (size_t)src->size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_copy_intra_device(M_driver_queue_t queue,
                                                   M_driver_memory_view_t dst,
                                                   M_driver_memory_view_t src) {
  (void)queue;
  if (dst && dst->memory && src && src->memory && dst->size > 0)
    memcpy((char *)dst->memory->ptr + dst->offset,
           (char *)src->memory->ptr + src->offset, (size_t)dst->size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_copy_inter_device(
    M_driver_queue_t queue, M_driver_memory_view_t dst,
    M_driver_memory_view_t src, M_driver_context_t src_context) {
  (void)queue;
  (void)src_context;
  if (dst && dst->memory && src && src->memory && dst->size > 0)
    memcpy((char *)dst->memory->ptr + dst->offset,
           (char *)src->memory->ptr + src->offset, (size_t)dst->size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_copy_to_device_sync(M_driver_context_t context,
                                               M_driver_memory_view_t dst,
                                               const void *src) {
  (void)context;
  if (dst && dst->memory && src && dst->size > 0)
    memcpy((char *)dst->memory->ptr + dst->offset, src, (size_t)dst->size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_copy_from_device_sync(M_driver_context_t context,
                                                 void *dst,
                                                 M_driver_memory_view_t src) {
  (void)context;
  if (dst && src && src->memory && src->size > 0)
    memcpy(dst, (char *)src->memory->ptr + src->offset, (size_t)src->size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_copy_intra_device_sync(M_driver_context_t context,
                                                  M_driver_memory_view_t dst,
                                                  M_driver_memory_view_t src) {
  (void)context;
  if (dst && dst->memory && src && src->memory && dst->size > 0)
    memcpy((char *)dst->memory->ptr + dst->offset,
           (char *)src->memory->ptr + src->offset, (size_t)dst->size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_copy_inter_device_sync(
    M_driver_context_t context, M_driver_memory_view_t dst,
    M_driver_memory_view_t src, M_driver_context_t src_context) {
  (void)context;
  (void)src_context;
  if (dst && dst->memory && src && src->memory && dst->size > 0)
    memcpy((char *)dst->memory->ptr + dst->offset,
           (char *)src->memory->ptr + src->offset, (size_t)dst->size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_set_memory(M_driver_queue_t queue,
                                            M_driver_memory_view_t dst,
                                            uint8_t value) {
  (void)queue;
  if (!dst || !dst->memory || dst->size == 0)
    return M_DRIVER_STATUS_SUCCESS;

  memset((char *)dst->memory->ptr + dst->offset, value, (size_t)dst->size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_fill(M_driver_queue_t queue,
                                      M_driver_memory_view_t dst,
                                      uint64_t value, uint64_t value_size) {
  (void)queue;
  if (!dst || !dst->memory || dst->size == 0)
    return M_DRIVER_STATUS_SUCCESS;

  char *p = (char *)dst->memory->ptr + dst->offset;
  for (uint64_t i = 0; i < dst->size; i += value_size) {
    uint64_t remaining = dst->size - i;
    uint64_t n = remaining < value_size ? remaining : value_size;
    memcpy(p + i, &value, (size_t)n);
  }
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t
M_driver_queue_execute(M_driver_queue_t queue, M_driver_function_t function,
                       M_driver_queue_execute_config_t *config, void **args,
                       uint64_t *arg_sizes, uint32_t num_args) {
  (void)queue;
  (void)function;
  (void)config;
  (void)args;
  (void)arg_sizes;
  (void)num_args;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_record_event(M_driver_queue_t queue,
                                              M_driver_event_t event) {
  (void)queue;
  (void)event;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_wait_for_events(M_driver_queue_t queue,
                                                 M_driver_event_t *events,
                                                 uint32_t num_events) {
  (void)queue;
  for (uint32_t i = 0; i < num_events; ++i)
    (void)events[i];
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_synchronize(M_driver_queue_t queue) {
  (void)queue;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_is_stream(M_driver_queue_t queue,
                                           bool *is_stream) {
  (void)queue;
  if (!is_stream)
    return M_DRIVER_STATUS_INVALID_ARG;
  *is_stream = false;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_launch_host_func(M_driver_queue_t queue,
                                                  void (*fn)(void *),
                                                  void *user_data) {
  if (!queue)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (!fn)
    return M_DRIVER_STATUS_INVALID_ARG;
  // The mock queue is synchronous: run the callback inline.
  fn(user_data);
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_wait_value64(M_driver_queue_t queue,
                                              uint64_t device_address,
                                              uint64_t value) {
  if (!queue)
    return M_DRIVER_STATUS_BAD_QUEUE;
  // The mock queue is synchronous, so the wait cannot be satisfied later:
  // check once instead of blocking.
  if (*(const uint64_t *)(uintptr_t)device_address != value)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;
  return M_DRIVER_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_event_create(M_driver_context_t context,
                                        M_driver_event_flags_t flags,
                                        M_driver_event_t *event) {
  (void)context;
  if (flags & ~M_DRIVER_EVENT_FLAGS_ALL)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (!event)
    return M_DRIVER_STATUS_INVALID_ARG;
  *event = (M_driver_event_t)&g_event_sentinel;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_event_destroy(M_driver_context_t context,
                                         M_driver_event_t event) {
  (void)context;
  (void)event;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_event_synchronize(M_driver_context_t context,
                                             M_driver_event_t event) {
  (void)context;
  (void)event;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_is_event_ready(M_driver_context_t context,
                                          M_driver_event_t event,
                                          bool *is_ready) {
  (void)context;
  (void)event;
  if (!is_ready)
    return M_DRIVER_STATUS_INVALID_ARG;
  *is_ready = true;
  return M_DRIVER_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Bundles and functions
// ---------------------------------------------------------------------------
M_DRIVER_API_EXPORT
M_driver_status_t
M_driver_bundle_load(M_driver_context_t context,
                     M_driver_static_bundle_t bundle,
                     M_driver_bundle_compilation_options_t opts,
                     M_driver_runtime_bundle_t *runtime_bundle) {
  (void)context;
  (void)bundle;
  (void)opts;
  if (!runtime_bundle)
    return M_DRIVER_STATUS_INVALID_ARG;
  *runtime_bundle = (M_driver_runtime_bundle_t)&g_bundle_sentinel;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_bundle_unload(M_driver_context_t context,
                                         M_driver_runtime_bundle_t bundle) {
  (void)context;
  (void)bundle;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_function_load(M_driver_context_t context,
                                         M_driver_runtime_bundle_t bundle,
                                         const char *function_name,
                                         uint64_t function_name_len,
                                         M_driver_function_t *function) {
  (void)context;
  (void)bundle;
  (void)function_name;
  (void)function_name_len;
  if (!function)
    return M_DRIVER_STATUS_INVALID_ARG;
  *function = (M_driver_function_t)&g_function_sentinel;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_function_unload(M_driver_context_t context,
                                           M_driver_function_t function) {
  (void)context;
  (void)function;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_function_property(M_driver_context_t context,
                                             M_driver_function_t function,
                                             const char *property_name,
                                             void *value) {
  (void)context;
  if (!function)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  if (!property_name || !value)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (strcmp(property_name, M_DRIVER_FUNCTION_PROPERTY_NATIVE_HANDLE) == 0) {
    // The mock's "native module handle" is the bundle sentinel.
    *(uint64_t *)value = (uint64_t)(uintptr_t)&g_bundle_sentinel;
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_function_set_attribute(M_driver_context_t context,
                                                  M_driver_function_t function,
                                                  int32_t attribute,
                                                  int32_t value) {
  (void)context;
  (void)attribute;
  (void)value;
  if (!function)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  return M_DRIVER_STATUS_SUCCESS;
}

// Fixed occupancy so tests can assert the plumbing end to end.
#define MOCK_OCCUPANCY_MAX_ACTIVE_BLOCKS 4

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_function_occupancy_max_active_blocks(
    M_driver_context_t context, M_driver_function_t function,
    int32_t block_size, uint64_t dynamic_shared_memory_bytes,
    int32_t *num_blocks) {
  (void)context;
  (void)block_size;
  (void)dynamic_shared_memory_bytes;
  if (!function)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  if (!num_blocks)
    return M_DRIVER_STATUS_INVALID_ARG;
  *num_blocks = MOCK_OCCUPANCY_MAX_ACTIVE_BLOCKS;
  return M_DRIVER_STATUS_SUCCESS;
}
