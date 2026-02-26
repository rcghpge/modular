//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#include "Error.h"
#include "HAL/M_driver_api.h"
#include "HAL/M_driver_common.h"

#include "llvm/ADT/SmallVector.h"

#include "cuda.h"

#include <array>
#include <cassert>
#include <cstring>
#include <type_traits>

namespace {
struct DriverState {
  llvm::SmallVector<CUcontext> cudaContexts{};
  llvm::SmallVector<bool> cudaContextsValid{};
};
} // namespace

static DriverState *getState() {
  static DriverState theDriverState;
  return &theDriverState;
}

// Ensure that address is not changed by the cast.
// In rare cases, reinterpret_cast can change the address.
// We don't want to ever find ourselves in such a situation,
// esp. the hard way.
template <typename ToPtr, typename FromPtr>
inline ToPtr paranoid_reinterpret_pointer_cast(FromPtr from) {
  static_assert(std::is_pointer_v<FromPtr>, "FromPtr must be a pointer");
  static_assert(std::is_pointer_v<ToPtr>, "ToPtr must be a pointer");
  return static_cast<ToPtr>(static_cast<void *>(from));
}

#define RETURN_IF_ERROR(stmt, error)                                           \
  do {                                                                         \
    M_driver_status_t status = (stmt);                                         \
    if (status != M_DRIVER_STATUS_SUCCESS)                                     \
      return error;                                                            \
  } while (0)

#define QUICK_RETURN_IF_ERROR(stmt)                                            \
  do {                                                                         \
    M_driver_status_t status = (stmt);                                         \
    if (status != M_DRIVER_STATUS_SUCCESS)                                     \
      return status;                                                           \
  } while (0)

#define CUDA_CHECK(stmt, errorCode)                                            \
  do {                                                                         \
    CUresult result = (stmt);                                                  \
    if (result != CUDA_SUCCESS)                                                \
      return errorCode;                                                        \
  } while (0)

M_driver_status_t initialize_devices(void) {
  DriverState *state = getState();
  if (!state)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t finalize_devices(void) {
  DriverState *state = getState();
  if (!state)
    return M_DRIVER_STATUS_SUCCESS;

  bool success = true;
  size_t contextCount = state->cudaContexts.size();
  for (size_t i = 0; i < contextCount; i++) {
    if (state->cudaContextsValid[i]) {
      M_driver_status_t status =
          M_driver_device_destroy_context(static_cast<M_driver_context_t>(i));
      success = (success && (status == M_DRIVER_STATUS_SUCCESS));
    }
  }
  if (!success)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  state->cudaContexts.clear();
  state->cudaContextsValid.clear();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device_count(int64_t *count) {
  int cudaCount = 0;
  CUresult result = cuDeviceGetCount(&cudaCount);
  if (result != CUDA_SUCCESS)
    return M_DRIVER_STATUS_NO_DEVICES;

  *count = static_cast<int64_t>(cudaCount);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device_get(int64_t id, M_driver_device_t *device) {
  static_assert(sizeof(CUdevice) <= sizeof(M_driver_device_t),
                "Mismatch in device-handle size");

  int64_t deviceCount = 0;
  QUICK_RETURN_IF_ERROR(M_driver_device_count(&deviceCount));
  if (id < 0 || id >= deviceCount)
    return M_DRIVER_STATUS_BAD_DEVICE;

  CUdevice cudaDevice = 0;
  CUresult result = cuDeviceGet(&cudaDevice, id);
  if (result != CUDA_SUCCESS)
    return M_DRIVER_STATUS_BAD_DEVICE;

  *paranoid_reinterpret_pointer_cast<CUdevice *>(device) = cudaDevice;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_device_property([[maybe_unused]] M_driver_device_t device,
                         [[maybe_unused]] const char *property_name,
                         [[maybe_unused]] void *value) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t M_driver_device_create_context(M_driver_device_t device,
                                                 M_driver_context_t *context) {
  DriverState *state = getState();
  CUdevice cudaDevice = *paranoid_reinterpret_pointer_cast<CUdevice *>(&device);
  CUcontext cudaContext = nullptr;
  if (state->cudaContexts.empty())
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&cudaContext, cudaDevice),
               M_DRIVER_STATUS_BAD_DEVICE);
  else
    CUDA_CHECK(cuCtxCreate(&cudaContext, nullptr, 0, cudaDevice),
               M_DRIVER_STATUS_BAD_DEVICE);
  state->cudaContexts.push_back(cudaContext);
  state->cudaContextsValid.push_back(true);
  *context = state->cudaContexts.size() - 1;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device_destroy_context(M_driver_context_t context) {
  DriverState *state = getState();
  if (static_cast<size_t>(context) >= state->cudaContexts.size() ||
      !state->cudaContextsValid[context]) {
    return M_DRIVER_STATUS_BAD_CONTEXT;
  }
  CUcontext cudaContext = state->cudaContexts[context];
  if (static_cast<size_t>(context) == 0) {
    // Primary context. Don't destroy, just release (ref count).
    CUdevice cudaDevice = 0x0ULL;
    CUDA_CHECK(cuCtxGetDevice_v2(&cudaDevice, cudaContext),
               M_DRIVER_STATUS_BAD_CONTEXT);
  } else {
    CUDA_CHECK(cuCtxDestroy(cudaContext), M_DRIVER_STATUS_BAD_CONTEXT);
  }
  state->cudaContextsValid[context] = false;
  return M_DRIVER_STATUS_SUCCESS;
}
