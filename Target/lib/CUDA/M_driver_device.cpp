//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#include "Driver.h"
#include "Error.h"
#include "HAL/M_driver_api.h"
#include "HAL/M_driver_common.h"

#include <cassert>
#include <type_traits>

using namespace M::Target::CUDA;

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

M_driver_status_t M_driver_device_count(M_driver_driver_t driver,
                                        int64_t *count) {
  Driver *state = Driver::from(driver);
  unsigned int nvmlCount = 0;
  if (state->nvml().nvmlDeviceGetCount_v2(&nvmlCount) != NVML_SUCCESS)
    return M_DRIVER_STATUS_NO_DEVICES;
  *count = static_cast<int64_t>(nvmlCount);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device_get(M_driver_driver_t driver, int64_t id,
                                      M_driver_device_t *device) {
  Driver *state = Driver::from(driver);

  int64_t deviceCount{};
  QUICK_RETURN_IF_ERROR(M_driver_device_count(driver, &deviceCount));
  if (id < 0 || id >= deviceCount)
    return M_DRIVER_STATUS_BAD_DEVICE;

  CUdevice cudaDevice{};
  CUDA_CHECK(state->cuda().cuDeviceGet(&cudaDevice, id),
             M_DRIVER_STATUS_BAD_DEVICE);

  Device *dev = state->findDevice(cudaDevice);
  if (!dev) {
    state->devices.emplace_back(cudaDevice);
    dev = &state->devices.back();
  }

  *device = dev->cast();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_device_property([[maybe_unused]] M_driver_device_t device,
                         [[maybe_unused]] const char *property_name,
                         [[maybe_unused]] void *value) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t M_driver_context_create(M_driver_driver_t driver,
                                          M_driver_device_t device,
                                          M_driver_context_t *context) {
  Driver *state = Driver::from(driver);
  Device &dev = Device::from(device);

  CUcontext primaryCtx;
  CUDA_CHECK(state->cuda().cuDevicePrimaryCtxRetain(&primaryCtx, dev.deviceId),
             M_DRIVER_STATUS_BAD_DEVICE);

  *context = reinterpret_cast<M_driver_context_t>(primaryCtx);

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context_destroy(M_driver_driver_t driver,
                                           M_driver_context_t context) {
  Driver *state = Driver::from(driver);
  auto primaryCtx = reinterpret_cast<CUcontext>(context);

  CUdevice device;
  CUDA_CHECK(state->cuda().cuCtxGetDevice_v2(&device, primaryCtx),
             M_DRIVER_STATUS_BAD_DEVICE);
  CUDA_CHECK(state->cuda().cuDevicePrimaryCtxRelease(device),
             M_DRIVER_STATUS_BAD_CONTEXT);
  return M_DRIVER_STATUS_SUCCESS;
}
