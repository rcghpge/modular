//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#include "HAL/M_driver_api.h"
#include "HAL/M_driver_common.h"
#include "error.h"

#include "cuda.h"

#include <array>
#include <cassert>
#include <cstring>
#include <mutex>
#include <vector>

struct CUDAContextState {
  std::vector<CUcontext> cudaContexts{};
  std::vector<bool> cudaContextsValid{};
};
static CUDAContextState *cudaContextState = nullptr;
static std::mutex cudaContextStateMutex;

#define RETURN_IF_ERROR(stmt, error)                                           \
  do {                                                                         \
    M_driver_status_t status = (stmt);                                         \
    if (status != M_DRIVER_STATUS_SUCCESS) {                                   \
      return error;                                                            \
    }                                                                          \
  } while (0)

#define QUICK_RETURN_IF_ERROR(stmt)                                            \
  do {                                                                         \
    M_driver_status_t status = (stmt);                                         \
    if (status != M_DRIVER_STATUS_SUCCESS) {                                   \
      return status;                                                           \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(stmt, errorCode)                                            \
  do {                                                                         \
    CUresult result = (stmt);                                                  \
    if (result != CUDA_SUCCESS) {                                              \
      return errorCode;                                                        \
    }                                                                          \
  } while (0)

extern "C" M_driver_status_t initialize_devices(void) {
  std::lock_guard<std::mutex> lock(cudaContextStateMutex);
  assert(cudaContextState == nullptr);
  cudaContextState = new (std::nothrow) CUDAContextState();
  if (!cudaContextState) {
    return M_DRIVER_STATUS_UNKNOWN_ERROR;
  }
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t finalize_devices(void) {
  std::lock_guard<std::mutex> lock(cudaContextStateMutex);
  if (!cudaContextState) {
    return M_DRIVER_STATUS_SUCCESS;
  }
  auto &cudaContexts = cudaContextState->cudaContexts;
  auto const &cudaContextsValid = cudaContextState->cudaContextsValid;
  for (size_t i = 0; i < cudaContexts.size(); i++) {
    if (cudaContextsValid[i]) {
      QUICK_RETURN_IF_ERROR(M_driver_device_destroy_context(i));
    }
  }
  delete cudaContextState;
  cudaContextState = nullptr;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device_count(int64_t *count) {
  int cudaCount = 0;
  CUresult result = cuDeviceGetCount(&cudaCount);
  if (result != CUDA_SUCCESS) {
    return M_DRIVER_STATUS_NO_DEVICES;
  }
  *count = int64_t(cudaCount);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device_get(int64_t id, M_driver_device_t *device) {
  static_assert(sizeof(CUdevice) <= sizeof(M_driver_device_t),
                "Mismatch in device-handle size");

  int64_t deviceCount = 0;
  QUICK_RETURN_IF_ERROR(M_driver_device_count(&deviceCount));
  if (id < 0 || id >= deviceCount) {
    return M_DRIVER_STATUS_BAD_DEVICE;
  }
  CUdevice cudaDevice = 0;
  CUresult result = cuDeviceGet(&cudaDevice, id);
  if (result != CUDA_SUCCESS) {
    return M_DRIVER_STATUS_BAD_DEVICE;
  }
  *device = *(M_driver_device_t *)&cudaDevice;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device_property(M_driver_device_t device,
                                           const char *property_name,
                                           void *value) {
  (void)device;
  (void)property_name;
  (void)value;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t M_driver_device_create_context(M_driver_device_t device,
                                                 M_driver_context_t *context) {
  std::lock_guard<std::mutex> lock(cudaContextStateMutex);
  assert(cudaContextState != nullptr);
  CUdevice cudaDevice = *(CUdevice *)&device;
  CUcontext cudaContext = nullptr;
  if (cudaContextState->cudaContexts.size() == 0) {
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&cudaContext, cudaDevice),
               M_DRIVER_STATUS_BAD_DEVICE);
  } else {
    CUDA_CHECK(cuCtxCreate(&cudaContext, 0, cudaDevice),
               M_DRIVER_STATUS_BAD_DEVICE);
  }
  cudaContextState->cudaContexts.push_back(cudaContext);
  cudaContextState->cudaContextsValid.push_back(true);
  *context = cudaContextState->cudaContexts.size() - 1;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device_destroy_context(M_driver_context_t context) {
  std::lock_guard<std::mutex> lock(cudaContextStateMutex);
  auto &cudaContexts = cudaContextState->cudaContexts;
  auto &cudaContextsValid = cudaContextState->cudaContextsValid;
  if (!cudaContextState || context >= cudaContexts.size() ||
      !cudaContextsValid[context]) {
    return M_DRIVER_STATUS_BAD_CONTEXT;
  }
  CUcontext cudaContext = cudaContexts[context];
  if (context == 0) {
    // Primary context. Don't destroy, just release (ref count).
    // CUDA v13+ has one concise API to retrieve a device from a context.
    // But till CUDA v12, we can only retrieve device for the current context.
    // So we save the previous context and restore it after retrieving device.
    CUcontext prevCudaContext = nullptr;
    CUDA_CHECK(cuCtxGetCurrent(&prevCudaContext),
               M_DRIVER_STATUS_UNKNOWN_ERROR);
    CUDA_CHECK(cuCtxSetCurrent(cudaContext), M_DRIVER_STATUS_UNKNOWN_ERROR);
    CUdevice cudaDevice = 0x0ULL;
    CUDA_CHECK(cuCtxGetDevice(&cudaDevice), M_DRIVER_STATUS_BAD_CONTEXT);
    CUDA_CHECK(cuDevicePrimaryCtxRelease(cudaDevice),
               M_DRIVER_STATUS_BAD_DEVICE);
    CUDA_CHECK(cuCtxSetCurrent(prevCudaContext), M_DRIVER_STATUS_UNKNOWN_ERROR);
  } else {
    CUDA_CHECK(cuCtxDestroy(cudaContext), M_DRIVER_STATUS_BAD_CONTEXT);
  }
  cudaContextsValid[context] = false;
  return M_DRIVER_STATUS_SUCCESS;
}
