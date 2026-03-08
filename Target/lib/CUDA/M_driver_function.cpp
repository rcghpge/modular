//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "Error.h"
#include "Function.h"
#include "HAL/M_driver_api.h"

#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstring>
#include <memory>
#include <utility>

static std::pair<llvm::SmallVector<CUjit_option, 2>,
                 llvm::SmallVector<uintptr_t, 2>>
setupCompilationOptions(const char *debugLevel, int32_t optimizationLevel) {
  static constexpr int kDefaultOptimizationLevel = 4;

  llvm::SmallVector<CUjit_option, 2> optionKind;
  llvm::SmallVector<uintptr_t, 2> optionValues;

  if (debugLevel != nullptr && std::strcmp(debugLevel, "full") == 0) {
    // The PTX compiler complains if debugLevel=full and optimization level is
    // something other than 0 or 4 (the default).
    assert(optimizationLevel < 0 || optimizationLevel == 0 ||
           optimizationLevel == kDefaultOptimizationLevel);

    optionKind.push_back(CU_JIT_GENERATE_DEBUG_INFO);
    optionValues.push_back(1);
  } else if (debugLevel != nullptr &&
             std::strcmp(debugLevel, "line-tables") == 0) {
    optionKind.push_back(CU_JIT_GENERATE_LINE_INFO);
    optionValues.push_back(1);
  }

  // The default optimization level for NVIDIA GPUs is 4.
  if (optimizationLevel < 0)
    optimizationLevel = kDefaultOptimizationLevel;

  // The Mojo compiler uses optimization level 3 as the highest optimization
  // level, but 4 is the highest for CUDA.
  if (optimizationLevel == 3)
    optimizationLevel = kDefaultOptimizationLevel;

  if (optimizationLevel != kDefaultOptimizationLevel) {
    optionKind.push_back(CU_JIT_OPTIMIZATION_LEVEL);
    optionValues.push_back(static_cast<uintptr_t>(optimizationLevel));
  }

  return {std::move(optionKind), std::move(optionValues)};
}

M_driver_status_t
M_driver_function_load(M_driver_driver_t driver, M_driver_context_t context,
                       const void *code, const char *function_name,
                       int32_t max_dynamic_shared_bytes,
                       const char *debug_level, int32_t optimization_level,
                       M_driver_function_t *function) {
  CUDA_CHECK(driver->cuda().cuCtxSetCurrent(context->primaryCtx),
             M_DRIVER_STATUS_BAD_CONTEXT);

  auto [optionKind, optionValues] =
      setupCompilationOptions(debug_level, optimization_level);

  // Load module from PTX.
  CUmodule module;
  // TODO: Might need equivalent of POLL_PENDING_EVENTS here after queue and
  // events are in place, line 1716 CUDADeviceContext.cpp
  CUDA_CHECK(driver->cuda().cuModuleLoadDataEx(
                 &module, code, optionKind.size(), optionKind.data(),
                 reinterpret_cast<void **>(optionValues.data())),
             M_DRIVER_STATUS_UNKNOWN_ERROR);

  // Get function from module.
  CUfunction func;
  CUresult result =
      driver->cuda().cuModuleGetFunction(&func, module, function_name);
  if (result != CUDA_SUCCESS) {
    driver->cuda().cuModuleUnload(module);
    return M_DRIVER_STATUS_UNKNOWN_ERROR;
  }

  CUDA_CHECK(driver->cuda().cuFuncSetAttribute(
                 func, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1),
             M_DRIVER_STATUS_UNKNOWN_ERROR);

  // Set max dynamic shared memory if specified.
  if (max_dynamic_shared_bytes >= 0) {
    CUDA_CHECK(driver->cuda().cuFuncSetAttribute(
                   func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                   max_dynamic_shared_bytes),
               M_DRIVER_STATUS_UNKNOWN_ERROR);
  }

  auto ptr =
      std::make_unique<M_driver_function>(M_driver_function{module, func});
  *function = ptr.release();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_function_unload(M_driver_driver_t driver,
                                           M_driver_context_t context,
                                           M_driver_function_t function) {
  CUDA_CHECK(driver->cuda().cuCtxSetCurrent(context->primaryCtx),
             M_DRIVER_STATUS_BAD_CONTEXT);

  CUDA_CHECK(driver->cuda().cuModuleUnload(function->module),
             M_DRIVER_STATUS_UNKNOWN_ERROR);

  delete function;
  return M_DRIVER_STATUS_SUCCESS;
}
