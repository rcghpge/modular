//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ALL_TOOLS_MODULAR_API_EXECUTOR_MODULARAPIINTERFACE_H
#define ALL_TOOLS_MODULAR_API_EXECUTOR_MODULARAPIINTERFACE_H

#include "Support/ErrorOr.h"
#include "modular/c/common.h"
#include "modular/c/tensorflow/config.h"
#include "modular/cpp/aliases.h"

namespace M {
class ModularAPIInterface {
public:
  ErrorOrSuccess loadModularAndInitSymbols();

  decltype(&M_version) version = nullptr;

  decltype(&M_isError) isError = nullptr;
  decltype(&M_getError) getError = nullptr;

  decltype(&M_newStatus) newStatus = nullptr;
  decltype(&M_freeStatus) freeStatus = nullptr;

  decltype(&M_newRuntimeConfig) newRuntimeConfig = nullptr;
  decltype(&M_freeRuntimeConfig) freeRuntimeConfig = nullptr;

  decltype(&M_newRuntimeContext) newRuntimeContext = nullptr;
  decltype(&M_freeRuntimeContext) freeRuntimeContext = nullptr;

  decltype(&M_newCompileConfig) newCompileConfig = nullptr;
  decltype(&M_freeCompileConfig) freeCompileConfig = nullptr;
  decltype(&M_setModelPath) setModelPath = nullptr;
  decltype(&M_setExportedName) setExportedName = nullptr;
  decltype(&M_setNumThreads) setNumThreads = nullptr;
  decltype(&M_compileModel) compileModel = nullptr;
  decltype(&M_freeCompiledModel) freeCompiledModel = nullptr;

  decltype(&M_initModel) initModel = nullptr;
  decltype(&M_waitForModel) waitForModel = nullptr;
  decltype(&M_freeModel) freeModel = nullptr;
  decltype(&M_getInputNames) getInputNames = nullptr;
  decltype(&M_getNumModelInputs) getNumModelInputs = nullptr;
  decltype(&M_getModelInputSpecByName) getModelInputSpecByName = nullptr;
  decltype(&M_getOutputNames) getOutputNames = nullptr;
  decltype(&M_getNumModelOutputs) getNumModelOutputs = nullptr;
  decltype(&M_getModelOutputSpecByName) getModelOutputSpecByName = nullptr;
  decltype(&M_freeTensorNameArray) freeTensorNameArray = nullptr;
  decltype(&M_getDtype) getDtype = nullptr;
  decltype(&M_sizeOf) sizeOf = nullptr;

  decltype(&M_newAsyncTensorMap) newAsyncTensorMap = nullptr;
  decltype(&M_freeAsyncTensorMap) freeAsyncTensorMap = nullptr;
  decltype(&M_borrowTensorInto) borrowTensorInto = nullptr;
  decltype(&M_getTensorByNameFrom) getTensorByNameFrom = nullptr;
  decltype(&M_getRank) getRank = nullptr;
  decltype(&M_getDimAt) getDimAt = nullptr;
  decltype(&M_getTensorType) getTensorType = nullptr;
  decltype(&M_getTensorNumElements) getTensorNumElements = nullptr;
  decltype(&M_getTensorNameAt) getTensorNameAt = nullptr;
  decltype(&M_getTensorData) getTensorData = nullptr;
  decltype(&M_freeTensor) freeTensor = nullptr;
  decltype(&M_newTensorSpec) newTensorSpec = nullptr;
  decltype(&M_getTensorSpec) getTensorSpec = nullptr;
  decltype(&M_freeTensorSpec) freeTensorSpec = nullptr;

  decltype(&M_executeModelSync) executeModelSync = nullptr;
};
} // namespace M

#endif // ALL_TOOLS_MODULAR_API_EXECUTOR_MODULARAPIINTERFACE_H
