//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_TOOLS_MODULAR_API_EXECUTOR_MODULARAPIINTERFACE_H
#define SDK_TOOLS_MODULAR_API_EXECUTOR_MODULARAPIINTERFACE_H

#include "Support/ErrorOr.h"
#include "max/c/common.h"
#include "max/c/custom_ops.h"
#include "max/c/internal/context.h"
#include "max/c/internal/device.h"
#include "max/c/onnx/config.h"
#include "max/c/pytorch/config.h"
#include "max/cpp/aliases.h"
#include "max/cpp/pytorch/aliases.h"

namespace M {
class ModularAPIInterface {
public:
  ErrorOrSuccess loadModularAndInitSymbols();

  decltype(&M_version) version = nullptr;
  decltype(&M_getDynamicDimensionValue) getDynamicDimensionValue = nullptr;
  decltype(&M_getDynamicRankValue) getDynamicRankValue = nullptr;

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
  decltype(&M_useKernelsFrom) useKernelsFrom = nullptr;
  decltype(&M_setNumThreads) setNumThreads = nullptr;
  decltype(&M_setMaxContext) setMaxContext = nullptr;
  decltype(&M_setAllocatorType) setAllocatorType = nullptr;
  decltype(&M_compileModel) compileModel = nullptr;
  decltype(&M_waitForCompilation) waitForCompilation = nullptr;
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
  decltype(&M_isDynamicRanked) isDynamicRanked = nullptr;
  decltype(&M_getDimAt) getDimAt = nullptr;
  decltype(&M_getTensorType) getTensorType = nullptr;
  decltype(&M_getTensorNumElements) getTensorNumElements = nullptr;
  decltype(&M_getTensorNameAt) getTensorNameAt = nullptr;
  decltype(&M_getTensorData) getTensorData = nullptr;
  decltype(&M_freeTensor) freeTensor = nullptr;
  decltype(&M_newTensorSpec) newTensorSpec = nullptr;
  decltype(&M_getTensorSpec) getTensorSpec = nullptr;
  decltype(&M_freeTensorSpec) freeTensorSpec = nullptr;

  decltype(&M_getValueByNameFrom) getValueByNameFrom = nullptr;
  decltype(&M_getValueType) getValueType = nullptr;
  decltype(&M_getStringFromValue) getStringFromValue = nullptr;
  decltype(&M_getDoubleFromValue) getDoubleFromValue = nullptr;
  decltype(&M_getLongFromValue) getLongFromValue = nullptr;
  decltype(&M_getBoolFromValue) getBoolFromValue = nullptr;
  decltype(&M_getTensorFromValue) getTensorFromValue = nullptr;
  decltype(&M_getDictFromValue) getDictFromValue = nullptr;
  decltype(&M_getDictSize) getDictSize = nullptr;
  decltype(&M_getDictKey) getDictKey = nullptr;
  decltype(&M_getDictValue) getDictValue = nullptr;
  decltype(&M_freeDict) freeDict = nullptr;
  decltype(&M_getListFromValue) getListFromValue = nullptr;
  decltype(&M_getListSize) getListSize = nullptr;
  decltype(&M_getListValue) getListValue = nullptr;
  decltype(&M_freeList) freeList = nullptr;
  decltype(&M_getTupleFromValue) getTupleFromValue = nullptr;
  decltype(&M_getTupleSize) getTupleSize = nullptr;
  decltype(&M_getTupleValue) getTupleValue = nullptr;
  decltype(&M_freeTuple) freeTuple = nullptr;
  decltype(&M_freeValue) freeValue = nullptr;

  decltype(&M_executeModelSync) executeModelSync = nullptr;
  decltype(&M_setDevice) setDevice = nullptr;
  decltype(&M_enableMemoryPlanning) enableMemoryPlanning = nullptr;
  decltype(&M_newTorchInputSpec) newTorchInputSpec = nullptr;
  decltype(&M_setTorchInputSpecs) setTorchInputSpecs = nullptr;
  decltype(&M_setTorchLibraryPath) setTorchLibraryPath = nullptr;
  decltype(&M_setTorchMetadataLibraryPath) setTorchMetadataLibraryPath =
      nullptr;
  decltype(&M_setTorchMetadataLibraryPtr) setTorchMetadataLibraryPtr = nullptr;
  decltype(&M_freeTorchInputSpec) freeTorchInputSpec = nullptr;
};
} // namespace M

#endif // SDK_TOOLS_MODULAR_API_EXECUTOR_MODULARAPIINTERFACE_H
