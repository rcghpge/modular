//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ModularAPIInterface.h"
#include "Support/ErrorOr.h"
#include <filesystem>
#include <string_view>

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Process.h"

using namespace M;

namespace {
#if defined(_WIN32)
constexpr std::string_view modularFrameworkLibraryName =
    "modular-framework-common.dll";
constexpr std::string_view modularONNXExtensionLibraryName =
    "modular-framework-onnx-ext.dll";
constexpr std::string_view modularTorchExtensionLibraryName =
    "modular-framework-torch-ext.dll";
#elif defined(__APPLE__)
constexpr std::string_view modularFrameworkLibraryName =
    "libmodular-framework-common.dylib";
constexpr std::string_view modularONNXExtensionLibraryName =
    "libmodular-framework-onnx-ext.dylib";
constexpr std::string_view modularTorchExtensionLibraryName =
    "libmodular-framework-torch-ext.dylib";
#else
constexpr std::string_view modularFrameworkLibraryName =
    "libmodular-framework-common.so";
constexpr std::string_view modularONNXExtensionLibraryName =
    "libmodular-framework-onnx-ext.so";
constexpr std::string_view modularTorchExtensionLibraryName =
    "libmodular-framework-torch-ext.so";
#endif
} // namespace

static M::ErrorOr<llvm::sys::DynamicLibrary>
fallibleLibraryLoad(std::string_view libraryName) {
  std::string errorMessage;
  std::string base =
      llvm::sys::Process::GetEnv("MODULAR_FRAMEWORK_ROOT").value_or("");
  auto basePath = std::filesystem::path(base);
  auto frameworkLibrary = basePath / libraryName;
  auto dylib = llvm::sys::DynamicLibrary::getPermanentLibrary(
      frameworkLibrary.string().c_str(), &errorMessage);
  if (!dylib.isValid())
    return M::Error(Twine("Failed to load ") + frameworkLibrary.string() +
                    ". The following errors were "
                    "encountered: \n" +
                    errorMessage);
  return dylib;
}

static M::ErrorOr<llvm::sys::DynamicLibrary> fallibleModularLoad() {
  return fallibleLibraryLoad(modularFrameworkLibraryName);
}

template <typename SymbolTy>
M::ErrorOr<SymbolTy>
fallibleGetSymbol(llvm::sys::DynamicLibrary &dynamicLibrary,
                  std::string_view symbolName, SymbolTy) {
  static_assert(std::is_pointer_v<SymbolTy>, "Should be a pointer type");
  void *symbolOr = dynamicLibrary.getAddressOfSymbol(symbolName.data());
  if (!symbolOr)
    return M::Error(Twine("Failed to get symbol") + symbolName);
  return reinterpret_cast<SymbolTy>(symbolOr);
}

ErrorOrSuccess M::ModularAPIInterface::loadModularAndInitSymbols() {
  auto dylibOr = fallibleModularLoad();
  if (dylibOr.isError())
    return dylibOr.takeError();

  auto onnxLibOr = fallibleLibraryLoad(modularONNXExtensionLibraryName);
  if (onnxLibOr.isError())
    return onnxLibOr.takeError();

  auto torchLibOr = fallibleLibraryLoad(modularTorchExtensionLibraryName);
  if (torchLibOr.isError())
    return torchLibOr.takeError();

  llvm::sys::DynamicLibrary dylib = dylibOr.takeValue();
  llvm::sys::DynamicLibrary onnxLib = onnxLibOr.takeValue();
  llvm::sys::DynamicLibrary torchLib = torchLibOr.takeValue();

#define STRINGIFY_SYMBOL(x) #x
#define GET_SYMBOL(VAR, SYMBOL_NAME)                                           \
  auto VAR##Or = fallibleGetSymbol(dylib, STRINGIFY_SYMBOL(SYMBOL_NAME), VAR); \
  if (VAR##Or.isError())                                                       \
    return VAR##Or.takeError();                                                \
  VAR = VAR##Or.takeValue();

  GET_SYMBOL(version, M_version);
  GET_SYMBOL(getDynamicDimensionValue, M_getDynamicDimensionValue);
  GET_SYMBOL(getDynamicRankValue, M_getDynamicRankValue);

  GET_SYMBOL(isError, M_isError);
  GET_SYMBOL(getError, M_getError);

  GET_SYMBOL(newStatus, M_newStatus);
  GET_SYMBOL(freeStatus, M_freeStatus);

  GET_SYMBOL(newRuntimeConfig, M_newRuntimeConfig);
  GET_SYMBOL(freeRuntimeConfig, M_freeRuntimeConfig);

  GET_SYMBOL(newRuntimeContext, M_newRuntimeContext);
  GET_SYMBOL(freeRuntimeContext, M_freeRuntimeContext);

  GET_SYMBOL(newCompileConfig, M_newCompileConfig);
  GET_SYMBOL(freeCompileConfig, M_freeCompileConfig);
  GET_SYMBOL(setNumThreads, M_setNumThreads);
  GET_SYMBOL(setMaxContext, M_setMaxContext);
  GET_SYMBOL(setAllocatorType, M_setAllocatorType);
  GET_SYMBOL(setModelPath, M_setModelPath);
  GET_SYMBOL(setDevice, M_setDevice);

  auto useKernelsOr =
      fallibleGetSymbol(dylib, "M_useKernelsFrom", useKernelsFrom);

  // This is an experimental feature and is okay to not be present.
  if (!useKernelsOr.isError())
    useKernelsFrom = useKernelsOr.takeValue();

  auto enableMemoryPlanningOr = fallibleGetSymbol(
      onnxLib, "M_enableMemoryPlanning", enableMemoryPlanning);
  if (enableMemoryPlanningOr.isError())
    return enableMemoryPlanningOr.takeError();
  enableMemoryPlanning = enableMemoryPlanningOr.takeValue();

  auto newTorchInputSpecOr =
      fallibleGetSymbol(torchLib, "M_newTorchInputSpec", newTorchInputSpec);
  if (newTorchInputSpecOr.isError())
    return newTorchInputSpecOr.takeError();
  newTorchInputSpec = newTorchInputSpecOr.takeValue();

  auto freeTorchInputSpecOr =
      fallibleGetSymbol(torchLib, "M_freeTorchInputSpec", freeTorchInputSpec);
  if (freeTorchInputSpecOr.isError())
    return freeTorchInputSpecOr.takeError();
  freeTorchInputSpec = freeTorchInputSpecOr.takeValue();

  auto setTorchInputSpecsOr =
      fallibleGetSymbol(torchLib, "M_setTorchInputSpecs", setTorchInputSpecs);
  if (setTorchInputSpecsOr.isError())
    return setTorchInputSpecsOr.takeError();
  setTorchInputSpecs = setTorchInputSpecsOr.takeValue();

  auto setTorchLibraryPathOr =
      fallibleGetSymbol(torchLib, "M_setTorchLibraryPath", setTorchLibraryPath);
  if (setTorchLibraryPathOr.isError())
    return setTorchLibraryPathOr.takeError();
  setTorchLibraryPath = setTorchLibraryPathOr.takeValue();

  auto setTorchMetadataLibraryPathOr = fallibleGetSymbol(
      torchLib, "M_setTorchMetadataLibraryPath", setTorchMetadataLibraryPath);
  if (setTorchMetadataLibraryPathOr.isError())
    return setTorchMetadataLibraryPathOr.takeError();
  setTorchMetadataLibraryPath = setTorchMetadataLibraryPathOr.takeValue();

  auto setTorchLibraryPtrOr = fallibleGetSymbol(
      torchLib, "M_setTorchMetadataLibraryPtr", setTorchMetadataLibraryPtr);
  if (setTorchLibraryPathOr.isError())
    return setTorchLibraryPathOr.takeError();
  setTorchLibraryPath = setTorchLibraryPathOr.takeValue();

  GET_SYMBOL(compileModel, M_compileModel);
  GET_SYMBOL(waitForCompilation, M_waitForCompilation);
  GET_SYMBOL(freeCompiledModel, M_freeCompiledModel);

  GET_SYMBOL(initModel, M_initModel);
  GET_SYMBOL(waitForModel, M_waitForModel);
  GET_SYMBOL(freeModel, M_freeModel);
  GET_SYMBOL(getInputNames, M_getInputNames);
  GET_SYMBOL(getNumModelInputs, M_getNumModelInputs);
  GET_SYMBOL(getModelInputSpecByName, M_getModelInputSpecByName);
  GET_SYMBOL(getOutputNames, M_getOutputNames);
  GET_SYMBOL(getNumModelOutputs, M_getNumModelOutputs);
  GET_SYMBOL(getModelOutputSpecByName, M_getModelOutputSpecByName);
  GET_SYMBOL(freeTensorNameArray, M_freeTensorNameArray);

  GET_SYMBOL(getDtype, M_getDtype);
  GET_SYMBOL(sizeOf, M_sizeOf);

  GET_SYMBOL(newAsyncTensorMap, M_newAsyncTensorMap);
  GET_SYMBOL(freeAsyncTensorMap, M_freeAsyncTensorMap);
  GET_SYMBOL(borrowTensorInto, M_borrowTensorInto);
  GET_SYMBOL(getTensorByNameFrom, M_getTensorByNameFrom);
  GET_SYMBOL(getRank, M_getRank);
  GET_SYMBOL(isDynamicRanked, M_isDynamicRanked);
  GET_SYMBOL(getDimAt, M_getDimAt);
  GET_SYMBOL(getTensorType, M_getTensorType);
  GET_SYMBOL(getTensorNumElements, M_getTensorNumElements);
  GET_SYMBOL(getTensorNameAt, M_getTensorNameAt);
  GET_SYMBOL(getTensorData, M_getTensorData);
  GET_SYMBOL(newTensorSpec, M_newTensorSpec);
  GET_SYMBOL(getTensorSpec, M_getTensorSpec);
  GET_SYMBOL(freeTensorSpec, M_freeTensorSpec);
  GET_SYMBOL(freeTensor, M_freeTensor);

  GET_SYMBOL(getValueByNameFrom, M_getValueByNameFrom);
  GET_SYMBOL(getValueType, M_getValueType);
  GET_SYMBOL(getStringFromValue, M_getStringFromValue);
  GET_SYMBOL(getLongFromValue, M_getLongFromValue);
  GET_SYMBOL(getDoubleFromValue, M_getDoubleFromValue);
  GET_SYMBOL(getBoolFromValue, M_getBoolFromValue);
  GET_SYMBOL(getTensorFromValue, M_getTensorFromValue);
  GET_SYMBOL(getDictFromValue, M_getDictFromValue);
  GET_SYMBOL(getDictSize, M_getDictSize);
  GET_SYMBOL(getDictKey, M_getDictKey);
  GET_SYMBOL(getDictValue, M_getDictValue);
  GET_SYMBOL(freeDict, M_freeDict);
  GET_SYMBOL(getListFromValue, M_getListFromValue);
  GET_SYMBOL(getListSize, M_getListSize);
  GET_SYMBOL(getListValue, M_getListValue);
  GET_SYMBOL(freeList, M_freeList);
  GET_SYMBOL(getTupleFromValue, M_getTupleFromValue);
  GET_SYMBOL(getTupleSize, M_getTupleSize);
  GET_SYMBOL(getTupleValue, M_getTupleValue);
  GET_SYMBOL(freeTuple, M_freeTuple);
  GET_SYMBOL(freeValue, M_freeValue);

  GET_SYMBOL(executeModelSync, M_executeModelSync);

#undef STRINGIFY_SYMBOL
#undef GET_SYMBOL

  return success();
}
