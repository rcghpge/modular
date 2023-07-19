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
constexpr std::string_view modularTFExtensionLibraryName =
    "modular-framework-tf-ext.dll";
#elif defined(__APPLE__)
constexpr std::string_view modularFrameworkLibraryName =
    "libmodular-framework-common.dylib";
constexpr std::string_view modularTFExtensionLibraryName =
    "libmodular-framework-tf-ext.dylib";
#else
constexpr std::string_view modularFrameworkLibraryName =
    "libmodular-framework-common.so";
constexpr std::string_view modularTFExtensionLibraryName =
    "libmodular-framework-tf-ext.so";
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

static M::ErrorOr<llvm::sys::DynamicLibrary> fallibleModularExtLoad() {
  return fallibleLibraryLoad(modularTFExtensionLibraryName);
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

  auto tfLibOr = fallibleModularExtLoad();
  if (tfLibOr.isError())
    return tfLibOr.takeError();

  llvm::sys::DynamicLibrary dylib = dylibOr.takeValue();
  llvm::sys::DynamicLibrary tfLib = tfLibOr.takeValue();

#define STRINGIFY_SYMBOL(x) #x
#define GET_SYMBOL(VAR, SYMBOL_NAME)                                           \
  auto VAR##Or = fallibleGetSymbol(dylib, STRINGIFY_SYMBOL(SYMBOL_NAME), VAR); \
  if (VAR##Or.isError())                                                       \
    return VAR##Or.takeError();                                                \
  VAR = VAR##Or.takeValue();

  GET_SYMBOL(version, M_version);

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
  GET_SYMBOL(setModelPath, M_setModelPath);

  auto exportedNameOr =
      fallibleGetSymbol(tfLib, "M_setExportedName", setExportedName);
  if (exportedNameOr.isError())
    return exportedNameOr.takeError();
  setExportedName = exportedNameOr.takeValue();

  GET_SYMBOL(compileModel, M_compileModel);
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
  GET_SYMBOL(getDimAt, M_getDimAt);
  GET_SYMBOL(getTensorType, M_getTensorType);
  GET_SYMBOL(getTensorNumElements, M_getTensorNumElements);
  GET_SYMBOL(getTensorNameAt, M_getTensorNameAt);
  GET_SYMBOL(getTensorData, M_getTensorData);
  GET_SYMBOL(newTensorSpec, M_newTensorSpec);
  GET_SYMBOL(getTensorSpec, M_getTensorSpec);
  GET_SYMBOL(freeTensorSpec, M_freeTensorSpec);
  GET_SYMBOL(freeTensor, M_freeTensor);

  GET_SYMBOL(executeModelSync, M_executeModelSync);

#undef STRINGIFY_SYMBOL
#undef GET_SYMBOL

  return success();
}
