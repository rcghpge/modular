##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(modular-api-executor
  ModularAPIInterface.cpp
  modular-api-executor.cpp

  DEPENDS
  modular-framework-common
  modular-framework-onnx-ext
  modular-framework-torch-ext

  LINK_LIBS PUBLIC
  ExternalFrameworksSupport
  GenericMLExecutionDriver
  GenericMLSupport
  GenericMLSupportGlobals
  MSupportGlobals
  AsyncRTRuntime
  AsyncRTRuntimeGlobals
  AsyncRTSupport
  )

target_include_directories(modular-api-executor PUBLIC
  ${MODULAR_SOURCE_DIR}/ONNX/include/ONNX/API)

target_include_directories(modular-api-executor PUBLIC
  ${MODULAR_SOURCE_DIR}/PyTorch/include/PyTorch/API)

target_include_directories(modular-api-executor PUBLIC
  ${MODULAR_SOURCE_DIR}/GenericML/include/GenericML/API)
