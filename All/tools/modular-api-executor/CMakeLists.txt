##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(modular-api-executor
  ModularAPIInterface.cpp
  modular-api-executor.cpp

  LINK_LIBS PUBLIC
  ExternalFrameworksSupport
  GenericMLExecutionDriver
  GenericMLSupport
  GenericMLSupportGlobals
  MSupportGlobals
  LLCLRuntime
  LLCLRuntimeGlobals
  LLCLSupport
  )


target_include_directories(modular-api-executor PUBLIC
  ${MODULAR_SOURCE_DIR}/TensorFlow/include/TensorFlow/API)

target_include_directories(modular-api-executor PUBLIC
  ${MODULAR_SOURCE_DIR}/GenericML/include/GenericML/API)
