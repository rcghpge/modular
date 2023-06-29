##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(modular-api-executor
  modular-api-executor.cpp

  LINK_LIBS PUBLIC
  modular-framework-common
  modular-framework-tf-ext
  ExternalFrameworksSupport
  GenericMLExecutionDriver
  GenericMLSupport
  GenericMLSupportGlobals
  LLCLRuntime
  LLCLRuntimeGlobals
  LLCLSupport
)
