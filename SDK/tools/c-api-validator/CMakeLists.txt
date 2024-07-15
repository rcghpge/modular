##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(c-api-validator
  c-api-validator.cpp

  LINK_LIBS PRIVATE
  modular-framework-common
  GenericMLExecutionDriver
  GenericMLSupportGlobals
  AsyncRTRuntimeGlobals
  MSupportGlobals
)
