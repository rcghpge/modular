##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(c-api-validator
  c-api-validator.cpp

  LINK_LIBS PUBLIC
  modular-framework-common
  modular-framework-tf-ext
  GenericMLExecutionDriver
)
