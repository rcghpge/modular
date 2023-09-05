##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(modbench
  modbench.cpp

  LINK_LIBS PRIVATE
  MCrashReporting
  MSupport
  modular-framework-common
)
