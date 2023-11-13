##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(max
  max.cpp

  LINK_LIBS PRIVATE
  BenchmarkCLI
  MCrashReporting
  MSupport
)
