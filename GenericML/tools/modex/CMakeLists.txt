##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(modex
  modex.cpp

  LINK_LIBS PRIVATE
  MCrashReporting
  MSupport
  ModexBenchmark
)
