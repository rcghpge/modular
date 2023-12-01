##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(max-engine
  max-engine.cpp

  LINK_LIBS PRIVATE
  BenchmarkCLI
  MCrashReporting
  MSupport
  VisualizeCLI
)
