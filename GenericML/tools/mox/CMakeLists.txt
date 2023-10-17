##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(mox
  Benchmark/benchmark.cpp
  mox.cpp

  LINK_LIBS PRIVATE
  MCrashReporting
  MSupport
  mlperf-loadgen
  modular-framework-common
)
