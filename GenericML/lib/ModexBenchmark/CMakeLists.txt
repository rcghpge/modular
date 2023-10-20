##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_genericml_library(ModexBenchmark
  Main.cpp

  LINK_LIBS PUBLIC
  MSupport
  mlperf-loadgen
  modular-framework-common
)
