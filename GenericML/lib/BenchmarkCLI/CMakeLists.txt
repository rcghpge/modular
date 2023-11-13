##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_genericml_library(BenchmarkCLI
  InputGenerationSchema.cpp
  InputGenerator.cpp
  MLPerfIntegration.cpp
  Main.cpp

  LINK_LIBS PUBLIC
  MSupport
  mlperf-loadgen
  modular-framework-common
)
