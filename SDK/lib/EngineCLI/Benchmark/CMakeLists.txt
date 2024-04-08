##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(
  EngineCLI BenchmarkOptions.td
  MANUAL_NAME max-benchmark
  MARKDOWN_NAME benchmark
)

add_genericml_library(EngineCLIBenchmark
  InputGenerationDefaults.cpp
  InputGenerator.cpp
  MLPerfIntegration.cpp
  Main.cpp
  NPYMetadata.cpp
  NPYReader.cpp
  Tensor.cpp

  DEPENDS
  EngineCLIDriverOptions

  LINK_LIBS
  PUBLIC
  EngineCLICommon
  MSupport
  MSupportDriver
  mlperf-loadgen
  modular-framework-common

  PRIVATE
  MSupportML
)

target_include_directories(EngineCLIBenchmark PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
