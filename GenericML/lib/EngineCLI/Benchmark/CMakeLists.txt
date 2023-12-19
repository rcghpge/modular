##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# TODO: Switch to add_driver_option_tablegen once all subcommands hooked up.
set(LLVM_TARGET_DEFINITIONS BenchmarkOptions.td)
tablegen(LLVM BenchmarkOptions.inc -gen-opt-parser-defs)
tablegen(DRIVER BenchmarkOptionsHelpText.inc -gen-help-text)
tablegen(DRIVER max-benchmark.1 -gen-man-page)
add_public_tablegen_target(EngineCLIDriverBenchmarkOptions)

add_genericml_library(EngineCLIBenchmark
  InputGenerationDefaults.cpp
  InputGenerator.cpp
  MLPerfIntegration.cpp
  Main.cpp
  NPYMetadata.cpp
  NPYReader.cpp
  Tensor.cpp

  DEPENDS
  EngineCLIDriverBenchmarkOptions

  LINK_LIBS PUBLIC
  EngineCLICommon
  MSupport
  MSupportDriver
  mlperf-loadgen
  modular-framework-common
)

target_include_directories(EngineCLIBenchmark PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
