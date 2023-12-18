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
  APIBridge.cpp
  InputGenerationDefaults.cpp
  InputGenerationSchema.cpp
  InputGenerator.cpp
  MLPerfIntegration.cpp
  Main.cpp
  NPYMetadata.cpp
  NPYReader.cpp
  Tensor.cpp

  DEPENDS
  EngineCLIDriverBenchmarkOptions

  LINK_LIBS PUBLIC
  MSupport
  MSupportDriver
  mlperf-loadgen
  modular-framework-common
)

target_include_directories(EngineCLIBenchmark PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

# Can't go into LINK_LIBS above because of a layering violation.
# add_genericml_library checks that all targets listed in LINK_LIBS exist.  But
# at the time we're running this, we're in GenericML, and PyTorch's CMakeLists
# have not run yet, so this target doesn't exist.  target_link_libraries is a
# little more lax -- it resolves target names at generation time, after all
# CMakeLists have run.  modular-framework-torch-ext should move back up into
# LINK_LIBS once the MAX CLI is moved out of GenericML.
target_link_libraries(EngineCLIBenchmark PRIVATE modular-framework-torch-ext)
