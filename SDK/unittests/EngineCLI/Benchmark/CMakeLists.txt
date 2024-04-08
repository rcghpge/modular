##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-genericml

  SUITE_NAME
  EngineCLIBenchmarkUnitTests

  TEST_NAME
  EngineCLIBenchmarkTests

  InputGeneratorTest.cpp
  MLPerfIntegrationTest.cpp
  NPYMetadataTest.cpp
  NPYReaderTest.cpp

  LINK_LIBS PRIVATE
  EngineCLIBenchmark
)
