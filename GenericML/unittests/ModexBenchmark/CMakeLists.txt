##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-genericml

  SUITE_NAME
  ModexBenchmarkUnitTests

  TEST_NAME
  ModexBenchmarkTests

  InputGeneratorTest.cpp
  MLPerfIntegrationTest.cpp

  LINK_LIBS PRIVATE
  ModexBenchmark
)
