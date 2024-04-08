##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-genericml

  SUITE_NAME
  EngineCLICommonUnitTests

  TEST_NAME
  EngineCLICommonTests

  APIBridgeTest.cpp

  LINK_LIBS PRIVATE
  EngineCLICommon
)
