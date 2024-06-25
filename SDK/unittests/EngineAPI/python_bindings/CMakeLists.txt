##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddPython)
include(RTTIExceptions)

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-sdk

  SUITE_NAME
  InferenceSessionUnitTest

  TEST_NAME
  InferenceSessionTest

  InferenceSessionTest.cpp
)

set_rtti_exceptions(llvm_gtest_main)
set_rtti_exceptions(llvm_gtest)
set_rtti_exceptions(InferenceSessionTest)

# find_package(Python3) has to be called before including pybind11 so that
# pybind11 picks up the correct Python development paths.
modular_find_python3_development()

include(AddPyBind11)

target_link_libraries(InferenceSessionTest PRIVATE
  ${Python3_LIBRARIES}
  modular-framework-common
  ModularFrameworkPython
  pybind11::embed
)

# NOTE: Ensure the CMake-managed Python interpreter is used.
# See the ScopedInterpreter note in InferenceSessionTest.cpp.
LIST(APPEND TEST_ENV "MODULAR_PYTHON_EXECUTABLE=${Python3_EXECUTABLE}")

set_tests_properties(InferenceSessionTest PROPERTIES ENVIRONMENT "${TEST_ENV}")
