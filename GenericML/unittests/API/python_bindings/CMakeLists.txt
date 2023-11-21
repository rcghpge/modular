##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddPython)

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-genericml

  SUITE_NAME
  InferenceSessionUnitTest

  TEST_NAME
  InferenceSessionTest

  InferenceSessionTest.cpp
)

target_compile_options(llvm_gtest_main PRIVATE -frtti -fexceptions)
target_compile_options(llvm_gtest PRIVATE -frtti -fexceptions)
target_compile_options(InferenceSessionTest PRIVATE
   -frtti -fexceptions
)

# find_package(Python3) has to be called before including pybind11 so that
# pybind11 picks up the correct Python development paths.
modular_find_python3_development()

include(AddPyBind11)

target_link_libraries(InferenceSessionTest PRIVATE
  ${Python3_LIBRARIES}
  modular-framework-common
  pybind11::embed
)

LIST(APPEND TEST_ENV "ASAN_OPTIONS=detect_leaks=0,abort_on_error=1,disable_coredump=0")

# NOTE: Ensure the CMake-managed Python interpreter is used.
# See the ScopedInterpreter note in InferenceSessionTest.cpp.
LIST(APPEND TEST_ENV "MODULAR_PYTHON_EXECUTABLE=${Python3_EXECUTABLE}")

set_tests_properties(InferenceSessionTest PROPERTIES ENVIRONMENT "${TEST_ENV}")
