##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddPyEnv)

pyenv_setup_findpython3_variables("${MODULAR_PYTHON_API_PYTHON_VERSION}" "${MODULAR_PYTHON_API_PYTHON_ROOT}")
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pip install --root-user-action=ignore -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
  OUTPUT_QUIET
  RESULT_VARIABLE PIP_INSTALL_DEPS_RESULT
)

if(PIP_INSTALL_DEPS_RESULT)
  message(FATAL_ERROR "Installation of Python doctest dependencies failed.")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
  set(HOST_ARCH x86_64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(HOST_ARCH aarch64)
else()
  message(FATAL_ERROR "Unsupported target architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

file(GLOB DOCTESTS "${CMAKE_CURRENT_SOURCE_DIR}/*.ipynb")

# Add Python tests for the Modular Python API.
# DYLD_INSERT_LIBRARIES is passed as a CMake cache variable so that pytest can
# run when configured as ASAN.
# Without this, on MacOS with ASAN, pytest fails with an error.
add_test(NAME PythonDocTests
  COMMAND ${CMAKE_COMMAND} -E env
  ${PYTHON_API_TESTS_ASAN_ENV}
  ${Python3_EXECUTABLE} -m jupyter nbconvert --to notebook --execute --inplace ${DOCTESTS}
)

add_custom_target(check-PythonDocTests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R PythonDocTests
)

add_dependencies(check-PythonDocTests modular-pybind-ext)
add_dependencies(inference_engine_docs check-PythonDocTests)
