##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
  set(HOST_ARCH x86_64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(HOST_ARCH aarch64)
else()
  message(FATAL_ERROR "Unsupported target architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

file(GLOB DOCTESTS "${CMAKE_CURRENT_SOURCE_DIR}/*.ipynb")
set(PYTHON_DOCTESTS_ENV "TF_CPP_MIN_LOG_LEVEL=3")
set(DOCTESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Add Python tests for the Modular Python API.
add_test(NAME PythonDocTests
  COMMAND ${CMAKE_COMMAND} -E env
  ${PYTHON_DOCTESTS_ENV}
  ${Python3_EXECUTABLE} ${DOCTESTS_DIR}/runtests.py ${DOCTESTS_DIR}
)

add_custom_target(check-PythonDocTests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R PythonDocTests
)

add_dependencies(check-PythonDocTests modular-pybind-ext)
if (TARGET inference_engine_docs)
  add_dependencies(inference_engine_docs check-PythonDocTests)
endif()
