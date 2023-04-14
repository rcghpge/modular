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

# Set up ASAN only if enabled since otherwise errors occur in non-ASAN builds.
if(LLVM_USE_SANITIZER)
  # Set environment variables to inject the ASAN runtime library since the
  # Python interpreter is not linked with ASAN.
  # Also, set LSAN_OPTIONS=detect_leaks=0 to avoid leaks exposed by Python.
  set(PYTHON_API_TESTS_ASAN_ENV "LSAN_OPTIONS=detect_leaks=0")

  # When ASAN is enabled, shared libraries such as libtf.so loaded at runtime
  # can be loaded by the ASAN runtime.
  # This causes an error because those shared libraries are not in the ASAN
  # runtime's RUNPATH.
  # As a workaround, add the path to the shared libraries to LD_LIBRARY_PATH
  # when ASAN is enabled.
  # TODO(#10762): Fix this with a more robust solution that does not rely on
  # LD_LIBRARY_PATH.
  set(LIBRARY_PATH_VAR_NAME
    $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Darwin>:DYLD_LIBRARY_PATH>$<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:LD_LIBRARY_PATH>)
  list(APPEND PYTHON_API_TESTS_ASAN_ENV
    "${LIBRARY_PATH_VAR_NAME}=$<TARGET_FILE_DIR:stock-tensorflow::stock-tensorflow>:$<TARGET_FILE_DIR:tf>")

  if(UNIX)
    if(APPLE)
      set(INJECT_LIB_ENV_VAR DYLD_INSERT_LIBRARIES)
      set(ASAN_RT_LIB_NAME libclang_rt.asan_osx_dynamic.dylib)
    else()
      set(INJECT_LIB_ENV_VAR LD_PRELOAD)

      if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
        set(ASAN_RT_LIB_NAME libclang_rt.asan-${HOST_ARCH}.so)
      elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
        set(ASAN_RT_LIB_NAME libasan.so)
      endif()
    endif()

    execute_process(
      COMMAND ${CMAKE_CXX_COMPILER} --print-file-name=${ASAN_RT_LIB_NAME}
      OUTPUT_VARIABLE ASAN_RT_LIB_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE PRINT_FILENAME_RESULT
    )

    if(PRINT_FILENAME_RESULT)
      message(FATAL_ERROR "Failed to get ASAN runtime library path.")
    endif()

    list(APPEND PYTHON_API_TESTS_ASAN_ENV "${INJECT_LIB_ENV_VAR}=${ASAN_RT_LIB_PATH}")
  endif()
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
