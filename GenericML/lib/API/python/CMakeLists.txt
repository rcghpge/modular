##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddPython)

# Currently not supporting Python wheel packaging on Windows.
if(NOT UNIX)
  message(WARNING "Wheel packaging is not supported on Windows yet.")
  return()
endif()

if(${DISABLE_ALL_EXTERNAL})
  message(WARNING "Cannot build Python C++ bindings since PyBind11 is not available.")
  return()
endif()

modular_find_python3_development()

include(AddPyBind11)

# Use pyproject.toml as a single source of truth for the version of the Modular
# Python framework.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import tomli; f = open('${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml', 'rb'); proj = tomli.load(f)['project']; print(proj['name'].replace('-', '_') + '-' + proj['version'])"
  RESULT_VARIABLE TOMLI_RESULT
  OUTPUT_VARIABLE MODULAR_PYPROJECT_NAME_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
if(TOMLI_RESULT)
  message(FATAL_ERROR "Failed to parse pyproject.toml for Modular python bindings.")
endif()

# Package the built extension into a wheel.
set(WHEEL_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(UNPACKED_WHEEL ${CMAKE_CURRENT_BINARY_DIR}/${MODULAR_PYPROJECT_NAME_VERSION})

# Get the expected name of the built Python wheel file before building it.
# This command is expected to error, which is intentionally ignored.
execute_process(
  COMMAND ${Python3_EXECUTABLE} setup.py --dry-run --quiet bdist_wheel
  OUTPUT_VARIABLE BDIST_WHEEL_OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  ERROR_QUIET
)
# For earlier versions of wheel/setuptools, setup.py emits other informational
# messages to stdout even with the --quiet flag.
# So ignore any additional output from setup.py after the wheel name.
string(REGEX MATCH "^[^\n]+" WHEEL_BASENAME "${BDIST_WHEEL_OUTPUT}")

# Add a custom target to build and install the Python bindings and wheel.
# Individual pybind11 modules are expected to be dependencies of
# modular-engine.
add_custom_target(modular-pybind-ext ALL
  COMMAND ${Python3_EXECUTABLE} -m pip install --quiet --force-reinstall ${WHEEL_DIR}/${WHEEL_BASENAME}.whl
  DEPENDS ${WHEEL_DIR}/${WHEEL_BASENAME}.whl
  COMMENT "Installing Modular Python API wheel."
)
add_dependencies(modular-pybind-ext modular-engine)
add_subdirectory(bindings/modular/engine)

set(MODULAR_FRAMEWORK_LIBS $<TARGET_FILE:monnx> $<TARGET_FILE:tfl>)

# The wheel is built in the build directory.
# However, setuptools is unaware of the pybind module since CMake built that.
# Therefore we have to put the pybind module shared library and all other
# needed shared libraries into the wheel.
# This way everything is self-contained in the wheel and will still work after
# the wheel is pip installed.
# TODO(#7439): /dev/null is not portable to Windows.
if(STOCK_TENSORFLOW_FOUND)
  set(EXTERNAL_TF_EXES
      "${MODULAR_DERIVED_DIR}/third-party/bin/external-tf-translate;"
      "${MODULAR_DERIVED_DIR}/third-party/bin/external-tf-opt")
  list(APPEND MODULAR_FRAMEWORK_LIBS $<TARGET_FILE:tf>)
endif()

set(MODULAR_PACKED_WHEEL_NAME ${WHEEL_DIR}/${WHEEL_BASENAME}.whl)

set(modular_python_package_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/modular/engine/api.py
  ${CMAKE_CURRENT_SOURCE_DIR}/modular/engine/core.pyi
)
add_custom_command(OUTPUT ${MODULAR_PACKED_WHEEL_NAME}
  COMMAND ${Python3_EXECUTABLE} -m build --wheel --outdir ${WHEEL_DIR} > /dev/null
  COMMAND ${Python3_EXECUTABLE} -m wheel unpack ${WHEEL_DIR}/${WHEEL_BASENAME}.whl --dest ${WHEEL_DIR} > /dev/null
  COMMAND ${CMAKE_COMMAND} -E make_directory ${UNPACKED_WHEEL}/modular/engine
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:core> ${UNPACKED_WHEEL}/modular/engine/
  COMMAND ${CMAKE_COMMAND} -E make_directory ${UNPACKED_WHEEL}/bin
  # NOTE: Use a temporary file so that the copy command succeeds when NOT
  # STOCK_TENSORFLOW_FOUND.
  COMMAND ${CMAKE_COMMAND} -E touch ${UNPACKED_WHEEL}/bin/temp
  COMMAND ${CMAKE_COMMAND} -E copy ${UNPACKED_WHEEL}/bin/temp ${EXTERNAL_TF_EXES} ${UNPACKED_WHEEL}/bin
  COMMAND ${CMAKE_COMMAND} -E remove ${UNPACKED_WHEEL}/bin/temp
  COMMAND ${Python3_EXECUTABLE} -m wheel pack ${UNPACKED_WHEEL} --dest ${WHEEL_DIR} > /dev/null
  COMMAND ${CMAKE_COMMAND} -E rename ${WHEEL_DIR}/${WHEEL_BASENAME}.whl ${MODULAR_PACKED_WHEEL_NAME}
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/build
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/modular_engine.egg-info
  BYPRODUCTS ${UNPACKED_WHEEL}
  DEPENDS $<TARGET_FILE:core> ${MODULAR_FRAMEWORK_LIBS} ${EXTERNAL_TF_EXES} ${modular_python_package_sources}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Packing wheel with bindings and shared library dependencies."
  VERBATIM
)

# Note: the `BYPRODUCTS` specification above is correct, but the generated
# `clean` target won't be correct (i.e. CMake will issue a hard error at
# configure time). Account for this by explicitly setting the
# `ADDITIONAL_CLEAN_FILES` on the directory which makes the generated `clean`
# target work correctly.
set_target_properties(modular-engine
  PROPERTIES
  ADDITIONAL_CLEAN_FILES ${UNPACKED_WHEEL}
)

add_subdirectory(docs)
add_subdirectory(notebooks)

# Set up wheel directory structure by copying the pyproject.toml and setup.py
# files needed for setuptools to build the wheel.
install(FILES pyproject.toml setup.py
  DESTINATION modular-engine-wheel
)

# Also copy the `modular.engine` Python files into the wheel.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/modular
  DESTINATION modular-engine-wheel
  FILES_MATCHING
  PATTERN "*.py*"
  PATTERN "py.typed"
)

# Finally, install the built Python extension module shared library into the
# wheel.
install(TARGETS core
  DESTINATION modular-engine-wheel/modular/engine
)
